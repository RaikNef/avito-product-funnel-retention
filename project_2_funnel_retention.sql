-- ============================================================================
-- AVITO PRODUCT ANALYTICS | ПРОЕКТ 2: ПРЕДОБРАБОТКА КЛИКСТРИМА (EDA & DQ),
-- ПОШАГОВАЯ ВОРОНКА CJM И КОГОРТНЫЙ RETENTION (D1/D3/D7/D14) НА СХЕМЕ REL-AVITO
-- Таблицы: SearchInfo, VisitStream, PhoneRequestsStream, AdsInfo, UserInfo
-- СУБД: Vertica / PostgreSQL
-- Автор: Нефедов Родион (GitHub: github.com/RaikNef)
-- ============================================================================

WITH
-- 1. Выявление аномальных бот-IP (>99.95% перцентиля просмотров карточек с одного IPID)
ip_activity_stats AS (
    SELECT
        IPID,
        COUNT(*) AS ip_views_cnt
    FROM VisitStream
    WHERE IPID IS NOT NULL
    GROUP BY IPID
),

bot_ip_blacklist AS (
    SELECT
        IPID
    FROM (
        SELECT
            IPID,
            ip_views_cnt,
            PERCENTILE_CONT(0.9995) WITHIN GROUP (ORDER BY ip_views_cnt) OVER () AS p9995_views
        FROM ip_activity_stats
    ) t
    WHERE ip_views_cnt > p9995_views
),

-- 2. Очистка потока поисковых сессий (SearchInfo: отсев пустых UserID, бот-IP и дребезга < 3 сек)
clean_searches AS (
    SELECT
        UserID,
        SearchID,
        SearchDate AS event_ts,
        CAST(SearchDate AS DATE) AS event_date,
        IsUserLoggedOn,
        CategoryID,
        LocationID
    FROM (
        SELECT
            s.UserID,
            s.SearchID,
            s.SearchDate,
            s.IsUserLoggedOn,
            s.CategoryID,
            s.LocationID,
            LAG(s.SearchDate) OVER (
                PARTITION BY s.UserID, s.CategoryID
                ORDER BY s.SearchDate ASC
            ) AS prev_search_ts
        FROM SearchInfo s
        LEFT JOIN bot_ip_blacklist b ON s.IPID = b.IPID
        WHERE s.UserID IS NOT NULL
          AND b.IPID IS NULL
    ) annotated_s
    WHERE prev_search_ts IS NULL
       OR EXTRACT(EPOCH FROM (SearchDate - prev_search_ts)) >= 3
),

-- 3. Очистка кликстрима просмотров карточек (VisitStream: отсев бот-IP и дублей кликов < 3 сек)
clean_visits AS (
    SELECT
        UserID,
        AdID,
        ViewDate AS event_ts,
        CAST(ViewDate AS DATE) AS event_date
    FROM (
        SELECT
            v.UserID,
            v.AdID,
            v.ViewDate,
            LAG(v.ViewDate) OVER (
                PARTITION BY v.UserID, v.AdID
                ORDER BY v.ViewDate ASC
            ) AS prev_view_ts
        FROM VisitStream v
        LEFT JOIN bot_ip_blacklist b ON v.IPID = b.IPID
        WHERE v.UserID IS NOT NULL
          AND b.IPID IS NULL
    ) annotated_v
    WHERE prev_view_ts IS NULL
       OR EXTRACT(EPOCH FROM (ViewDate - prev_view_ts)) >= 3
),

-- 4. Очистка целевых конверсий в запрос телефона продавца (PhoneRequestsStream + AdsInfo.Price)
clean_phone_requests AS (
    SELECT
        p.UserID,
        p.AdID,
        p.PhoneRequestDate AS event_ts,
        CAST(p.PhoneRequestDate AS DATE) AS event_date,
        CASE
            WHEN a.Price BETWEEN 100 AND 500000 THEN a.Price
            ELSE NULL
        END AS clean_ad_price_rub
    FROM (
        SELECT
            pr.UserID,
            pr.AdID,
            pr.PhoneRequestDate,
            LAG(pr.PhoneRequestDate) OVER (
                PARTITION BY pr.UserID, pr.AdID
                ORDER BY pr.PhoneRequestDate ASC
            ) AS prev_phone_ts
        FROM PhoneRequestsStream pr
        LEFT JOIN bot_ip_blacklist b ON pr.IPID = b.IPID
        WHERE pr.UserID IS NOT NULL
          AND b.IPID IS NULL
    ) p
    LEFT JOIN AdsInfo a ON p.AdID = a.AdID
    WHERE p.prev_phone_ts IS NULL
       OR EXTRACT(EPOCH FROM (p.PhoneRequestDate - p.prev_phone_ts)) >= 3
),

-- 5. Объединение дневной активности пользователей для расчета когортного Retention (D1, D3, D7, D14)
user_daily_activity AS (
    SELECT UserID, event_date FROM clean_searches
    UNION
    SELECT UserID, event_date FROM clean_visits
    UNION
    SELECT UserID, event_date FROM clean_phone_requests
),

user_cohorts AS (
    SELECT
        UserID,
        MIN(event_date) AS cohort_date
    FROM user_daily_activity
    GROUP BY UserID
),

user_retention_flags AS (
    SELECT
        c.UserID,
        c.cohort_date,
        MAX(CASE WHEN (a.event_date - c.cohort_date) = 1  THEN 1 ELSE 0 END) AS ret_d1,
        MAX(CASE WHEN (a.event_date - c.cohort_date) = 3  THEN 1 ELSE 0 END) AS ret_d3,
        MAX(CASE WHEN (a.event_date - c.cohort_date) = 7  THEN 1 ELSE 0 END) AS ret_d7,
        MAX(CASE WHEN (a.event_date - c.cohort_date) = 14 THEN 1 ELSE 0 END) AS ret_d14
    FROM user_cohorts c
    INNER JOIN user_daily_activity a ON c.UserID = a.UserID
    GROUP BY c.UserID, c.cohort_date
),

-- 6. Сборка пользовательской витрины воронки CJM и товарных метрик
user_funnel_mart AS (
    SELECT
        u.UserID,
        u.UserAgentOSID,
        u.UserDeviceID,
        COALESCE(s.is_logged_on, 0) AS is_logged_on,
        COALESCE(s.searches_cnt, 0) AS searches_cnt,
        COALESCE(v.views_cnt, 0)    AS views_cnt,
        COALESCE(p.contacts_cnt, 0) AS contacts_cnt,
        p.avg_contact_price_rub,
        COALESCE(p.sum_contact_price_rub, 0.0) AS sum_contact_price_rub,
        1 AS step_1_search,
        CASE WHEN COALESCE(v.views_cnt, 0) >= 1 THEN 1 ELSE 0 END AS step_2_view_card,
        CASE WHEN COALESCE(v.views_cnt, 0) >= 5 THEN 1 ELSE 0 END AS step_3_deep_view,
        CASE WHEN COALESCE(v.views_cnt, 0) >= 5 AND COALESCE(p.contacts_cnt, 0) >= 1 THEN 1 ELSE 0 END AS step_4_phone_contact
    FROM UserInfo u
    INNER JOIN (
        SELECT
            UserID,
            COUNT(*) AS searches_cnt,
            MAX(IsUserLoggedOn) AS is_logged_on
        FROM clean_searches
        GROUP BY UserID
    ) s ON u.UserID = s.UserID
    LEFT JOIN (
        SELECT
            UserID,
            COUNT(*) AS views_cnt
        FROM clean_visits
        GROUP BY UserID
    ) v ON u.UserID = v.UserID
    LEFT JOIN (
        SELECT
            UserID,
            COUNT(*) AS contacts_cnt,
            AVG(clean_ad_price_rub) AS avg_contact_price_rub,
            SUM(clean_ad_price_rub) AS sum_contact_price_rub
        FROM clean_phone_requests
        GROUP BY UserID
    ) p ON u.UserID = p.UserID
)

-- Финальный запрос для дашборда в Redash: пошаговая воронка, Retention и цена объявлений по сегментам
SELECT
    CASE
        WHEN m.is_logged_on >= 1 THEN 'Logged-In Users'
        ELSE 'Guest Users'
    END AS user_auth_segment,
    COUNT(m.UserID)                                                        AS s1_search_users,
    SUM(m.step_2_view_card)                                                AS s2_view_users,
    ROUND(100.0 * SUM(m.step_2_view_card) / NULLIF(COUNT(m.UserID), 0), 2) AS step_cr_s1_to_s2_pct,
    SUM(m.step_3_deep_view)                                                AS s3_deep_view_users,
    ROUND(100.0 * SUM(m.step_3_deep_view) / NULLIF(SUM(m.step_2_view_card), 0), 2) AS step_cr_s2_to_s3_pct,
    SUM(m.step_4_phone_contact)                                            AS s4_contact_users,
    ROUND(100.0 * SUM(m.step_4_phone_contact) / NULLIF(SUM(m.step_3_deep_view), 0), 2) AS step_cr_s3_to_s4_pct,
    ROUND(100.0 * SUM(m.step_4_phone_contact) / NULLIF(COUNT(m.UserID), 0), 2) AS overall_cr_pct,
    ROUND(100.0 * AVG(CASE WHEN r.cohort_date <= DATE '2015-05-19' THEN r.ret_d1  END), 2) AS retention_d1_pct,
    ROUND(100.0 * AVG(CASE WHEN r.cohort_date <= DATE '2015-05-17' THEN r.ret_d3  END), 2) AS retention_d3_pct,
    ROUND(100.0 * AVG(CASE WHEN r.cohort_date <= DATE '2015-05-13' THEN r.ret_d7  END), 2) AS retention_d7_pct,
    ROUND(100.0 * AVG(CASE WHEN r.cohort_date <= DATE '2015-05-06' THEN r.ret_d14 END), 2) AS retention_d14_pct,
    ROUND(AVG(m.avg_contact_price_rub), 2)                                 AS avg_contacted_ad_price_rub
FROM user_funnel_mart m
LEFT JOIN user_retention_flags r ON m.UserID = r.UserID
GROUP BY 1
ORDER BY s1_search_users DESC;
