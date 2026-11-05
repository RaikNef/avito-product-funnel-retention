#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проект 2: Разведочный анализ (EDA) и предобработка кликстрима Авито, пошаговая воронка CJM,
когортный Retention (D1, D3, D7, D14) и сегментный анализ конверсии и стоимости объявлений
Данные: Реальный продакшен-датасет Avito Context Ad Clicks (rel-avito, 2015-04-25 — 2015-05-20):
        - SearchInfo.parquet (2 579 289 поисковых сессий)
        - VisitStream.parquet (6 454 562 просмотра карточек объявлений)
        - PhoneRequestsStream.parquet (302 974 запроса телефона продавца)
        - SearchStream.parquet (9 254 702 показа объявлений в выдаче: обычные, VIP, контекстные)
        - AdsInfo.parquet (5 960 558 объявлений с реальными ценами Price в рублях)
        - UserInfo.parquet (98 250 профилей пользователей)
Автор: Нефедов Родион (GitHub: github.com/RaikNef)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def resolve_data_dir() -> str:
    """Находит директорию с реальными таблицами Avito (avito_real_data/db)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "avito_real_data", "db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "avito_real_data", "db"),
        r"d:\Work\resume\avito_real_data\db",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "UserInfo.parquet")):
            return os.path.abspath(path)
    raise FileNotFoundError("Не найдена папка avito_real_data/db с parquet-таблицами Авито.")


def analyze_real_avito_funnel_and_retention(db_dir: str) -> dict:
    """
    Выполняет полный цикл EDA, очистки сырых логов, построения пошаговой воронки CJM,
    когортного Retention (D1, D3, D7, D14) и сегментного анализа на реальных таблицах Авито.
    """
    users = pd.read_parquet(
        os.path.join(db_dir, "UserInfo.parquet"),
        columns=["UserID", "UserAgentOSID", "UserDeviceID"],
    )
    searches = pd.read_parquet(
        os.path.join(db_dir, "SearchInfo.parquet"),
        columns=["UserID", "SearchID", "SearchDate", "IPID", "IsUserLoggedOn", "CategoryID"],
    )
    visits = pd.read_parquet(
        os.path.join(db_dir, "VisitStream.parquet"),
        columns=["UserID", "IPID", "AdID", "ViewDate"],
    )
    phones = pd.read_parquet(
        os.path.join(db_dir, "PhoneRequestsStream.parquet"),
        columns=["UserID", "IPID", "AdID", "PhoneRequestDate"],
    )
    ads = pd.read_parquet(
        os.path.join(db_dir, "AdsInfo.parquet"),
        columns=["AdID", "Price", "IsContext", "CategoryID"],
    )
    stream = pd.read_parquet(
        os.path.join(db_dir, "SearchStream.parquet"),
        columns=["ObjectType", "IsClick", "HistCTR"],
    )

    # 1. Аудит качества данных (Data Quality & EDA) на объединенном пользовательском потоке
    #    (SearchInfo + VisitStream + PhoneRequestsStream = 9 336 825 событий)
    raw_searches_cnt = len(searches)
    raw_visits_cnt = len(visits)
    raw_phones_cnt = len(phones)
    raw_total_events = raw_searches_cnt + raw_visits_cnt + raw_phones_cnt

    # 1.1 Записи без UserID в поисковых сессиях
    null_user_searches = int(searches["UserID"].isna().sum())
    si_valid = searches[searches["UserID"].notna()].copy()
    si_valid["UserID"] = si_valid["UserID"].astype("int64")

    # 1.2 Выявление аномальной бот-активности по IPID (>99.95-го перцентиля просмотров с одного IP)
    ip_view_counts = visits.groupby("IPID").size()
    bot_ip_threshold = float(ip_view_counts.quantile(0.9995))
    bot_ips = set(ip_view_counts[ip_view_counts > bot_ip_threshold].index)

    bot_si_mask = si_valid["IPID"].isin(bot_ips)
    bot_vs_mask = visits["IPID"].isin(bot_ips)
    bot_pr_mask = phones["IPID"].isin(bot_ips)
    bot_events_total = int(bot_si_mask.sum() + bot_vs_mask.sum() + bot_pr_mask.sum())

    si_nobot = si_valid[~bot_si_mask].sort_values(["UserID", "CategoryID", "SearchDate"])
    vs_nobot = visits[~bot_vs_mask].sort_values(["UserID", "AdID", "ViewDate"])
    pr_nobot = phones[~bot_pr_mask].sort_values(["UserID", "AdID", "PhoneRequestDate"])

    # 1.3 Фильтрация клиентского дребезга (< 3 секунд между идентичными событиями одного UserID)
    si_diff = si_nobot.groupby(["UserID", "CategoryID"])["SearchDate"].diff().dt.total_seconds()
    vs_diff = vs_nobot.groupby(["UserID", "AdID"])["ViewDate"].diff().dt.total_seconds()
    pr_diff = pr_nobot.groupby(["UserID", "AdID"])["PhoneRequestDate"].diff().dt.total_seconds()

    si_bounce_mask = si_diff.notna() & (si_diff < 3.0)
    vs_bounce_mask = vs_diff.notna() & (vs_diff < 3.0)
    pr_bounce_mask = pr_diff.notna() & (pr_diff < 3.0)
    bounce_events_total = int(si_bounce_mask.sum() + vs_bounce_mask.sum() + pr_bounce_mask.sum())

    si_clean = si_nobot[~si_bounce_mask].copy()
    vs_clean = vs_nobot[~vs_bounce_mask].copy()
    pr_clean = pr_nobot[~pr_bounce_mask].copy()

    clean_total_events = len(si_clean) + len(vs_clean) + len(pr_clean)
    filtered_total_events = raw_total_events - clean_total_events
    filtered_pct = (filtered_total_events / raw_total_events) * 100.0

    # Проверка аномалий цен в AdsInfo.parquet
    invalid_price_ads = int((ads["Price"].isna() | (ads["Price"] <= 0)).sum())
    valid_price_ads = int((ads["Price"] > 0).sum())

    # 2. Построение пошаговой продуктовой воронки CJM по очищенным логам
    # Шаг 1: Поисковая сессия (SearchInfo)
    # Шаг 2: Переход к просмотру карточки объявления (VisitStream, >= 1 просмотра)
    # Шаг 3: Глубокий просмотр и сравнение предложений (>= 5 просмотров карточек в VisitStream)
    # Шаг 4: Целевой контакт — запрос номера телефона продавца (PhoneRequestsStream)
    u_search = si_clean.groupby("UserID").agg(
        searches_cnt=("SearchID", "count"),
        is_logged_on=("IsUserLoggedOn", "max"),
        first_search_dt=("SearchDate", "min"),
    )
    u_visit = vs_clean.groupby("UserID").agg(
        views_cnt=("AdID", "count"),
        unique_ads_viewed=("AdID", "nunique"),
    )

    # Привязка реальных рублёвых цен объявлений (винзоризация 100 руб. .. 500 000 руб. для товарных категорий)
    ads_valid_price = ads[(ads["Price"] >= 100.0) & (ads["Price"] <= 500000.0)][["AdID", "Price"]]
    pr_with_price = pr_clean.merge(ads_valid_price, on="AdID", how="left")
    u_phone = pr_with_price.groupby("UserID").agg(
        contacts_cnt=("AdID", "count"),
        contacted_price_sum_rub=("Price", "sum"),
        contacted_price_mean_rub=("Price", "mean"),
    )

    user_mart = (
        users.merge(u_search, on="UserID", how="inner")
        .merge(u_visit, on="UserID", how="left")
        .merge(u_phone, on="UserID", how="left")
    )
    user_mart["views_cnt"] = user_mart["views_cnt"].fillna(0).astype(int)
    user_mart["unique_ads_viewed"] = user_mart["unique_ads_viewed"].fillna(0).astype(int)
    user_mart["contacts_cnt"] = user_mart["contacts_cnt"].fillna(0).astype(int)
    user_mart["contacted_price_sum_rub"] = user_mart["contacted_price_sum_rub"].fillna(0.0)

    user_mart["step_1_search"] = 1
    user_mart["step_2_view"] = (user_mart["views_cnt"] >= 1).astype(int)
    user_mart["step_3_deep_view"] = (user_mart["views_cnt"] >= 5).astype(int)
    user_mart["step_4_contact"] = ((user_mart["step_3_deep_view"] == 1) & (user_mart["contacts_cnt"] >= 1)).astype(int)

    s1_users = int(user_mart["step_1_search"].sum())
    s2_users = int(user_mart["step_2_view"].sum())
    s3_users = int(user_mart["step_3_deep_view"].sum())
    s4_users = int(user_mart["step_4_contact"].sum())

    funnel_steps = [
        {
            "step": "1. Поисковая сессия (SearchInfo)",
            "users": s1_users,
            "step_cr_pct": 100.0,
            "dropoff_pct": 0.0,
        },
        {
            "step": "2. Просмотр карточки объявления (VisitStream >= 1)",
            "users": s2_users,
            "step_cr_pct": round(100.0 * s2_users / s1_users, 2),
            "dropoff_pct": round(100.0 * (1.0 - s2_users / s1_users), 2),
        },
        {
            "step": "3. Глубокий просмотр / сравнение (VisitStream >= 5)",
            "users": s3_users,
            "step_cr_pct": round(100.0 * s3_users / s2_users, 2),
            "dropoff_pct": round(100.0 * (1.0 - s3_users / s2_users), 2),
        },
        {
            "step": "4. Запрос телефона продавца (PhoneRequestsStream)",
            "users": s4_users,
            "step_cr_pct": round(100.0 * s4_users / s3_users, 2),
            "dropoff_pct": round(100.0 * (1.0 - s4_users / s3_users), 2),
        },
    ]
    overall_cr_pct = round(100.0 * s4_users / s1_users, 2)

    # 3. Расчет реального когортного Retention (D1, D3, D7, D14) на 26-дневном интервале (2015-04-25 .. 2015-05-20)
    # Объединяем уникальные дни активности пользователя по всем очищенным потокам (поиск, просмотры, контакты)
    si_days = si_clean[["UserID"]].assign(act_date=si_clean["SearchDate"].dt.floor("D")).drop_duplicates()
    vs_days = vs_clean[["UserID"]].assign(act_date=vs_clean["ViewDate"].dt.floor("D")).drop_duplicates()
    pr_days = pr_clean[["UserID"]].assign(act_date=pr_clean["PhoneRequestDate"].dt.floor("D")).drop_duplicates()

    user_days = pd.concat([si_days, vs_days, pr_days], ignore_index=True).drop_duplicates()
    cohorts = user_days.groupby("UserID", as_index=False)["act_date"].min().rename(columns={"act_date": "cohort_date"})
    user_days = user_days.merge(cohorts, on="UserID", how="inner")
    user_days["day_offset"] = (user_days["act_date"] - user_days["cohort_date"]).dt.days

    max_log_date = pd.Timestamp("2015-05-20")
    retention_overall = {}
    for d in [1, 3, 7, 14]:
        # Учитываем когорты, у которых наступил горизонт наблюдения d дней до конца периода логов
        eligible_cohorts = cohorts[cohorts["cohort_date"] <= (max_log_date - pd.Timedelta(days=d))]
        eligible_users = set(eligible_cohorts["UserID"])
        returned_users = set(
            user_days[(user_days["day_offset"] == d) & (user_days["UserID"].isin(eligible_users))]["UserID"]
        )
        retention_overall[f"D{d}_pct"] = round(100.0 * len(returned_users) / len(eligible_users), 2)

    # 4. Сегментный анализ: Авторизованные vs. Неавторизованные пользователи и основные типы ОС/устройств
    user_mart = user_mart.merge(cohorts, on="UserID", how="left")
    d7_returned_set = set(user_days[user_days["day_offset"] == 7]["UserID"])
    d1_returned_set = set(user_days[user_days["day_offset"] == 1]["UserID"])
    user_mart["ret_d1"] = user_mart["UserID"].isin(d1_returned_set).astype(int)
    user_mart["ret_d7"] = user_mart["UserID"].isin(d7_returned_set).astype(int)

    # Топ-2 мобильные/десктопные ОС по числу пользователей
    top_os_ids = user_mart["UserAgentOSID"].value_counts().head(2).index.tolist()

    def build_segment_label(row):
        auth = "Logged-In" if row["is_logged_on"] >= 1.0 else "Guest"
        os_group = "Desktop OS (Primary)" if row["UserAgentOSID"] == top_os_ids[0] else "Mobile / Other OS"
        return f"{auth} | {os_group}"

    user_mart["segment"] = [
        ("Авторизованные | Основная ОС" if (log >= 1.0 and os_id == top_os_ids[0])
         else "Авторизованные | Мобильные и др. ОС" if log >= 1.0
         else "Гости | Основная ОС" if os_id == top_os_ids[0]
         else "Гости | Мобильные и др. ОС")
        for log, os_id in zip(user_mart["is_logged_on"], user_mart["UserAgentOSID"])
    ]

    segment_rows = []
    for seg_name, grp in user_mart.groupby("segment"):
        u_cnt = len(grp)
        contact_users = int(grp["step_4_contact"].sum())
        contacts_total = int(grp["contacts_cnt"].sum())
        avg_item_price = float(
            grp.loc[grp["contacted_price_mean_rub"].notna(), "contacted_price_mean_rub"].mean()
        )
        gmv_per_user = float(grp["contacted_price_sum_rub"].mean())

        elig_d1 = grp[grp["cohort_date"] <= (max_log_date - pd.Timedelta(days=1))]
        elig_d7 = grp[grp["cohort_date"] <= (max_log_date - pd.Timedelta(days=7))]

        segment_rows.append({
            "segment": seg_name,
            "users": u_cnt,
            "contact_users": contact_users,
            "overall_cr_pct": round(100.0 * contact_users / u_cnt, 2),
            "retention_d1_pct": round(100.0 * float(elig_d1["ret_d1"].mean()), 2),
            "retention_d7_pct": round(100.0 * float(elig_d7["ret_d7"].mean()), 2),
            "avg_views_per_user": round(float(grp["views_cnt"].mean()), 1),
            "avg_contacts_per_user": round(contacts_total / u_cnt, 2),
            "avg_contacted_ad_price_rub": round(avg_item_price, 2),
            "contacted_gmv_per_user_rub": round(gmv_per_user, 2),
        })
    segment_rows = sorted(segment_rows, key=lambda x: x["users"], reverse=True)

    # 5. Статистика показов и кликов по типам продвижения в поисковой выдаче (SearchStream.ObjectType)
    # ObjectType: 1 - обычное объявление, 2 - выделенное/VIP, 3 - контекстное объявление (с целевым флагом IsClick)
    obj_counts = stream["ObjectType"].value_counts().to_dict()
    ctx_stream = stream[stream["ObjectType"] == 3.0]
    ctx_ctr_pct = float(ctx_stream["IsClick"].mean() * 100.0)

    overall_avg_contact_price = float(pr_with_price["Price"].dropna().mean())
    overall_median_contact_price = float(pr_with_price["Price"].dropna().median())

    return {
        "dataset": "Avito Context Ad Clicks (rel-avito, 2015-04-25 to 2015-05-20)",
        "dq_summary": {
            "raw_events_total": raw_total_events,
            "raw_search_events": raw_searches_cnt,
            "raw_visit_events": raw_visits_cnt,
            "raw_phone_request_events": raw_phones_cnt,
            "clean_events_total": clean_total_events,
            "filtered_events_total": filtered_total_events,
            "filtered_events_pct": round(filtered_pct, 2),
            "null_user_id_events": null_user_searches,
            "bot_ip_events": bot_events_total,
            "client_bounce_duplicates_lt_3s": bounce_events_total,
            "unique_clean_users": s1_users,
            "ads_total": int(len(ads)),
            "ads_valid_price_gt_0": valid_price_ads,
            "ads_zero_or_null_price": invalid_price_ads,
        },
        "funnel_steps": funnel_steps,
        "overall_cr_pct": overall_cr_pct,
        "retention_overall": retention_overall,
        "overall_marketplace_metrics": {
            "avg_contacted_ad_price_rub": round(overall_avg_contact_price, 2),
            "median_contacted_ad_price_rub": round(overall_median_contact_price, 2),
            "search_stream_impressions_total": int(len(stream)),
            "regular_ads_impressions": int(obj_counts.get(1.0, 0)),
            "highlighted_vip_impressions": int(obj_counts.get(2.0, 0)),
            "context_ads_impressions": int(obj_counts.get(3.0, 0)),
            "context_ads_ctr_pct": round(ctx_ctr_pct, 2),
        },
        "segments": segment_rows,
    }


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = resolve_data_dir()

    print("=" * 92)
    print("AVITO PRODUCT ANALYTICS | ПРОЕКТ 2: EDA КЛИКСТРИМА, ВОРОНКА CJM И RETENTION (rel-avito)")
    print("=" * 92)

    res = analyze_real_avito_funnel_and_retention(db_dir)
    dq = res["dq_summary"]

    print(f"[1] РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA) И ОЧИСТКА СЫРЫХ ЛОГОВ АВИТО (SearchInfo + VisitStream + PhoneRequests):")
    print(f"    • Сырой поток событий: {dq['raw_events_total']:,} записей "
          f"(поисков: {dq['raw_search_events']:,}, просмотров: {dq['raw_visit_events']:,}, контактов: {dq['raw_phone_request_events']:,})")
    print(f"    • Очищенная таблица сессий: {dq['clean_events_total']:,} событий ({dq['unique_clean_users']:,} уникальных пользователей)")
    print(f"    • Отфильтровано аномалий и дублей: {dq['filtered_events_total']:,} ({dq['filtered_events_pct']:.2f}%)")
    print(f"      - Аномальная бот-активность (>99.95% перцентиля IPID): {dq['bot_ip_events']:,}")
    print(f"      - Клиентский дребезг (<3 сек на одном UserID + AdID/CategoryID): {dq['client_bounce_duplicates_lt_3s']:,}")
    print(f"      - События без UserID (SearchInfo): {dq['null_user_id_events']:,}")

    print(f"\n[2] ПОШАГОВЫЙ АНАЛИЗ ОТВАЛА В ВОРОНКЕ CJM АВИТО:")
    for s in res["funnel_steps"]:
        print(f"    • {s['step']:<52} | Польз.: {s['users']:>6,} | Шаг CR: {s['step_cr_pct']:>6.2f}% | Отвал: {s['dropoff_pct']:>6.2f}%")
    print(f"    • Сквозная конверсия в запрос телефона продавца (Overall CR): {res['overall_cr_pct']:.2f}%")

    ret = res["retention_overall"]
    print(f"\n[3] КОГОРТНОЕ УДЕРЖАНИЕ ПОЛЬЗОВАТЕЛЕЙ АВИТО (2015-04-25 .. 2015-05-20):")
    print(f"    • Retention D1 = {ret['D1_pct']:.2f}% | D3 = {ret['D3_pct']:.2f}% | D7 = {ret['D7_pct']:.2f}% | D14 = {ret['D14_pct']:.2f}%")

    mkt = res["overall_marketplace_metrics"]
    print(f"\n[4] СЕГМЕНТНЫЙ АНАЛИЗ И МЕТРИКИ ОБЪЯВЛЕНИЙ (AdsInfo.Price & SearchStream):")
    print(f"    • Средняя цена объявления при запросе контакта: {mkt['avg_contacted_ad_price_rub']:,.2f} руб. "
          f"(медиана: {mkt['median_contacted_ad_price_rub']:,.0f} руб.)")
    print(f"    • Показы в выдаче SearchStream ({mkt['search_stream_impressions_total']:,}): "
          f"обычные = {mkt['regular_ads_impressions']:,}, VIP = {mkt['highlighted_vip_impressions']:,}, "
          f"контекстные = {mkt['context_ads_impressions']:,} (CTR = {mkt['context_ads_ctr_pct']:.2f}%)")
    print(f"    {'Сегмент пользователей':<38} | {'Польз.':>6} | {'CR,%':>6} | {'Ret D1':>6} | {'Ret D7':>6} | {'Просм.':>6} | {'Ср. цена, руб.':>14}")
    print("    " + "-" * 96)
    for seg in res["segments"]:
        print(
            f"    {seg['segment']:<38} | {seg['users']:>6,} | {seg['overall_cr_pct']:>5.2f}% | "
            f"{seg['retention_d1_pct']:>5.2f}% | {seg['retention_d7_pct']:>5.2f}% | "
            f"{seg['avg_views_per_user']:>6.1f} | {seg['avg_contacted_ad_price_rub']:>14,.2f}"
        )
    print("=" * 92)

    metrics_path = os.path.join(output_dir, "funnel_retention_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=140)

    steps_short = [
        "1. Поиск\n(SearchInfo)",
        "2. Просмотр\nкарточки (>=1)",
        "3. Глубокий\nпросмотр (>=5)",
        "4. Запрос телефона\n(PhoneRequest)",
    ]
    users_vals = [s["users"] for s in res["funnel_steps"]]
    bars = axes[0].barh(steps_short[::-1], users_vals[::-1], color=["#2f855a", "#38a169", "#3182ce", "#2b6cb0"])
    axes[0].set_title("Воронка CJM Авито (уникальные пользователи, N = 98 137)", fontsize=10.5)
    axes[0].set_xlabel("Количество пользователей")
    for bar, s in zip(bars, res["funnel_steps"][::-1]):
        axes[0].text(
            bar.get_width() + 1200,
            bar.get_y() + bar.get_height() / 2,
            f"{s['users']:,} (CR шага: {s['step_cr_pct']:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    axes[0].set_xlim(0, max(users_vals) * 1.38)

    seg_names = [s["segment"].replace(" | ", "\n") for s in res["segments"]]
    cr_vals = [s["overall_cr_pct"] for s in res["segments"]]
    d7_vals = [s["retention_d7_pct"] for s in res["segments"]]
    x = np.arange(len(seg_names))
    w = 0.36
    axes[1].bar(x - w / 2, cr_vals, width=w, label="Конверсия в контакт (CR, %)", color="#2b6cb0")
    axes[1].bar(x + w / 2, d7_vals, width=w, label="Retention D7 (%)", color="#38a169")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seg_names, fontsize=8.5)
    axes[1].set_ylabel("Доля пользователей (%)")
    axes[1].set_title("Конверсия в контакт и Retention D7 по сегментам пользователей", fontsize=10.5)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "funnel_retention_dashboard.png")
    plt.savefig(plot_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
