from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import algorithm.h_qweather_api as weather_api
import algorithm.k_weather_data_storage as storage
from scripts.utils.logger import log


def get_latest_seed_row(site_id: int, end_date_str: str) -> dict[str, Any]:
    """
    获取某点位在指定日期之前最近一天的日表，作为土壤状态初值。
    如果没有，则返回空 dict，由 h_qweather_api 内部默认值兜底。
    """
    recent_rows = storage.get_recent_weather_daily_rows(
        site_id=site_id,
        end_date_str=end_date_str,
        n_days=1,
    )
    if recent_rows:
        return recent_rows[-1]
    return {}


def filter_full_days_only(hourly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    只保留完整自然日（每个 date 必须 24 小时）。
    """
    date_counter = {}
    for row in hourly_rows:
        date_str = str(row["date"])
        date_counter[date_str] = date_counter.get(date_str, 0) + 1

    valid_dates = {d for d, cnt in date_counter.items() if cnt >= 24}
    return [row for row in hourly_rows if str(row["date"]) in valid_dates]


def run_forecast_task_for_site(site_row: dict[str, Any]) -> None:
    """
    单点位：拉未来 168h 小时预报 → 只保留完整自然日 → 聚合成日表 → 落库
    """
    site_id = int(site_row["site_id"])
    site_name = site_row["site_name"]
    lat = float(site_row["lat"])
    lon = float(site_row["lon"])

    log(f"[预报任务] 开始处理 site_id={site_id}, site_name={site_name}", "forecast")

    today_str = datetime.now().strftime("%Y-%m-%d")
    seed_row = get_latest_seed_row(site_id=site_id, end_date_str=today_str)

    api_json = weather_api.fetch_qweather_hourly_forecast_by_latlon(
        lat=lat,
        lon=lon,
        hourly_steps=168,
    )

    hourly_rows = weather_api.normalize_qweather_hourly_forecast_response(api_json)
    log(f"[预报任务] site_id={site_id} 原始小时预报条数={len(hourly_rows)}", "forecast")

    if not hourly_rows:
        log(f"[预报任务] site_id={site_id} 未获取到任何 168h 预报小时数据，跳过", "forecast")
        return

    hourly_rows = filter_full_days_only(hourly_rows)
    log(f"[预报任务] site_id={site_id} 完整自然日小时条数={len(hourly_rows)}", "forecast")

    if not hourly_rows:
        log(f"[预报任务] site_id={site_id} 168h 预报中没有完整自然日，跳过", "forecast")
        return

    daily_rows = weather_api.aggregate_hourly_rows_to_daily_rows(
        hourly_rows=hourly_rows,
        last_history_row=seed_row,
        min_hours_per_day=24,
    )
    log(f"[预报任务] site_id={site_id} 聚合后日表条数={len(daily_rows)}", "forecast")

    if not daily_rows:
        log(f"[预报任务] site_id={site_id} 聚合后没有任何完整预报日表，跳过", "forecast")
        return

    storage.upsert_weather_daily_rows(
        site_id=site_id,
        daily_rows=daily_rows,
        data_source="forecast_hourly",
    )

    log(
        f"[预报任务] site_id={site_id} 预报日表写入成功，条数={len(daily_rows)}，"
        f"日期={[row['date'] for row in daily_rows]}",
        "forecast"
    )


def run_forecast_task_for_all_sites() -> None:
    """
    所有点位：未来 168h 预报任务
    """
    sites = storage.get_all_active_sites()
    log(f"共读取到 {len(sites)} 个点位，开始执行 [23:55 预报任务]...", "forecast")

    for site_row in sites:
        try:
            run_forecast_task_for_site(site_row)
        except Exception as e:
            log(
                f"[失败][预报任务] site_id={site_row.get('site_id')}, "
                f"site_name={site_row.get('site_name')}，错误：{e}",
                "forecast"
            )

    log("\n===== [23:55 预报任务] 所有点位执行完成 =====", "forecast")


def run_history_override_task_for_site(site_row: dict[str, Any]) -> None:
    """
    单点位：拉“昨天”的历史逐小时数据 → 聚合成昨天的日表 → 覆盖 forecast_hourly
    """
    site_id = int(site_row["site_id"])
    site_name = site_row["site_name"]
    lat = float(site_row["lat"])
    lon = float(site_row["lon"])

    log(f"[历史覆盖任务] 开始处理 site_id={site_id}, site_name={site_name}", "history")

    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday_str_ymd = yesterday_dt.strftime("%Y-%m-%d")
    yesterday_str_api = yesterday_dt.strftime("%Y%m%d")

    seed_end_date_str = (yesterday_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    seed_row = get_latest_seed_row(site_id=site_id, end_date_str=seed_end_date_str)

    api_json = weather_api.fetch_qweather_history_hourly_by_latlon(
        lat=lat,
        lon=lon,
        target_date_str=yesterday_str_api,
    )

    hourly_rows = weather_api.normalize_qweather_history_hourly_response(api_json)
    log(f"[历史覆盖任务] site_id={site_id} 原始历史小时条数={len(hourly_rows)}", "history")

    if not hourly_rows:
        log(f"[历史覆盖任务] site_id={site_id} 未获取到任何历史小时数据，跳过", "history")
        return

    hourly_rows = [row for row in hourly_rows if str(row["date"]) == yesterday_str_ymd]
    log(f"[历史覆盖任务] site_id={site_id} 昨天小时条数={len(hourly_rows)}", "history")

    if len(hourly_rows) < 24:
        log(
            f"[历史覆盖任务] site_id={site_id} 昨天 {yesterday_str_ymd} 小时数不足24，当前={len(hourly_rows)}，跳过",
            "history"
        )
        return

    daily_rows = weather_api.aggregate_hourly_rows_to_daily_rows(
        hourly_rows=hourly_rows,
        last_history_row=seed_row,
        min_hours_per_day=24,
    )
    log(f"[历史覆盖任务] site_id={site_id} 聚合后日表条数={len(daily_rows)}", "history")

    daily_rows = [row for row in daily_rows if row["date"] == yesterday_str_ymd]

    if not daily_rows:
        log(f"[历史覆盖任务] site_id={site_id} 聚合后没有昨天 {yesterday_str_ymd} 的日表，跳过", "history")
        return

    storage.upsert_weather_daily_rows(
        site_id=site_id,
        daily_rows=daily_rows,
        data_source="history",
    )

    log(
        f"[历史覆盖任务] site_id={site_id} 历史覆盖成功，日期={yesterday_str_ymd}，条数={len(daily_rows)}",
        "history"
    )


def run_history_override_task_for_all_sites() -> None:
    """
    所有点位：昨天历史覆盖任务
    """
    sites = storage.get_all_active_sites()
    log(f"[历史覆盖任务] 共读取到 {len(sites)} 个点位", "history")

    for site_row in sites:
        try:
            run_history_override_task_for_site(site_row)
        except Exception as e:
            log(
                f"[失败][历史覆盖任务] site_id={site_row.get('site_id')}, "
                f"site_name={site_row.get('site_name')}，错误：{e}",
                "history"
            )

    log("[历史覆盖任务] 所有点位执行完成", "history")