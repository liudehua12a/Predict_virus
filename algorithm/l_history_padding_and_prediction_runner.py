'''
    从 weather_daily 查某点位最近历史日表
    如果不足 21 天，用 mock 日表往前补齐
    从 weather_daily 查未来预报日表
    调 09_online_prediction_preparation.py 构造 future_rows
'''
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any
import i_online_prediction_preparation as prep
import k_weather_data_storage as storage
from pathlib import Path


MIN_HISTORY_DAYS = 21
FORECAST_FULL_DAYS = 7


def normalize_date_str(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()[:10]


def parse_date(value: Any) -> datetime:
    return datetime.strptime(normalize_date_str(value), "%Y-%m-%d")


def generate_mock_history_rows(
    earliest_real_row: dict[str, Any],
    missing_days: int,
) -> list[dict[str, Any]]:
    """
    以最早一条真实日表为模板，向前复制生成缺失的 mock 历史日表。
    第一版不加扰动，先保证链路跑通。
    """
    if missing_days <= 0:
        return []

    base_date = parse_date(earliest_real_row["date"])
    mock_rows: list[dict[str, Any]] = []

    for offset in range(missing_days, 0, -1):
        new_date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        new_row = dict(earliest_real_row)
        new_row["date"] = new_date
        new_row["data_source"] = "mock_seed"
        mock_rows.append(new_row)

    return mock_rows


def ensure_min_history_days(
    site_id: int,
    end_date_str: str,
    min_days: int = MIN_HISTORY_DAYS,
) -> list[dict[str, Any]]:
    """
    从数据库取最近历史日表；如果不足 min_days 天，则用 mock 数据向前补齐。
    """
    real_rows = storage.get_recent_weather_daily_rows(
        site_id=site_id,
        end_date_str=end_date_str,
        n_days=min_days,
    )

    if not real_rows:
        raise ValueError(
            f"点位 {site_id} 截止 {end_date_str} 在 weather_daily 中没有任何历史数据，无法补齐。"
        )

    if len(real_rows) >= min_days:
        return real_rows

    missing_days = min_days - len(real_rows)
    earliest_real_row = real_rows[0]

    mock_rows = generate_mock_history_rows(
        earliest_real_row=earliest_real_row,
        missing_days=missing_days,
    )

    merged_rows = mock_rows + real_rows
    merged_rows.sort(key=lambda x: x["date"])
    return merged_rows


def get_future_forecast_daily_rows_from_db(
    site_id: int,
    start_date_str: str,
    end_date_str: str,
) -> list[dict[str, Any]]:
    """
    从 weather_daily 读取未来预报日表。
    """
    rows = storage.get_future_forecast_daily_rows(
        site_id=site_id,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    if not rows:
        raise ValueError(
            f"点位 {site_id} 在 {start_date_str} ~ {end_date_str} 范围内没有未来预报日表，请先执行 forecast 入库。"
        )
    return rows

def get_forecast_window_dates(
    today_date: date | None = None,
    forecast_days: int = 7,
) -> tuple[str, str]:
    """
    计算“包含当天”的未来预报读取窗口。
    
    例如：
    - today_date = 2026-04-01
    - forecast_days = 7
    返回：
    - start_date_str = '2026-04-01'
    - end_date_str   = '2026-04-07'
    """
    if today_date is None:
        today_date = datetime.now().date()

    end_date = today_date + timedelta(days=forecast_days - 1)
    return today_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def build_predict_dates_from_forecast_rows(
    forecast_daily_rows: list[dict[str, Any]],
) -> list[str]:
    """
    从未来预报日表中提取待预测日期列表。
    """
    return [normalize_date_str(row["date"]) for row in forecast_daily_rows]


def run_prediction_preparation_demo():
    """
    演示：
    1. 从数据库读取最近历史
    2. 不足 21 天则 mock 补齐
    3. 从数据库读取未来 7 天预报日表
    4. 构造 future_rows
    """
    site_id = 6
    history_end_date_str = "2026-03-29"

    print("========== 预测准备联调测试 ==========")
    print(f"点位: {site_id}")
    print(f"历史截止日期: {history_end_date_str}")

    # Step 1: 历史不足 21 天时补齐
    history_daily_rows = ensure_min_history_days(
        site_id=site_id,
        end_date_str=history_end_date_str,
        min_days=MIN_HISTORY_DAYS,
    )

    print(f"\n历史日表条数（补齐后）: {len(history_daily_rows)}")
    print("历史起止日期：", history_daily_rows[0]["date"], "->", history_daily_rows[-1]["date"])
    print("历史前3条：")
    for row in history_daily_rows[:3]:
        print(row["date"], row.get("data_source", "unknown"))
    print("历史后3条：")
    for row in history_daily_rows[-3:]:
        print(row["date"], row.get("data_source", "unknown"))

    # Step 2: 自动计算“包含当天”的未来 7 天窗口
    today_date = datetime.now().date()
    forecast_start_date_str, forecast_end_date_str = get_forecast_window_dates(
        today_date=today_date,
        forecast_days=7,
    )

    print(f"\n未来预报读取窗口: {forecast_start_date_str} -> {forecast_end_date_str}")

    # Step 3: 从数据库读取未来预报日表
    forecast_daily_rows = get_future_forecast_daily_rows_from_db(
        site_id=site_id,
        start_date_str=forecast_start_date_str,
        end_date_str=forecast_end_date_str,
    )

    print(f"\n未来预报日表条数: {len(forecast_daily_rows)}")
    print("未来预报日期：")
    for row in forecast_daily_rows:
        print(normalize_date_str(row["date"]), row.get("data_source", "unknown"))

    # Step 4: 构造 future_rows
    predict_dates = build_predict_dates_from_forecast_rows(forecast_daily_rows)
    print("\n待预测日期：", predict_dates)

    # Step 5: 构造 future_rows
    future_rows = prep.build_future_prediction_rows(
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        site_id=site_id,
        predict_dates=predict_dates,
    )

    print(f"\nfuture_rows 条数: {len(future_rows)}")

    if future_rows:
        print("\n第一条 future_row 的关键信息：")
        first_row = future_rows[0]
        print("date:", normalize_date_str(first_row["date"]))
        print("site_id:", first_row["site_id"])
        print("weather_seq_21 shape:", first_row["weather_seq_21"].shape)

        print("\nfuture_rows 日期列表：")
        for row in future_rows:
            print(normalize_date_str(row["date"]))


if __name__ == "__main__":
    run_prediction_preparation_demo()