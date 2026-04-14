from __future__ import annotations

from datetime import datetime, date
from typing import Any
from pathlib import Path

import numpy as np
import a_config as cfg
import c_feature_engineering as fe

'''
    把数据库里的历史日表和API来的预报日表拼成连续日表
    为未来每一天构造 weather_seq_21
    产出后面滚动预测需要的“未来样本行”
    只负责拼接、构造模型输入
'''


def parse_date(value: Any) -> date:
    """
    将字符串/日期对象统一转为 date。
    支持:
    - '2026-03-30'
    - datetime/date
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def ensure_daily_row_fields(row: dict[str, Any]) -> None:
    """
    检查日尺度天气行是否包含构造 weather_seq_21 所需的全部原始字段。
    """
    required_fields = list(cfg.SEQ_FEATURES) + ["date"]
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"日天气行缺少字段: {field_name}")
        if row[field_name] is None:
            raise ValueError(f"日天气行字段为 None: {field_name}")


def convert_daily_rows_to_weather_site(
    daily_rows: list[dict[str, Any]],
    site_id: int,
) -> dict[str, Any]:
    """
    将连续日表转换成与 03_feature_engineering.py 中 process_features 所需格式尽量一致的 weather_site。
    """
    if not daily_rows:
        raise ValueError("daily_rows 不能为空")

    normalized_rows: list[dict[str, Any]] = []
    for row in daily_rows:
        copied = dict(row)
        copied["date"] = parse_date(copied["date"])
        ensure_daily_row_fields(copied)
        normalized_rows.append(copied)

    normalized_rows.sort(key=lambda x: x["date"])

    dates = [row["date"] for row in normalized_rows]
    date_index = {d: i for i, d in enumerate(dates)}

    arrays: dict[str, np.ndarray] = {}
    for feature_name in cfg.SEQ_FEATURES:
        arrays[feature_name] = np.asarray(
            [float(row[feature_name]) for row in normalized_rows],
            dtype=np.float32,
        )

    # ===== 衍生数组：尽量复用训练阶段 add_process_features 所需内容 =====
    temp_avg = arrays["temp_avg_c"]
    precip = arrays["precip_sum"]
    rh = arrays["relative_humidity"]
    soil_rel_humidity = arrays["soil_rel_humidity"]
    radiation = arrays["radiation_avg"]
    wind = arrays["wind_avg"]

    arrays["temp_range_24h_c"] = arrays["temp_max_c"] - arrays["temp_min_c"]
    arrays["humidity_range_daily"] = arrays["relative_humidity_max"] - arrays["relative_humidity_min"]

    rainy_streak_days, rain_gap_days = fe.dc.compute_rain_streaks(precip)
    arrays["rainy_streak_days"] = rainy_streak_days
    arrays["rain_gap_days"] = rain_gap_days

    # 阈值：沿用训练逻辑里“按点位温度分位数”
    # ===== 温度阈值：按固定业务规则，不再使用分位数 =====
    temp_low_threshold = float(cfg.TEMP_LOW_THRESHOLD)
    temp_optimal_low = float(cfg.TEMP_OPTIMAL_LOW)
    temp_optimal_high = float(cfg.TEMP_OPTIMAL_HIGH)
    temp_high_threshold = float(cfg.TEMP_HIGH_THRESHOLD)

    temp_low_flag = temp_avg <= temp_low_threshold
    temp_high_flag = temp_avg >= temp_high_threshold
    temp_optimal_flag = (
        (temp_avg >= temp_optimal_low) & (temp_avg < temp_optimal_high)
    )

    arrays["temp_high_streak_days"] = fe.dc.compute_boolean_streaks(temp_high_flag)
    arrays["temp_low_streak_days"] = fe.dc.compute_boolean_streaks(temp_low_flag)
    arrays["temp_optimal_streak_days"] = fe.dc.compute_boolean_streaks(temp_optimal_flag)

    # 高湿阈值：沿用经验阈值 85%
        # ===== 湿度阈值：按固定业务规则 =====
    high_humidity_threshold = float(cfg.HIGH_HUMIDITY_THRESHOLD)
    medium_high_humidity_threshold = float(cfg.MEDIUM_HIGH_HUMIDITY_THRESHOLD)

    high_humidity_flag = rh >= high_humidity_threshold
    medium_high_humidity_flag = (
        (rh >= medium_high_humidity_threshold) & (rh < high_humidity_threshold)
    )

    arrays["high_humidity_flag"] = high_humidity_flag.astype(np.float32)
    arrays["medium_high_humidity_flag"] = medium_high_humidity_flag.astype(np.float32)

    arrays["high_humidity_streak_days"] = fe.dc.compute_boolean_streaks(high_humidity_flag)
    arrays["medium_high_humidity_streak_days"] = fe.dc.compute_boolean_streaks(
        medium_high_humidity_flag
    )

    high_humidity_7d_count = []
    for idx in range(len(rh)):
        start = max(0, idx - 6)
        high_humidity_7d_count.append(float(np.sum(high_humidity_flag[start : idx + 1])))
    arrays["high_humidity_7d_count"] = np.asarray(high_humidity_7d_count, dtype=np.float32)

    # 强降水阈值：经验阈值 25mm
    heavy_rain_threshold = float(cfg.HEAVY_RAIN_THRESHOLD)
    heavy_rain_flag = precip >= heavy_rain_threshold
    arrays["heavy_rain_flag"] = heavy_rain_flag.astype(np.float32)
    arrays["heavy_rain_streak_days"] = fe.dc.compute_boolean_streaks(heavy_rain_flag)

    heavy_rain_7d_count = []
    max_single_day_rain_7d = []
    for idx in range(len(precip)):
        start = max(0, idx - 6)
        window = precip[start : idx + 1]
        heavy_rain_7d_count.append(float(np.sum(window >= heavy_rain_threshold)))
        max_single_day_rain_7d.append(float(np.max(window)))
    arrays["heavy_rain_7d_count"] = np.asarray(heavy_rain_7d_count, dtype=np.float32)
    arrays["max_single_day_rain_7d"] = np.asarray(max_single_day_rain_7d, dtype=np.float32)

    # 弱风：经验阈值 <= 3
    weak_wind_flag = wind <= 3.0
    arrays["weak_wind_flag"] = weak_wind_flag.astype(np.float32)
    arrays["weak_wind_streak_days"] = fe.dc.compute_boolean_streaks(weak_wind_flag)

    # 寡照
    low_radiation_threshold = float(cfg.LOW_RADIATION_THRESHOLD)
    low_radiation_flag = radiation <= low_radiation_threshold
    arrays["low_radiation_streak_days"] = fe.dc.compute_boolean_streaks(low_radiation_flag)

    # 组合条件
    hot_humid_flag = temp_high_flag & high_humidity_flag
    optimal_temp_humid_flag = temp_optimal_flag & high_humidity_flag
    weak_wind_humid_flag = weak_wind_flag & high_humidity_flag

    arrays["hot_humid_streak_days"] = fe.dc.compute_boolean_streaks(hot_humid_flag)
    arrays["optimal_temp_humid_streak_days"] = fe.dc.compute_boolean_streaks(optimal_temp_humid_flag)
    arrays["weak_wind_humid_streak_days"] = fe.dc.compute_boolean_streaks(weak_wind_humid_flag)

    # 按 cfg.SEQ_FEATURES 固定顺序构造矩阵
    feature_matrix = np.column_stack(
        [arrays[feature_name] for feature_name in cfg.SEQ_FEATURES]
    ).astype(np.float32)

    weather_site = {
        "site_id": site_id,
        "dates": dates,
        "date_index": date_index,
        "arrays": arrays,
        "matrix": feature_matrix,
        "temp_thresholds": {
            "low": temp_low_threshold,
            "optimal_low": temp_optimal_low,
            "optimal_high": temp_optimal_high,
            "high": temp_high_threshold,
        },
    }

    validate_weather_site_arrays(weather_site)

    return weather_site

def validate_weather_site_arrays(weather_site: dict[str, Any]) -> None:
    required_array_keys = [
        *cfg.SEQ_FEATURES,

        "rainy_streak_days",
        "rain_gap_days",
        "temp_range_24h_c",
        "humidity_range_daily",

        "weak_wind_flag",
        "weak_wind_streak_days",
        "low_radiation_streak_days",

        "temp_high_streak_days",
        "temp_low_streak_days",
        "temp_optimal_streak_days",

        "high_humidity_flag",
        "medium_high_humidity_flag",
        "high_humidity_streak_days",
        "medium_high_humidity_streak_days",
        "high_humidity_7d_count",

        "heavy_rain_flag",
        "heavy_rain_streak_days",
        "heavy_rain_7d_count",
        "max_single_day_rain_7d",

        "hot_humid_streak_days",
        "optimal_temp_humid_streak_days",
        "weak_wind_humid_streak_days",
    ]

    arrays = weather_site["arrays"]
    for key in required_array_keys:
        if key not in arrays:
            raise ValueError(f"weather_site['arrays'] 缺少在线过程特征所需字段: {key}")

def validate_model_row_features(row: dict[str, Any]) -> None:
    for name in cfg.BASE_MODEL_FEATURES:
        if name not in row:
            raise ValueError(f"future_row 缺少 BASE_MODEL_FEATURES 字段: {name}")
        if row[name] is None:
            raise ValueError(f"future_row 的 BASE_MODEL_FEATURES 字段为 None: {name}")

    if "weather_seq_21" not in row:
        raise ValueError("future_row 缺少 weather_seq_21")

    if row["weather_seq_21"].shape[1] != len(cfg.SEQ_FEATURES):
        raise ValueError(
            f"weather_seq_21 列数不匹配：实际 {row['weather_seq_21'].shape[1]}，"
            f"配置要求 {len(cfg.SEQ_FEATURES)}"
        )

def build_online_panel_rows(
    continuous_daily_rows: list[dict[str, Any]],
    site_id: int,
) -> list[dict[str, Any]]:
    """
    将连续日天气表转成在线预测使用的 panel_rows 基础结构。
    此时病害真实值未知，先置空。
    """
    panel_rows: list[dict[str, Any]] = []
    ordered_rows = sorted(continuous_daily_rows, key=lambda x: parse_date(x["date"]))

    for idx, row in enumerate(ordered_rows, start=1):
        panel_rows.append(
            {
                "site_id": site_id,
                "date": parse_date(row["date"]),
                "record_id": idx,
                "replicate_id_same_day": 1,
                "stage_code": 0.0,
                "gray_incidence": None,
                "gray_index": None,
                "blight_incidence": None,
                "blight_index": None,
                "white_incidence": None,
                "white_index": None,
                **{k: row[k] for k in cfg.SEQ_FEATURES},
            }
        )
    return panel_rows


def inject_weather_seq_21_into_rows(
    panel_rows: list[dict[str, Any]],
    weather_site: dict[str, Any],
) -> None:
    """
    给每一行注入 weather_seq_21。
    """
    feature_matrix = np.column_stack([weather_site["arrays"][f] for f in cfg.SEQ_FEATURES]).astype(np.float32)

    for row in panel_rows:
        current_idx = fe.nearest_weather_index(weather_site, row["date"])
        row["weather_seq_21"] = fe.padded_sequence(
            matrix=feature_matrix,
            end_index=current_idx,
            lookback=cfg.LOOKBACK_DAYS,
        )


def build_continuous_daily_weather_rows(
    history_daily_rows: list[dict[str, Any]],
    forecast_daily_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    将历史日表与未来预报日表按日期拼接，去重后返回连续日表。
    若同一天同时存在，优先保留历史数据。
    """
    merged_by_date: dict[date, dict[str, Any]] = {}

    for row in forecast_daily_rows:
        row_date = parse_date(row["date"])
        merged_by_date[row_date] = dict(row)

    for row in history_daily_rows:
        row_date = parse_date(row["date"])
        merged_by_date[row_date] = dict(row)

    continuous_rows = [merged_by_date[d] for d in sorted(merged_by_date.keys())]
    return continuous_rows


def build_future_prediction_rows(
    history_daily_rows: list[dict[str, Any]],
    forecast_daily_rows: list[dict[str, Any]],
    site_id: int,
    predict_dates: list[str],
) -> list[dict[str, Any]]:
    """
    构造未来待预测日期对应的完整样本行。
    返回的每一行都包含：
    - weather_seq_21
    - BASE_MODEL_FEATURES 所需过程特征
    - date / site_id 等基础信息
    """
    continuous_daily_rows = build_continuous_daily_weather_rows(
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
    )

    weather_site = convert_daily_rows_to_weather_site(
        daily_rows=continuous_daily_rows,
        site_id=site_id,
    )

    panel_rows = build_online_panel_rows(
        continuous_daily_rows=continuous_daily_rows,
        site_id=site_id,
    )

    site_rows = fe.add_process_features(
        panel_rows=panel_rows,
        weather_by_site={site_id: weather_site},
    )[site_id]

    inject_weather_seq_21_into_rows(
        panel_rows=site_rows,
        weather_site=weather_site,
    )

    predict_date_set = {parse_date(x) for x in predict_dates}
    future_rows = [row for row in site_rows if row["date"] in predict_date_set]
    future_rows.sort(key=lambda x: x["date"])

    for row in future_rows:
        validate_model_row_features(row)

    return future_rows