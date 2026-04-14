from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any
import g_qweather_client as qweather_api


"""
和风天气 API 相关工具函数。
负责取天气、标准化、聚合。
"""


# ===== 导入配置与客户端 =====

QWEATHER_API_HOST = "https://nb2k5payfn.re.qweatherapi.com"
QWEATHER_PRIVATE_KEY_PATH = "ed25519-private.pem"
QWEATHER_SUB = "4FKRV33M9W"
QWEATHER_KID = "KJ59BN995H"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 简化土壤-地表状态参数（经验参数，先统一收口） =====
SOIL_MOISTURE_MIN = 5.0
SOIL_MOISTURE_MAX = 45.0
SOIL_MOISTURE_FIELD_CAPACITY = 32.0

SOIL_TEMP_LAG_ALPHA = 0.18     # 土壤温度对地表温度的响应系数
SOIL_WATER_RECHARGE_COEF = 0.65
SOIL_WATER_ET_TEMP_COEF = 0.10
SOIL_WATER_ET_RAD_COEF = 0.010
SOIL_WATER_ET_WIND_COEF = 0.06


def build_qweather_client() -> Any:
    """
    创建和风天气客户端。
    """
    return qweather_api.QWeatherClient(
        api_host=QWEATHER_API_HOST,
        private_key_path=QWEATHER_PRIVATE_KEY_PATH,
        sub=QWEATHER_SUB,
        kid=QWEATHER_KID,
        timeout=30,
    )


def get_next_midnight_and_required_hours(full_days: int = 7) -> tuple[str, int]:
    """
    根据当前本地时间，自动计算：
    1. 过滤起始时间：明天 00:00:00
    2. 为保证从该时刻开始仍有 full_days 个完整自然日，需要请求的总小时数
    """
    now_dt = datetime.now()
    next_midnight = (now_dt + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    seconds_to_midnight = (next_midnight - now_dt).total_seconds()
    hours_to_midnight = ceil(seconds_to_midnight / 3600.0)

    hourly_steps = hours_to_midnight + full_days * 24
    start_datetime_str = next_midnight.strftime("%Y-%m-%dT%H:%M:%S")

    return start_datetime_str, hourly_steps


def to_float(value: Any, default: float | None = 0.0) -> float | None:
    """
    将输入值安全转成 float。
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def clamp(value: float, low: float, high: float) -> float:
    """
    将数值裁剪到指定范围内。
    """
    return max(low, min(high, float(value)))


def convert_pressure_pa_to_kpa(value: Any, default: float = 0.0) -> float:
    """
    pressure 字段统一转成 kPa。
    当前和风返回样本已是 hPa/mbar 量级数值（如 955），这里按 kPa 使用时需 /10。
    如果你后续确认统一单位策略，可再调整。
    """
    raw = to_float(value, default)
    return float(raw) / 10.0

def init_soil_state_if_missing(last_history_row: dict[str, Any]) -> tuple[float, float, float]:
    """
    如果没有历史土壤状态，用合理默认值初始化
    """
    soil_moisture = to_float(last_history_row.get("soil_moisture"), None)
    soil_temp = to_float(last_history_row.get("soil_temp_c"), None)
    soil_rh = to_float(last_history_row.get("soil_rel_humidity"), None)

    # ===== 默认值（关键）=====
    if soil_moisture is None or soil_moisture <= 0:
        soil_moisture = 25.0  # 中等湿润（核心默认值）

    if soil_temp is None or soil_temp <= 0:
        soil_temp = 18.0

    if soil_rh is None or soil_rh <= 0:
        soil_rh = 70.0

    return float(soil_moisture), float(soil_rh), float(soil_temp)

def estimate_surface_temperature_from_daily(
    temp_avg_c: float,
    temp_max_c: float,
    temp_min_c: float,
    radiation_avg: float,
    wind_avg: float,
    relative_humidity: float,
) -> tuple[float, float, float]:
    """
    用日尺度气象要素估算地表温度。
    逻辑：
    - 白天地表温度高于2m气温，增幅受辐射增强
    - 风速越大、湿度越高，增幅越受抑制
    - 夜间最低地表温度通常略低于2m最低气温
    """
    rh_factor = clamp(relative_humidity / 100.0, 0.0, 1.0)
    rad_factor = max(radiation_avg, 0.0) / 300.0
    wind_factor = max(wind_avg, 0.0)

    daytime_boost = 2.5 + 2.2 * rad_factor - 0.35 * wind_factor - 1.2 * rh_factor
    nighttime_offset = 0.6 + 0.15 * wind_factor - 0.25 * rad_factor

    surface_temp_max_c = temp_max_c + daytime_boost
    surface_temp_min_c = temp_min_c - nighttime_offset

    # 地表均温不直接取 max/min 平均，而是相对 2m 气温均值做轻微上移
    surface_temp_avg_c = temp_avg_c + 0.25 * daytime_boost - 0.10 * nighttime_offset

    return (
        surface_temp_avg_c,
        surface_temp_max_c,
        surface_temp_min_c,
    )


def estimate_next_soil_temperature(
    prev_soil_temp_c: float,
    surface_temp_avg_c: float,
) -> float:
    """
    用一阶滞后模型估算 5cm 土壤温度。
    土壤温度相对地表温度变化更平滑。
    """
    next_soil_temp_c = (
        (1.0 - SOIL_TEMP_LAG_ALPHA) * prev_soil_temp_c
        + SOIL_TEMP_LAG_ALPHA * surface_temp_avg_c
    )
    return next_soil_temp_c


def estimate_next_soil_moisture(
    prev_soil_moisture: float,
    precip_sum: float,
    temp_avg_c: float,
    radiation_avg: float,
    wind_avg: float,
    relative_humidity: float,
) -> float:
    """
    简化水桶模型：
    - 降水作为补给
    - 温度 / 辐射 / 风速 作为蒸散消耗
    - 相对湿度越高，蒸散消耗越弱
    输出保持在一个较合理的体积含水范围内
    """
    rh_factor = clamp(relative_humidity / 100.0, 0.0, 1.0)

    recharge = SOIL_WATER_RECHARGE_COEF * max(precip_sum, 0.0)

    evap_loss = (
        SOIL_WATER_ET_TEMP_COEF * max(temp_avg_c - 5.0, 0.0)
        + SOIL_WATER_ET_RAD_COEF * max(radiation_avg, 0.0)
        + SOIL_WATER_ET_WIND_COEF * max(wind_avg, 0.0)
    ) * (1.0 - 0.45 * rh_factor)

    next_value = prev_soil_moisture + recharge - evap_loss

    return clamp(next_value, SOIL_MOISTURE_MIN, SOIL_MOISTURE_MAX)


def estimate_soil_rel_humidity_from_moisture(
    soil_moisture: float,
) -> float:
    """
    将土壤含水量映射成土壤相对湿度（0~100）。
    这里不再做二次递推，而是直接按含水状态换算。
    """
    rel = 100.0 * (soil_moisture - SOIL_MOISTURE_MIN) / (
        SOIL_MOISTURE_FIELD_CAPACITY - SOIL_MOISTURE_MIN
    )
    return clamp(rel, 0.0, 100.0)


def estimate_radiation_from_hour(
    hour_int: int,
    cloud: float | None = None,
) -> float:
    """
    第一版：
    - 夜间辐射为 0
    - 白天按时间位置估算晴空辐射
    - forecast 有 cloud 时，用 cloud 修正
    - history 无 cloud 时，用固定 0.85 系数
    """
    if hour_int < 6 or hour_int > 18:
        return 0.0

    peak = max(0.0, 1.0 - abs(hour_int - 12) / 6.0)

    if cloud is None:
        return 700.0 * peak * 0.85

    cloud = to_float(cloud, 0.0)
    cloud_ratio = clamp(cloud / 100.0, 0.0, 1.0)

    return 800.0 * peak * (1.0 - 0.75 * cloud_ratio)


def normalize_qweather_hourly_items(
    hourly_items: list[dict[str, Any]],
    time_field_name: str,
) -> list[dict[str, float | str]]:
    """
    将和风天气小时数据统一标准化为本项目内部逐小时结构。
    适用于：
    1. 历史接口 history.weatherHourly（time）
    2. 168h 预报接口 forecast.hourly（fxTime）
    """
    normalized_rows: list[dict[str, float | str]] = []

    for item in hourly_items:
        datetime_str = str(item.get(time_field_name, "")).strip()
        if not datetime_str:
            continue

        date_str = datetime_str[:10]
        hour_str = datetime_str[11:13] if len(datetime_str) >= 13 else "00"

        hour_int = int(hour_str)
        cloud_value = to_float(item.get("cloud"), None)

        row = {
            "datetime": datetime_str,
            "date": date_str,
            "hour": hour_str,
            "temp_c": float(to_float(item.get("temp"), 0.0)),
            "precip_mm": float(to_float(item.get("precip"), 0.0)),
            "precip_probability": float(to_float(item.get("pop"), 0.0)),
            "wind_speed": float(to_float(item.get("windSpeed"), 0.0)),
            "wind_direction": float(to_float(item.get("wind360"), 0.0)),
            "relative_humidity": clamp(float(to_float(item.get("humidity"), 0.0)), 0.0, 100.0),
            "pressure_kpa": convert_pressure_pa_to_kpa(item.get("pressure"), 0.0),
            "radiation_wm2": estimate_radiation_from_hour(
                hour_int=hour_int,
                cloud=cloud_value,
            ),
        }

        normalized_rows.append(row)

    return normalized_rows


def normalize_qweather_history_hourly_response(
    api_json: dict[str, Any],
) -> list[dict[str, float | str]]:
    """
    将和风历史天气接口返回的 history.weatherHourly 标准化为逐小时结构。
    """
    history = api_json.get("history", {})
    hourly_items = history.get("weatherHourly", [])
    if not hourly_items:
        raise RuntimeError(f"和风历史接口缺少 weatherHourly 字段: {api_json}")

    return normalize_qweather_hourly_items(
        hourly_items=hourly_items,
        time_field_name="time",
    )


def normalize_qweather_hourly_forecast_response(
    api_json: dict[str, Any],
) -> list[dict[str, float | str]]:
    """
    将和风 168h 小时预报返回的 forecast.hourly 标准化为逐小时结构。
    """
    forecast = api_json.get("forecast", {})
    hourly_items = forecast.get("hourly", [])
    if not hourly_items:
        raise RuntimeError(f"和风168h预报接口缺少 hourly 字段: {api_json}")

    return normalize_qweather_hourly_items(
        hourly_items=hourly_items,
        time_field_name="fxTime",
    )


def normalize_qweather_daily_forecast_response(
    api_json: dict[str, Any],
    last_history_row: dict[str, Any],
) -> list[dict[str, float | str]]:
    """
    将和风 daily 预报标准化为项目内部日结构。
    注意：daily 没有 temp_avg_c，这里先用 (max + min) / 2 估算。
    该函数只作为备用，不作为主链路。
    """
    forecast = api_json.get("forecast", {})
    daily_items = forecast.get("daily", [])
    if not daily_items:
        raise RuntimeError(f"和风daily预报接口缺少 daily 字段: {api_json}")

    normalized_rows: list[dict[str, float | str]] = []

    prev_soil_moisture, prev_soil_rel_humidity, prev_soil_temp_c = \
        init_soil_state_if_missing(last_history_row)
    
    for item in daily_items:
        date_str = str(item.get("fxDate", "")).strip()
        if not date_str:
            continue

        temp_max_c = float(to_float(item.get("tempMax"), 0.0))
        temp_min_c = float(to_float(item.get("tempMin"), 0.0))
        temp_avg_c = (temp_max_c + temp_min_c) / 2.0

        wind_day = float(to_float(item.get("windSpeedDay"), 0.0))
        wind_night = float(to_float(item.get("windSpeedNight"), 0.0))
        wind_avg = (wind_day + wind_night) / 2.0
        wind_max = max(wind_day, wind_night)
        wind_min = min(wind_day, wind_night)

        relative_humidity = clamp(float(to_float(item.get("humidity"), 0.0)), 0.0, 100.0)
        relative_humidity_max = relative_humidity
        relative_humidity_min = relative_humidity

        precip_sum = float(to_float(item.get("precip"), 0.0))
        precip_max = precip_sum
        precip_min = 0.0

        pressure_kpa = convert_pressure_pa_to_kpa(item.get("pressure"), 0.0)
        pressure_max_kpa = pressure_kpa
        pressure_min_kpa = pressure_kpa

        cloud_value = float(to_float(item.get("cloud"), 0.0))
        radiation_avg = estimate_radiation_from_hour(12, cloud_value)
        radiation_max = radiation_avg
        radiation_min = 0.0

        (
            surface_temp_avg_c,
            surface_temp_max_c,
            surface_temp_min_c,
        ) = estimate_surface_temperature_from_daily(
            temp_avg_c=temp_avg_c,
            temp_max_c=temp_max_c,
            temp_min_c=temp_min_c,
            radiation_avg=radiation_avg,
            wind_avg=wind_avg,
            relative_humidity=relative_humidity,
        )

        soil_temp_c = estimate_next_soil_temperature(
            prev_soil_temp_c=prev_soil_temp_c,
            surface_temp_avg_c=surface_temp_avg_c,
        )

        soil_moisture = estimate_next_soil_moisture(
            prev_soil_moisture=prev_soil_moisture,
            precip_sum=precip_sum,
            temp_avg_c=temp_avg_c,
            radiation_avg=radiation_avg,
            wind_avg=wind_avg,
            relative_humidity=relative_humidity,
        )

        soil_rel_humidity = estimate_soil_rel_humidity_from_moisture(
            soil_moisture=soil_moisture,
        )

        row = {
            "date": date_str,
            "wind_avg": wind_avg,
            "wind_max": wind_max,
            "wind_min": wind_min,
            "precip_max": precip_max,
            "precip_min": precip_min,
            "precip_sum": precip_sum,
            "relative_humidity": relative_humidity,
            "relative_humidity_max": relative_humidity_max,
            "relative_humidity_min": relative_humidity_min,
            "temp_avg_c": temp_avg_c,
            "temp_max_c": temp_max_c,
            "temp_min_c": temp_min_c,
            "soil_moisture": soil_moisture,
            "surface_temp_avg_c": surface_temp_avg_c,
            "surface_temp_max_c": surface_temp_max_c,
            "surface_temp_min_c": surface_temp_min_c,
            "pressure_kpa": pressure_kpa,
            "pressure_max_kpa": pressure_max_kpa,
            "pressure_min_kpa": pressure_min_kpa,
            "radiation_avg": radiation_avg,
            "radiation_max": radiation_max,
            "radiation_min": radiation_min,
            "soil_rel_humidity": soil_rel_humidity,
            "soil_temp_c": soil_temp_c,
        }
        normalized_rows.append(row)

        prev_soil_moisture = soil_moisture
        prev_soil_rel_humidity = soil_rel_humidity
        prev_soil_temp_c = soil_temp_c

    for row in normalized_rows:
        validate_daily_row(row)

    return normalized_rows


def fetch_qweather_history_hourly_by_latlon(
    lat: float,
    lon: float,
    target_date_str: str,
) -> dict[str, Any]:
    """
    获取和风历史天气原始 JSON。
    target_date_str: YYYYMMDD
    """
    client = build_qweather_client()
    return client.get_historical_weather_by_lonlat(
        lon=lon,
        lat=lat,
        date=target_date_str,
        range_="cn",
        lang="zh",
        unit="m",
    )


def fetch_qweather_hourly_forecast_by_latlon(
    lat: float,
    lon: float,
    hourly_steps: int = 168,
) -> dict[str, Any]:
    """
    获取和风小时预报原始 JSON。
    """
    client = build_qweather_client()
    return client.get_hourly_forecast_by_lonlat(
        lon=lon,
        lat=lat,
        hourly_steps=hourly_steps,
        range_="cn",
        lang="zh",
        unit="m",
    )


def fetch_qweather_daily_forecast_by_latlon(
    lat: float,
    lon: float,
    days: int = 7,
) -> dict[str, Any]:
    """
    获取和风 daily 预报原始 JSON。
    """
    client = build_qweather_client()
    return client.get_daily_forecast_by_lonlat(
        lon=lon,
        lat=lat,
        days=days,
        range_="cn",
        lang="zh",
        unit="m",
    )


def validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"纬度超出范围: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"经度超出范围: {lon}")


def filter_hourly_rows_from_datetime(
    hourly_rows: list[dict[str, float | str]],
    start_datetime_str: str,
) -> list[dict[str, float | str]]:
    """
    仅保留 datetime >= start_datetime_str 的逐小时记录。
    """
    start_dt = datetime.fromisoformat(start_datetime_str)
    if start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)

    filtered_rows: list[dict[str, float | str]] = []

    for row in hourly_rows:
        row_dt = datetime.fromisoformat(str(row["datetime"]))
        if row_dt.tzinfo is not None:
            row_dt = row_dt.replace(tzinfo=None)

        if row_dt >= start_dt:
            filtered_rows.append(row)

    return filtered_rows


def validate_daily_row(row: dict[str, Any]) -> None:
    required_fields = [
        "date",
        "wind_avg", "wind_max", "wind_min",
        "precip_sum", "precip_max", "precip_min",
        "relative_humidity", "relative_humidity_max", "relative_humidity_min",
        "temp_avg_c", "temp_max_c", "temp_min_c",
        "soil_moisture",
        "surface_temp_avg_c", "surface_temp_max_c", "surface_temp_min_c",
        "pressure_kpa", "pressure_max_kpa", "pressure_min_kpa",
        "radiation_avg", "radiation_max", "radiation_min",
        "soil_rel_humidity", "soil_temp_c",
    ]

    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"聚合后的日记录缺少字段: {field_name}")
        if row[field_name] is None:
            raise ValueError(f"聚合后的日记录字段为 None: {field_name}")


def aggregate_hourly_rows_to_daily_rows(
    hourly_rows: list[dict[str, float | str]],
    last_history_row: dict[str, Any],
    min_hours_per_day: int = 24,
) -> list[dict[str, float | str]]:
    """
    将标准化后的逐小时记录聚合为模型所需的日尺度记录。
    """
    if not hourly_rows:
        return []

    hourly_rows = sorted(hourly_rows, key=lambda x: str(x["datetime"]))

    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in hourly_rows:
        date_str = str(row["date"])
        grouped.setdefault(date_str, []).append(row)

    daily_rows: list[dict[str, float | str]] = []

    prev_soil_moisture, prev_soil_rel_humidity, prev_soil_temp_c = \
        init_soil_state_if_missing(last_history_row)

    for date_str in sorted(grouped.keys()):
        rows = grouped[date_str]
        if len(rows) < min_hours_per_day:
            continue

        temp_values = [float(to_float(r["temp_c"], 0.0)) for r in rows]
        precip_values = [float(to_float(r["precip_mm"], 0.0)) for r in rows]
        wind_values = [float(to_float(r["wind_speed"], 0.0)) for r in rows]
        rh_values = [float(to_float(r["relative_humidity"], 0.0)) for r in rows]
        pressure_values = [float(to_float(r["pressure_kpa"], 0.0)) for r in rows]
        radiation_values = [float(to_float(r["radiation_wm2"], 0.0)) for r in rows]

        temp_avg_c = sum(temp_values) / len(temp_values)
        temp_max_c = max(temp_values)
        temp_min_c = min(temp_values)

        precip_sum = sum(precip_values)
        precip_max = max(precip_values)
        precip_min = min(precip_values)

        wind_avg = sum(wind_values) / len(wind_values)
        wind_max = max(wind_values)
        wind_min = min(wind_values)

        relative_humidity = sum(rh_values) / len(rh_values)
        relative_humidity_max = max(rh_values)
        relative_humidity_min = min(rh_values)

        pressure_kpa = sum(pressure_values) / len(pressure_values)
        pressure_max_kpa = max(pressure_values)
        pressure_min_kpa = min(pressure_values)

        radiation_avg = sum(radiation_values) / len(radiation_values)
        radiation_max = max(radiation_values)
        radiation_min = min(radiation_values)

        (
            surface_temp_avg_c,
            surface_temp_max_c,
            surface_temp_min_c,
        ) = estimate_surface_temperature_from_daily(
            temp_avg_c=temp_avg_c,
            temp_max_c=temp_max_c,
            temp_min_c=temp_min_c,
            radiation_avg=radiation_avg,
            wind_avg=wind_avg,
            relative_humidity=relative_humidity,
        )

        soil_temp_c = estimate_next_soil_temperature(
            prev_soil_temp_c=prev_soil_temp_c,
            surface_temp_avg_c=surface_temp_avg_c,
        )

        soil_moisture = estimate_next_soil_moisture(
            prev_soil_moisture=prev_soil_moisture,
            precip_sum=precip_sum,
            temp_avg_c=temp_avg_c,
            radiation_avg=radiation_avg,
            wind_avg=wind_avg,
            relative_humidity=relative_humidity,
        )

        soil_rel_humidity = estimate_soil_rel_humidity_from_moisture(
            soil_moisture=soil_moisture,
        )

        daily_row = {
            "date": date_str,
            "wind_avg": wind_avg,
            "wind_max": wind_max,
            "wind_min": wind_min,
            "precip_max": precip_max,
            "precip_min": precip_min,
            "precip_sum": precip_sum,
            "relative_humidity": relative_humidity,
            "relative_humidity_max": relative_humidity_max,
            "relative_humidity_min": relative_humidity_min,
            "temp_avg_c": temp_avg_c,
            "temp_max_c": temp_max_c,
            "temp_min_c": temp_min_c,
            "soil_moisture": soil_moisture,
            "surface_temp_avg_c": surface_temp_avg_c,
            "surface_temp_max_c": surface_temp_max_c,
            "surface_temp_min_c": surface_temp_min_c,
            "pressure_kpa": pressure_kpa,
            "pressure_max_kpa": pressure_max_kpa,
            "pressure_min_kpa": pressure_min_kpa,
            "radiation_avg": radiation_avg,
            "radiation_max": radiation_max,
            "radiation_min": radiation_min,
            "soil_rel_humidity": soil_rel_humidity,
            "soil_temp_c": soil_temp_c,
        }

        daily_rows.append(daily_row)

        prev_soil_moisture = soil_moisture
        prev_soil_rel_humidity = soil_rel_humidity
        prev_soil_temp_c = soil_temp_c

    for row in daily_rows:
        validate_daily_row(row)

    return daily_rows


def get_next_7_full_days_forecast_by_latlon(
    lat: float,
    lon: float,
    last_history_row: dict[str, Any],
    app_key: str,
    app_secret: str,
) -> list[dict[str, float | str]]:
    """
    使用和风168h自动获取未来7个完整自然日。
    """
    start_datetime_str, hourly_steps = get_next_midnight_and_required_hours(full_days=7)

    print(f"\n[自动计算] 过滤起始时间: {start_datetime_str}")
    print(f"[自动计算] 预报总小时数: {hourly_steps}")

    return get_hourly_forecast_daily_rows_by_latlon(
        lat=lat,
        lon=lon,
        hourly_steps=hourly_steps,
        last_history_row=last_history_row,
        app_key=app_key,
        app_secret=app_secret,
        start_datetime_str=start_datetime_str,
    )


def get_hourly_forecast_daily_rows_by_latlon(
    lat: float,
    lon: float,
    hourly_steps: int,
    last_history_row: dict[str, Any],
    app_key: str,
    app_secret: str,
    start_datetime_str: str | None = None,
) -> list[dict[str, float | str]]:
    """
    使用和风168h小时预报 → 聚合成日尺度数据。
    """
    api_json = fetch_qweather_hourly_forecast_by_latlon(
        lat=lat,
        lon=lon,
        hourly_steps=hourly_steps,
    )

    hourly_rows = normalize_qweather_hourly_forecast_response(api_json)
    print(f"[调试] 和风预报逐小时记录条数: {len(hourly_rows)}")

    if start_datetime_str:
        hourly_rows = filter_hourly_rows_from_datetime(
            hourly_rows=hourly_rows,
            start_datetime_str=start_datetime_str,
        )

    print(f"[调试] 过滤后逐小时记录条数: {len(hourly_rows)}")

    date_counter = {}
    for row in hourly_rows:
        date_str = str(row["date"])
        date_counter[date_str] = date_counter.get(date_str, 0) + 1

    print("[调试] 各日期小时数：")
    for d in sorted(date_counter.keys()):
        print(d, date_counter[d])

    daily_rows = aggregate_hourly_rows_to_daily_rows(
        hourly_rows=hourly_rows,
        last_history_row=last_history_row,
        min_hours_per_day=24,
    )

    return daily_rows


def get_history_by_latlon(
    lat: float,
    lon: float,
    target_date_str: str,
    app_key: str,
    app_secret: str,
) -> list[dict[str, float | str]]:
    """
    使用和风历史API获取：前一天 + 目标日期 的逐小时数据。
    """
    target_dt = datetime.strptime(target_date_str, "%Y%m%d")
    prev_dt = target_dt - timedelta(days=1)

    date_list = [
        prev_dt.strftime("%Y%m%d"),
        target_dt.strftime("%Y%m%d"),
    ]

    all_rows: list[dict[str, float | str]] = []

    for d in date_list:
        api_json = fetch_qweather_history_hourly_by_latlon(
            lat=lat,
            lon=lon,
            target_date_str=d,
        )

        hourly_rows = normalize_qweather_history_hourly_response(api_json)
        all_rows.extend(hourly_rows)

    all_rows.sort(key=lambda x: str(x["datetime"]))
    return all_rows


def print_json_structure(data: Any, indent: int = 0, max_list_items: int = 1) -> None:
    """
    递归打印 JSON 结构，帮助查看 API 实际返回了哪些字段。
    """
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            value_type = type(value).__name__
            print(f"{prefix}{key}: {value_type}")
            if isinstance(value, (dict, list)):
                print_json_structure(value, indent + 1, max_list_items=max_list_items)

    elif isinstance(data, list):
        print(f"{prefix}[list] len={len(data)}")
        for idx, item in enumerate(data[:max_list_items]):
            print(f"{prefix}  [{idx}] -> {type(item).__name__}")
            if isinstance(item, (dict, list)):
                print_json_structure(item, indent + 2, max_list_items=max_list_items)


def save_json_to_file(data: Any, output_path: str | Path) -> None:
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("========== 和风168h预报 → 日聚合测试 ==========")
    lat = float(input("请输入纬度，例如 30.67：").strip())
    lon = float(input("请输入经度，例如 104.14：").strip())

    last_history_row = {
        "soil_moisture": 26.5,
        "soil_rel_humidity": 72.0,
        "soil_temp_c": 21.4,
    }

    # ===== 1. 直接获取 168h 原始 JSON =====
    start_datetime_str, hourly_steps = get_next_midnight_and_required_hours(full_days=7)

    forecast_raw_json = fetch_qweather_hourly_forecast_by_latlon(
        lat=lat,
        lon=lon,
        hourly_steps=hourly_steps,
    )
    save_json_to_file(
        forecast_raw_json,
        OUTPUTS_DIR / "qweather_168h_预测.json",
    )

    # ===== 2. 标准化后的逐小时 =====
    forecast_hourly_rows = normalize_qweather_hourly_forecast_response(forecast_raw_json)
    save_json_to_file(
        forecast_hourly_rows,
        OUTPUTS_DIR / "qweather_168h_预测标准化.json",
    )

    # ===== 3. 聚合后的未来日表 =====
    daily_rows = get_next_7_full_days_forecast_by_latlon(
        lat=lat,
        lon=lon,
        last_history_row=last_history_row,
        app_key="",
        app_secret="",
    )
    save_json_to_file(
        daily_rows,
        OUTPUTS_DIR / "qweather_未来7天预报.json",
    )

    print("\n【未来7天日尺度结果】")
    for row in daily_rows:
        print(row)

    print("\n========== 和风历史48小时测试 ==========")
    lat = float(input("请输入纬度，例如 30.67：").strip())
    lon = float(input("请输入经度，例如 104.14：").strip())
    target_date_str = input("请输入目标日期 YYYYMMDD，例如 20260329：").strip()

    # ===== 4. 历史 48h 原始 JSON（两天分别保存）=====
    target_dt = datetime.strptime(target_date_str, "%Y%m%d")
    prev_dt = target_dt - timedelta(days=1)

    prev_date_str = prev_dt.strftime("%Y%m%d")
    curr_date_str = target_dt.strftime("%Y%m%d")

    history_raw_prev_json = fetch_qweather_history_hourly_by_latlon(
        lat=lat,
        lon=lon,
        target_date_str=prev_date_str,
    )
    save_json_to_file(
        history_raw_prev_json,
        OUTPUTS_DIR / f"qweather_历史逐小时_{prev_date_str}.json",
    )

    history_raw_curr_json = fetch_qweather_history_hourly_by_latlon(
        lat=lat,
        lon=lon,
        target_date_str=curr_date_str,
    )
    save_json_to_file(
        history_raw_curr_json,
        OUTPUTS_DIR / f"qweather_历史逐小时_{curr_date_str}.json",
    )

    history_rows = get_history_by_latlon(
        lat=lat,
        lon=lon,
        target_date_str=target_date_str,
        app_key="",
        app_secret="",
    )
    print(f"\n【历史逐小时记录数】{len(history_rows)}")
    print("【前5条】")
    for row in history_rows[:5]:
        print(row)

    print("\nJSON 已保存到目录：")
    print(OUTPUTS_DIR)