from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import uuid
import h_qweather_api as weather_api

# ======分区1：连接与基础工具======
# ===== 数据库文件路径 =====

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "nky-CornPre.db"
DATA_SOURCE_PRIORITY = {
    "mock": 1,
    "forecast_daily": 2,
    "forecast_hourly": 3,
    "history": 4,
}

def get_db_path() -> Path:
    """
    返回 SQLite 数据库路径。
    """
    return DB_PATH

def get_connection() -> sqlite3.Connection:
    """
    获取 SQLite 连接。
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def normalize_date_str(date_value: Any) -> str:
    """
    将输入统一转成 YYYY-MM-DD 字符串。
    """
    if hasattr(date_value, "strftime"):
        return date_value.strftime("%Y-%m-%d")

    text = str(date_value).strip()
    return text[:10]


# =========================================================
# 2. weather_daily：建表、写入、查询
# =========================================================
def create_weather_daily_table() -> None:
    """
    创建 weather_daily 表。
    所有点位共用一张表，用 site_id + date 做唯一键。
    """
    sql = """
    CREATE TABLE IF NOT EXISTS weather_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        site_id INTEGER NOT NULL,
        date TEXT NOT NULL,

        wind_avg REAL,
        wind_max REAL,
        wind_min REAL,

        precip_sum REAL,
        precip_max REAL,
        precip_min REAL,

        relative_humidity REAL,
        relative_humidity_max REAL,
        relative_humidity_min REAL,

        temp_avg_c REAL,
        temp_max_c REAL,
        temp_min_c REAL,

        soil_moisture REAL,

        surface_temp_avg_c REAL,
        surface_temp_max_c REAL,
        surface_temp_min_c REAL,

        pressure_kpa REAL,
        pressure_max_kpa REAL,
        pressure_min_kpa REAL,

        radiation_avg REAL,
        radiation_max REAL,
        radiation_min REAL,

        soil_rel_humidity REAL,
        soil_temp_c REAL,

        data_source TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,

        UNIQUE(site_id, date)
    );
    """

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(sql)

def get_data_source_priority(data_source: str | None) -> int:
    """
    获取数据源优先级。
    未知来源按最低优先级处理。
    """
    if not data_source:
        return 0
    return int(DATA_SOURCE_PRIORITY.get(str(data_source), 0))

def validate_weather_daily_row(row: dict[str, Any]) -> None:
    """
    校验待写入数据库的日天气记录字段是否完整。
    """
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
            raise ValueError(f"待写入数据库的天气日记录缺少字段: {field_name}")

def get_weather_daily_row_by_site_and_date(
    site_id: int,
    date_str: str,
) -> dict[str, Any] | None:
    """
    查询某点位某一天的 weather_daily 记录。
    """
    sql = """
    SELECT *
    FROM weather_daily
    WHERE site_id = ?
      AND date = ?
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (site_id, normalize_date_str(date_str)),
        ).fetchone()

    return dict(row) if row else None

def should_replace_existing_row(
    existing_row: dict[str, Any] | None,
    new_data_source: str,
) -> bool:
    """
    判断是否允许用新数据覆盖旧记录。

    规则：
    1. 旧记录不存在 -> 允许插入
    2. 新数据优先级高于旧数据 -> 允许覆盖
    3. 新旧优先级相同 -> 允许覆盖（视为刷新同类来源数据）
    4. 新数据优先级低于旧数据 -> 不允许覆盖
    """
    if existing_row is None:
        return True

    old_source = existing_row.get("data_source")
    old_priority = get_data_source_priority(old_source)
    new_priority = get_data_source_priority(new_data_source)

    return new_priority >= old_priority

def upsert_weather_daily_rows(
    site_id: int,
    daily_rows: list[dict[str, Any]],
    data_source: str,
) -> None:
    """
    将一批日尺度天气记录写入 weather_daily。

    规则：
    - 若 site_id + date 不存在，则插入
    - 若已存在，则只有当新 data_source 优先级 >= 旧 data_source 优先级时才覆盖
    """
    if not daily_rows:
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = """
    INSERT INTO weather_daily (
        site_id, date,

        wind_avg, wind_max, wind_min,
        precip_sum, precip_max, precip_min,

        relative_humidity, relative_humidity_max, relative_humidity_min,

        temp_avg_c, temp_max_c, temp_min_c,

        soil_moisture,

        surface_temp_avg_c, surface_temp_max_c, surface_temp_min_c,

        pressure_kpa, pressure_max_kpa, pressure_min_kpa,

        radiation_avg, radiation_max, radiation_min,

        soil_rel_humidity, soil_temp_c,

        data_source, created_at, updated_at
    ) VALUES (
        ?, ?,

        ?, ?, ?,
        ?, ?, ?,

        ?, ?, ?,

        ?, ?, ?,

        ?,

        ?, ?, ?,

        ?, ?, ?,

        ?, ?, ?,

        ?, ?,

        ?, ?, ?
    )
    ON CONFLICT(site_id, date) DO UPDATE SET
        wind_avg = excluded.wind_avg,
        wind_max = excluded.wind_max,
        wind_min = excluded.wind_min,

        precip_sum = excluded.precip_sum,
        precip_max = excluded.precip_max,
        precip_min = excluded.precip_min,

        relative_humidity = excluded.relative_humidity,
        relative_humidity_max = excluded.relative_humidity_max,
        relative_humidity_min = excluded.relative_humidity_min,

        temp_avg_c = excluded.temp_avg_c,
        temp_max_c = excluded.temp_max_c,
        temp_min_c = excluded.temp_min_c,

        soil_moisture = excluded.soil_moisture,

        surface_temp_avg_c = excluded.surface_temp_avg_c,
        surface_temp_max_c = excluded.surface_temp_max_c,
        surface_temp_min_c = excluded.surface_temp_min_c,

        pressure_kpa = excluded.pressure_kpa,
        pressure_max_kpa = excluded.pressure_max_kpa,
        pressure_min_kpa = excluded.pressure_min_kpa,

        radiation_avg = excluded.radiation_avg,
        radiation_max = excluded.radiation_max,
        radiation_min = excluded.radiation_min,

        soil_rel_humidity = excluded.soil_rel_humidity,
        soil_temp_c = excluded.soil_temp_c,

        data_source = excluded.data_source,
        updated_at = excluded.updated_at
    ;
    """

    params_list: list[tuple[Any, ...]] = []

    for row in daily_rows:
        validate_weather_daily_row(row)

        row_date_str = normalize_date_str(row["date"])
        existing_row = get_weather_daily_row_by_site_and_date(
            site_id=site_id,
            date_str=row_date_str,
        )

        if not should_replace_existing_row(existing_row, data_source):
            old_source = existing_row.get("data_source") if existing_row else None
            print(
                f"[跳过覆盖] site_id={site_id}, date={row_date_str}, "
                f"旧来源={old_source}, 新来源={data_source}"
            )
            continue

        params = (
            site_id,
            row_date_str,

            float(row["wind_avg"]),
            float(row["wind_max"]),
            float(row["wind_min"]),

            float(row["precip_sum"]),
            float(row["precip_max"]),
            float(row["precip_min"]),

            float(row["relative_humidity"]),
            float(row["relative_humidity_max"]),
            float(row["relative_humidity_min"]),

            float(row["temp_avg_c"]),
            float(row["temp_max_c"]),
            float(row["temp_min_c"]),

            float(row["soil_moisture"]),

            float(row["surface_temp_avg_c"]),
            float(row["surface_temp_max_c"]),
            float(row["surface_temp_min_c"]),

            float(row["pressure_kpa"]),
            float(row["pressure_max_kpa"]),
            float(row["pressure_min_kpa"]),

            float(row["radiation_avg"]),
            float(row["radiation_max"]),
            float(row["radiation_min"]),

            float(row["soil_rel_humidity"]),
            float(row["soil_temp_c"]),

            data_source,
            now_str,
            now_str,
        )
        params_list.append(params)

    if not params_list:
        print(f"[提示] 本次没有可写入的 weather_daily 记录，data_source={data_source}")
        return

    with closing(get_connection()) as conn:
        with conn:
            conn.executemany(sql, params_list)

def get_recent_weather_daily_rows(
    site_id: int,
    end_date_str: str,
    n_days: int,
) -> list[dict[str, Any]]:
    """
    查询某点位截止到 end_date_str（含）最近 n_days 天的日尺度天气记录。
    返回按日期升序排列的 list[dict]。
    """
    sql = """
    SELECT
        site_id, date,

        wind_avg, wind_max, wind_min,
        precip_sum, precip_max, precip_min,

        relative_humidity, relative_humidity_max, relative_humidity_min,

        temp_avg_c, temp_max_c, temp_min_c,

        soil_moisture,

        surface_temp_avg_c, surface_temp_max_c, surface_temp_min_c,

        pressure_kpa, pressure_max_kpa, pressure_min_kpa,

        radiation_avg, radiation_max, radiation_min,

        soil_rel_humidity, soil_temp_c,

        data_source, created_at, updated_at
    FROM weather_daily
    WHERE site_id = ?
      AND date <= ?
    ORDER BY date DESC
    LIMIT ?
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(sql, (site_id, normalize_date_str(end_date_str), n_days)).fetchall()

    result = [dict(row) for row in rows]
    result.sort(key=lambda x: x["date"])
    return result

def get_weather_daily_rows_by_date_range(
    site_id: int,
    start_date_str: str,
    end_date_str: str,
) -> list[dict[str, Any]]:
    """
    查询某点位在指定日期范围内的日尺度天气记录。
    返回按日期升序排列。
    """
    sql = """
    SELECT
        site_id, date,

        wind_avg, wind_max, wind_min,
        precip_sum, precip_max, precip_min,

        relative_humidity, relative_humidity_max, relative_humidity_min,

        temp_avg_c, temp_max_c, temp_min_c,

        soil_moisture,

        surface_temp_avg_c, surface_temp_max_c, surface_temp_min_c,

        pressure_kpa, pressure_max_kpa, pressure_min_kpa,

        radiation_avg, radiation_max, radiation_min,

        soil_rel_humidity, soil_temp_c,

        data_source, created_at, updated_at
    FROM weather_daily
    WHERE site_id = ?
      AND date >= ?
      AND date <= ?
    ORDER BY date ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(
            sql,
            (
                site_id,
                normalize_date_str(start_date_str),
                normalize_date_str(end_date_str),
            ),
        ).fetchall()

    return [dict(row) for row in rows]

def get_future_forecast_daily_rows(
    site_id: int,
    start_date_str: str,
    end_date_str: str,
) -> list[dict[str, Any]]:
    """
    查询某点位未来预报日表。
    这里本质上仍然是从 weather_daily 查，
    但只保留 forecast_hourly / forecast_daily 来源的数据。
    """
    sql = """
    SELECT
        site_id, date,

        wind_avg, wind_max, wind_min,
        precip_sum, precip_max, precip_min,

        relative_humidity, relative_humidity_max, relative_humidity_min,

        temp_avg_c, temp_max_c, temp_min_c,

        soil_moisture,

        surface_temp_avg_c, surface_temp_max_c, surface_temp_min_c,

        pressure_kpa, pressure_max_kpa, pressure_min_kpa,

        radiation_avg, radiation_max, radiation_min,

        soil_rel_humidity, soil_temp_c,

        data_source, created_at, updated_at
    FROM weather_daily
    WHERE site_id = ?
      AND date >= ?
      AND date <= ?
      AND data_source IN ('forecast_hourly', 'forecast_daily')
    ORDER BY date ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(
            sql,
            (
                site_id,
                normalize_date_str(start_date_str),
                normalize_date_str(end_date_str),
            ),
        ).fetchall()

    return [dict(row) for row in rows]

def upsert_next_7_full_days_forecast_to_db(
    site_id: int,
    lat: float,
    lon: float,
    history_end_date_str: str,
) -> list[dict[str, Any]]:
    """
    将未来 168h 聚合后的完整自然日日表写入 weather_daily。
    
    history_end_date_str:
    - 历史数据截止日期，例如 '2026-03-29'
    - 会先查截止该日期的最近历史，取最后一条作为土壤/地表递推初值
    """
    history_rows = get_recent_weather_daily_rows(
        site_id=site_id,
        end_date_str=history_end_date_str,
        n_days=1,
    )
    if not history_rows:
        raise ValueError(
            f"点位 {site_id} 截止 {history_end_date_str} 没有历史日表，无法生成未来预报日表。"
        )

    last_history_row = history_rows[-1]

    forecast_daily_rows = weather_api.get_next_7_full_days_forecast_by_latlon(
        lat=lat,
        lon=lon,
        last_history_row=last_history_row,
        app_key="",
        app_secret="",
    )

    upsert_weather_daily_rows(
        site_id=site_id,
        daily_rows=forecast_daily_rows,
        data_source="forecast_hourly",
    )

    return forecast_daily_rows

def count_weather_daily_rows(site_id: int | None = None) -> int:
    """
    统计 weather_daily 表中的记录数。
    """
    if site_id is None:
        sql = "SELECT COUNT(*) AS cnt FROM weather_daily;"
        params = ()
    else:
        sql = "SELECT COUNT(*) AS cnt FROM weather_daily WHERE site_id = ?;"
        params = (site_id,)

    with closing(get_connection()) as conn:
        row = conn.execute(sql, params).fetchone()

    return int(row["cnt"])


# =====分区3：disease_prediction相关=====
def create_disease_prediction_table() -> None:
    """
    按 1.sql 中的定义创建 disease_prediction 表。
    不自行改表结构。
    """
    sql = """
    CREATE TABLE IF NOT EXISTS disease_prediction (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,   -- 预测结果ID
        prediction_run_id TEXT NOT NULL,                   -- 预测批次运行ID
        site_id INTEGER NOT NULL,                          -- 点位ID
        batch_id INTEGER NOT NULL,                         -- 批次ID
        model_type TEXT,                                   -- 预测使用的模型类型（例如 LSTM、XGBoost 等） 
        predict_date TEXT NOT NULL,                        -- 预测日期

        gray_incidence REAL,                               -- 灰斑病预测发病株率
        gray_index REAL,                                   -- 灰斑病预测病情指数
        gray_risk_level TEXT,                              -- 灰斑病风险等级

        blight_incidence REAL,                             -- 大斑病预测发病株率
        blight_index REAL,                                 -- 大斑病预测病情指数
        blight_risk_level TEXT,                            -- 大斑病风险等级

        white_incidence REAL,                              -- 白斑病预测发病株率
        white_index REAL,                                  -- 白斑病预测病情指数
        white_risk_level TEXT,                             -- 白斑病风险等级

        base_observation_date TEXT,                        -- 启动本轮预测所依据的基准日期
        base_source_type TEXT NOT NULL,                    -- 基准来源类型（observation/prediction/zero_init）
        base_source_id TEXT,                               -- 基准来源ID

        is_current INTEGER NOT NULL DEFAULT 1,             -- 是否当前有效版本（1=当前有效，0=历史版本）
        created_at TEXT NOT NULL,                          -- 创建时间
        updated_at TEXT NOT NULL,                          -- 更新时间

        FOREIGN KEY (site_id) REFERENCES site_info(site_id),
        FOREIGN KEY (batch_id) REFERENCES survey_batch(batch_id)
    );
    """

    sql_index_1 = """
    CREATE INDEX IF NOT EXISTS idx_disease_prediction_site_batch_date
    ON disease_prediction(site_id, batch_id, predict_date);
    """

    sql_index_2 = """
    CREATE INDEX IF NOT EXISTS idx_disease_prediction_run_id
    ON disease_prediction(prediction_run_id);
    """

    sql_index_3 = """
    CREATE INDEX IF NOT EXISTS idx_disease_prediction_current
    ON disease_prediction(site_id, batch_id, is_current, predict_date);
    """

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(sql)
            conn.execute(sql_index_1)
            conn.execute(sql_index_2)
            conn.execute(sql_index_3)

def validate_single_disease_prediction_row(row: dict[str, Any]) -> None:
    """
    校验 10_online_rolling_forecast.py 返回的单病害预测行。
    """
    required_fields = [
        "date",
        "site_id",
        "disease_key",
        "model_type",
        "pred_target_1_value",
        "pred_target_2_value",
        "pred_overall_risk",
    ]
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"预测结果缺少字段: {field_name}")

def build_prediction_update_fields(
    disease_key: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    把当前单病害 prediction_results 的一行，映射成 disease_prediction 宽表对应字段。
    """
    disease_key = str(disease_key).strip()

    if disease_key == "gray":
        return {
            "gray_incidence": float(row["pred_target_1_value"]),
            "gray_index": float(row["pred_target_2_value"]),
            "gray_risk_level": row["pred_overall_risk"],
        }

    if disease_key == "blight":
        return {
            "blight_incidence": float(row["pred_target_1_value"]),
            "blight_index": float(row["pred_target_2_value"]),
            "blight_risk_level": row["pred_overall_risk"],
        }

    if disease_key == "white":
        return {
            "white_incidence": float(row["pred_target_1_value"]),
            "white_index": float(row["pred_target_2_value"]),
            "white_risk_level": row["pred_overall_risk"],
        }

    raise ValueError(f"未知 disease_key: {disease_key}")

def disable_current_prediction_rows(
    site_id: int,
    batch_id: int,
    model_type: str,
    predict_dates: list[str],
) -> None:
    """
    将同 site + batch + predict_date 的旧 current 版本置为历史版本。
    注意：
    这里是“整批失效”，为新的 prediction_run_id 腾位置。
    """
    if not predict_dates:
        return

    placeholders = ",".join(["?"] * len(predict_dates))
    sql = f"""
    UPDATE disease_prediction
    SET
        is_current = 0,
        updated_at = ?
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND predict_date IN ({placeholders})
      AND is_current = 1
    ;
    """

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(sql, (now_str, site_id, batch_id, model_type, *predict_dates))

def insert_or_update_prediction_rows_for_one_disease(
    prediction_run_id: str,
    site_id: int,
    batch_id: int,
    model_type: str,
    disease_key: str,
    prediction_results: list[dict[str, Any]],
    base_observation_date: str | None,
    base_source_type: str,
    base_source_id: str | None = None,
    allow_insert: bool = False,
) -> None:
    """
    将单病害 prediction_results 落到 disease_prediction 宽表。

    规则：
    1. 这批 run 使用同一个 prediction_run_id
    2. 先把同 site + batch + predict_date 的旧 current 记录失效
    3. 再按当前 disease_key 写入对应列
    4. 如果同一个 run/date 行已存在，则只补当前病害列
    """
    if not prediction_results:
        return

    create_disease_prediction_table()

    for row in prediction_results:
        validate_single_disease_prediction_row(row)
        if int(row["site_id"]) != int(site_id):
            raise ValueError(
                f"site_id 不一致，期望 {site_id}，实际 {row['site_id']}"
            )
        if str(row["disease_key"]) != str(disease_key):
            raise ValueError(
                f"disease_key 不一致，期望 {disease_key}，实际 {row['disease_key']}"
            )

    predict_dates = sorted({normalize_date_str(row["date"]) for row in prediction_results})


    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    select_sql = """
    SELECT prediction_id
    FROM disease_prediction
    WHERE prediction_run_id = ?
      AND site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND predict_date = ?
    LIMIT 1
    ;
    """

    insert_sql = """
    INSERT INTO disease_prediction (
        prediction_run_id,
        site_id,
        batch_id,
        model_type,
        predict_date,

        gray_incidence,
        gray_index,
        gray_risk_level,

        blight_incidence,
        blight_index,
        blight_risk_level,

        white_incidence,
        white_index,
        white_risk_level,

        base_observation_date,
        base_source_type,
        base_source_id,

        is_current,
        created_at,
        updated_at
    ) VALUES (
        ?, ?, ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?
    )
    ;
    """

    update_sql = """
    UPDATE disease_prediction
    SET
        gray_incidence = COALESCE(?, gray_incidence),
        gray_index = COALESCE(?, gray_index),
        gray_risk_level = COALESCE(?, gray_risk_level),

        blight_incidence = COALESCE(?, blight_incidence),
        blight_index = COALESCE(?, blight_index),
        blight_risk_level = COALESCE(?, blight_risk_level),

        white_incidence = COALESCE(?, white_incidence),
        white_index = COALESCE(?, white_index),
        white_risk_level = COALESCE(?, white_risk_level),

        base_observation_date = ?,
        base_source_type = ?,
        base_source_id = ?,
        is_current = 1,
        updated_at = ?
    WHERE prediction_run_id = ?
      AND site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND predict_date = ?
    ;
    """

    with closing(get_connection()) as conn:
        with conn:
            for row in prediction_results:
                predict_date = normalize_date_str(row["date"])
                mapped = build_prediction_update_fields(disease_key, row)

                existing = conn.execute(
                    select_sql,
                    (prediction_run_id, site_id, batch_id, model_type,predict_date),
                ).fetchone()

                gray_incidence = mapped.get("gray_incidence")
                gray_index = mapped.get("gray_index")
                gray_risk_level = mapped.get("gray_risk_level")

                blight_incidence = mapped.get("blight_incidence")
                blight_index = mapped.get("blight_index")
                blight_risk_level = mapped.get("blight_risk_level")

                white_incidence = mapped.get("white_incidence")
                white_index = mapped.get("white_index")
                white_risk_level = mapped.get("white_risk_level")

                if existing is None:
                    if not allow_insert:
                        raise RuntimeError(
                            f"宽表写入错误：prediction_run_id={prediction_run_id}, "
                            f"site_id={site_id}, batch_id={batch_id}, predict_date={predict_date} "
                            f"在写入 disease_key={disease_key} 时未找到已有行。"
                            f"只有首个病害允许插入新行，后续病害只能更新已有行。"
                        )

                    conn.execute(
                        insert_sql,
                        (
                            prediction_run_id,
                            site_id,
                            batch_id,
                            model_type,
                            predict_date,

                            gray_incidence,
                            gray_index,
                            gray_risk_level,

                            blight_incidence,
                            blight_index,
                            blight_risk_level,

                            white_incidence,
                            white_index,
                            white_risk_level,

                            base_observation_date,
                            base_source_type,
                            base_source_id,

                            1,
                            now_str,
                            now_str,
                        ),
                    )
                else:
                    conn.execute(
                        update_sql,
                        (
                            gray_incidence,
                            gray_index,
                            gray_risk_level,

                            blight_incidence,
                            blight_index,
                            blight_risk_level,

                            white_incidence,
                            white_index,
                            white_risk_level,

                            base_observation_date,
                            base_source_type,
                            base_source_id,
                            now_str,

                            prediction_run_id,
                            site_id,
                            batch_id,
                            model_type,
                            predict_date,
                        ),
                    )

def build_prediction_run_id() -> str:
    """
    生成一次预测运行ID。
    """
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = uuid.uuid4().hex[:8]
    return f"predrun_{now_str}_{random_part}"

def get_current_prediction_rows(
    site_id: int,
    batch_id: int,
    model_type: str,
) -> list[dict[str, Any]]:
    """
    查询当前有效版本预测结果。
    """
    sql = """
    SELECT *
    FROM disease_prediction
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND is_current = 1
    ORDER BY predict_date ASC, prediction_id ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(sql, (site_id, batch_id, model_type)).fetchall()

    return [dict(row) for row in rows]

def validate_prediction_run_completeness(
    prediction_run_id: str,
    site_id: int,
    batch_id: int,
) -> None:
    """
    校验某个 prediction_run_id 在宽表中是否每一行都已填满三种病害。
    若存在 NULL，则直接报错，防止脏数据继续往后流。
    """
    sql = """
    SELECT
        prediction_id,
        predict_date,
        gray_incidence,
        gray_index,
        gray_risk_level,
        blight_incidence,
        blight_index,
        blight_risk_level,
        white_incidence,
        white_index,
        white_risk_level
    FROM disease_prediction
    WHERE prediction_run_id = ?
      AND site_id = ?
      AND batch_id = ?
    ORDER BY predict_date ASC, prediction_id ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(sql, (prediction_run_id, site_id, batch_id)).fetchall()

    for row in rows:
        row_dict = dict(row)
        required_fields = [
            "gray_incidence", "gray_index", "gray_risk_level",
            "blight_incidence", "blight_index", "blight_risk_level",
            "white_incidence", "white_index", "white_risk_level",
        ]
        missing_fields = [field for field in required_fields if row_dict.get(field) is None]
        if missing_fields:
            raise RuntimeError(
                f"宽表校验失败：prediction_run_id={prediction_run_id}, "
                f"predict_date={row_dict['predict_date']} 存在未填满字段: {missing_fields}"
            )

def get_current_prediction_rows_by_date_range(
    site_id: int,
    batch_id: int,
    start_date_str: str,
    end_date_str: str,
    model_type: str,
) -> list[dict[str, Any]]:
    """
    查询当前有效版本中，某个日期范围内的预测记录。
    """
    sql = """
    SELECT *
    FROM disease_prediction
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND is_current = 1
      AND predict_date >= ?
      AND predict_date <= ?
    ORDER BY predict_date ASC, prediction_id ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(
            sql,
            (
                site_id,
                batch_id,
                model_type,
                normalize_date_str(start_date_str),
                normalize_date_str(end_date_str),
            ),
        ).fetchall()

    return [dict(row) for row in rows]

def disable_current_predictions_from_date(
    site_id: int,
    batch_id: int,
    model_type: str,
    start_date_str: str,
) -> None:
    """
    将某个批次从指定日期开始的当前预测版本置为历史版本。
    用于真实值回填后触发重算。
    """
    sql = """
    UPDATE disease_prediction
    SET
        is_current = 0,
        updated_at = ?
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND is_current = 1
      AND predict_date >= ?
    ;
    """

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                sql,
                (
                    now_str,
                    site_id,
                    batch_id,
                    model_type,
                    normalize_date_str(start_date_str),
                ),
            )

def get_current_prediction_by_date(
    site_id: int,
    batch_id: int,
    model_type: str,
    predict_date: str,
) -> dict[str, Any] | None:
    """
    查询某批次某一天当前有效预测记录。
    """
    sql = """
    SELECT *
    FROM disease_prediction
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND predict_date = ?
      AND is_current = 1
    ORDER BY prediction_id DESC
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (
                site_id,
                batch_id,
                model_type,
                normalize_date_str(predict_date),
            ),
        ).fetchone()

    return dict(row) if row else None

def get_current_prediction_run_by_date(
    site_id: int,
    batch_id: int,
    model_type: str,
    predict_date: str,
) -> dict[str, Any] | None:
    """
    查询某个批次某一天当前有效预测对应的 run 基本信息。
    """
    sql = """
    SELECT
        prediction_run_id,
        site_id,
        batch_id,
        model_type,
        predict_date,
        base_observation_date,
        base_source_type,
        is_current
    FROM disease_prediction
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND predict_date = ?
      AND is_current = 1
    ORDER BY prediction_id DESC
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (site_id, batch_id, model_type,normalize_date_str(predict_date)),
        ).fetchone()

    return dict(row) if row else None

def get_prediction_rows_by_run_id(
    prediction_run_id: str,
    site_id: int,
    batch_id: int,
    model_type: str,
) -> list[dict[str, Any]]:
    """
    查询某个 prediction_run_id 对应的整批预测结果。
    """
    sql = """
    SELECT *
    FROM disease_prediction
    WHERE prediction_run_id = ?
      AND site_id = ?
      AND batch_id = ?
      AND model_type = ?
    ORDER BY predict_date ASC, prediction_id ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(
            sql,
            (prediction_run_id, site_id, batch_id, model_type),
        ).fetchall()

    return [dict(row) for row in rows]

# =====分区4：disease_observation 相关=====
def get_latest_observation_for_batch(
    site_id: int,
    batch_id: int,
) -> dict[str, Any] | None:
    """
    获取某个 site + batch 最近一次真实调查记录
    """
    sql = """
    SELECT *
    FROM disease_observation
    WHERE site_id = ?
      AND batch_id = ?
    ORDER BY survey_date DESC, observation_id DESC
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(sql, (site_id, batch_id)).fetchone()

    return dict(row) if row else None

def get_survey_batch_by_name(batch_name: str) -> dict[str, Any] | None:
    """
    根据 batch_name 查询 survey_batch。
    """
    sql = """
    SELECT *
    FROM survey_batch
    WHERE batch_name = ?
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(sql, (str(batch_name).strip(),)).fetchone()

    return dict(row) if row else None

def insert_disease_observation_row(row: dict[str, Any]) -> int:
    """
    写入一条真实调查记录到 disease_observation。
    返回新插入的 observation_id。
    不做模板校验，只假定传入 row 已经是可入库结构。
    """
    required_fields = [
        "site_id",
        "batch_id",
        "survey_date",
        "crop_variety",
        "growth_stage",
        "source_file_name",
        "source_row_no",
        "gray_incidence",
        "gray_index",
        "blight_incidence",
        "blight_index",
        "white_incidence",
        "white_index",
    ]
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"disease_observation 入库缺少字段: {field_name}")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    site_id = int(row["site_id"])
    batch_id = int(row["batch_id"])
    survey_date = normalize_date_str(row["survey_date"])

    # 旧逻辑（仅插入，不处理重复）
    # sql = """
    # INSERT INTO disease_observation (
    #     site_id,
    #     batch_id,
    #     survey_date,
    #     crop_variety,
    #     growth_stage,
    #     source_file_name,
    #     source_row_no,
    #     gray_incidence,
    #     gray_index,
    #     blight_incidence,
    #     blight_index,
    #     white_incidence,
    #     white_index,
    #     created_at,
    #     updated_at
    # ) VALUES (
    #     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    # )
    # ;
    # """
    #
    # params = (
    #     int(row["site_id"]),
    #     int(row["batch_id"]),
    #     normalize_date_str(row["survey_date"]),
    #     row["crop_variety"],
    #     row["growth_stage"],
    #     row["source_file_name"],
    #     int(row["source_row_no"]),
    #     None if row["gray_incidence"] is None else float(row["gray_incidence"]),
    #     None if row["gray_index"] is None else float(row["gray_index"]),
    #     None if row["blight_incidence"] is None else float(row["blight_incidence"]),
    #     None if row["blight_index"] is None else float(row["blight_index"]),
    #     None if row["white_incidence"] is None else float(row["white_incidence"]),
    #     None if row["white_index"] is None else float(row["white_index"]),
    #     now_str,
    #     now_str,
    # )
    #
    # with closing(get_connection()) as conn:
    #     with conn:
    #         cursor = conn.execute(sql, params)
    #         return int(cursor.lastrowid)

    with closing(get_connection()) as conn:
        with conn:
            existing_row = conn.execute(
                """
                SELECT observation_id
                FROM disease_observation
                WHERE site_id = ?
                  AND batch_id = ?
                  AND survey_date = ?
                ORDER BY observation_id DESC
                LIMIT 1
                ;
                """,
                (site_id, batch_id, survey_date),
            ).fetchone()

            if existing_row:
                observation_id = int(existing_row["observation_id"])
                conn.execute(
                    """
                    UPDATE disease_observation
                    SET
                        crop_variety = ?,
                        growth_stage = ?,
                        source_file_name = ?,
                        source_row_no = ?,
                        gray_incidence = ?,
                        gray_index = ?,
                        blight_incidence = ?,
                        blight_index = ?,
                        white_incidence = ?,
                        white_index = ?,
                        updated_at = ?
                    WHERE observation_id = ?
                    ;
                    """,
                    (
                        row["crop_variety"],
                        row["growth_stage"],
                        row["source_file_name"],
                        int(row["source_row_no"]),
                        None if row["gray_incidence"] is None else float(row["gray_incidence"]),
                        None if row["gray_index"] is None else float(row["gray_index"]),
                        None if row["blight_incidence"] is None else float(row["blight_incidence"]),
                        None if row["blight_index"] is None else float(row["blight_index"]),
                        None if row["white_incidence"] is None else float(row["white_incidence"]),
                        None if row["white_index"] is None else float(row["white_index"]),
                        now_str,
                        observation_id,
                    ),
                )
                return observation_id

            cursor = conn.execute(
                """
                INSERT INTO disease_observation (
                    site_id,
                    batch_id,
                    survey_date,
                    crop_variety,
                    growth_stage,
                    source_file_name,
                    source_row_no,
                    gray_incidence,
                    gray_index,
                    blight_incidence,
                    blight_index,
                    white_incidence,
                    white_index,
                    created_at,
                    updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ;
                """,
                (
                    site_id,
                    batch_id,
                    survey_date,
                    row["crop_variety"],
                    row["growth_stage"],
                    row["source_file_name"],
                    int(row["source_row_no"]),
                    None if row["gray_incidence"] is None else float(row["gray_incidence"]),
                    None if row["gray_index"] is None else float(row["gray_index"]),
                    None if row["blight_incidence"] is None else float(row["blight_incidence"]),
                    None if row["blight_index"] is None else float(row["blight_index"]),
                    None if row["white_incidence"] is None else float(row["white_incidence"]),
                    None if row["white_index"] is None else float(row["white_index"]),
                    now_str,
                    now_str,
                ),
            )
            return int(cursor.lastrowid)
        
def insert_disease_observation_rows(rows: list[dict[str, Any]]) -> list[int]:
    """
    批量写入真实调查记录。
    返回 observation_id 列表。
    """
    inserted_ids: list[int] = []
    for row in rows:
        observation_id = insert_disease_observation_row(row)
        inserted_ids.append(observation_id)
    return inserted_ids

def get_latest_observation_on_or_before_date(
    site_id: int,
    batch_id: int,
    survey_date: str,
) -> dict[str, Any] | None:
    """
    查询某个批次在指定日期及以前最近的一条真实调查记录。
    用于“从该真实日期+1开始重算”时，确定新的预测起点。
    """
    sql = """
    SELECT *
    FROM disease_observation
    WHERE site_id = ?
      AND batch_id = ?
      AND survey_date <= ?
    ORDER BY survey_date DESC, observation_id DESC
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (site_id, batch_id, normalize_date_str(survey_date)),
        ).fetchone()

    return dict(row) if row else None


# =====分区5：真实值/预测值转换与冲突 相关=====
def build_last_observed_by_disease_from_observation(
    observation_row: dict[str, Any] | None,
) -> dict[str, dict[str, float] | None]:
    """
    将 observation 转换为 rolling 预测所需格式
    """
    if not observation_row:
        return {
            "gray": None,
            "blight": None,
            "white": None,
        }

    return {
        "gray": {
            "gray_incidence": observation_row.get("gray_incidence"),
            "gray_index": observation_row.get("gray_index"),
        },
        "blight": {
            "blight_incidence": observation_row.get("blight_incidence"),
            "blight_index": observation_row.get("blight_index"),
        },
        "white": {
            "white_incidence": observation_row.get("white_incidence"),
            "white_index": observation_row.get("white_index"),
        },
    }

def build_last_observed_by_disease_from_prediction(
    prediction_row: dict[str, Any] | None,
) -> dict[str, dict[str, float] | None]:
    """
    将 disease_prediction 的一行 current 预测结果
    转换为 rolling 预测需要的 last_observed_by_disease 结构。
    """
    if not prediction_row:
        return {
            "gray": None,
            "blight": None,
            "white": None,
        }

    return {
        "gray": {
            "gray_incidence": prediction_row.get("gray_incidence"),
            "gray_index": prediction_row.get("gray_index"),
        },
        "blight": {
            "blight_incidence": prediction_row.get("blight_incidence"),
            "blight_index": prediction_row.get("blight_index"),
        },
        "white": {
            "white_incidence": prediction_row.get("white_incidence"),
            "white_index": prediction_row.get("white_index"),
        },
    }

def observation_conflicts_with_current_prediction(
    observation_row: dict[str, Any],
    prediction_row: dict[str, Any] | None,
    tolerance: float = 1e-6,
) -> bool:
    """
    判断某条真实调查值是否与同日 current 预测结果冲突。
    只要任一病害任一指标有差异，就视为冲突。
    """
    if prediction_row is None:
        return True

    check_pairs = [
        ("gray_incidence", "gray_incidence"),
        ("gray_index", "gray_index"),
        ("blight_incidence", "blight_incidence"),
        ("blight_index", "blight_index"),
        ("white_incidence", "white_incidence"),
        ("white_index", "white_index"),
    ]

    for obs_field, pred_field in check_pairs:
        obs_value = observation_row.get(obs_field)
        pred_value = prediction_row.get(pred_field)

        if obs_value is None and pred_value is None:
            continue
        if obs_value is None or pred_value is None:
            return True

        if abs(float(obs_value) - float(pred_value)) > tolerance:
            return True

    return False


    print("数据库路径：", get_db_path())

    create_weather_daily_table()
    create_disease_prediction_table()
    print("weather_daily 表已创建或已存在。")
    print("disease_prediction 表已创建或已存在。")

    site_id = 6
    lat = 30.67
    lon = 104.14
    history_end_date_str = "2026-03-29"

    print("\n=== Step 1: 写入未来 168h 聚合日表 ===")
    forecast_rows = upsert_next_7_full_days_forecast_to_db(
        site_id=site_id,
        lat=lat,
        lon=lon,
        history_end_date_str=history_end_date_str,
    )

    print(f"写入 forecast_hourly 日表条数: {len(forecast_rows)}")
    for row in forecast_rows:
        print(row["date"], "forecast_hourly")

    print("\n=== Step 2: 从数据库查未来预报日表 ===")
    future_rows = get_future_forecast_daily_rows(
        site_id=site_id,
        start_date_str="2026-04-01",
        end_date_str="2026-04-10",
    )

    for row in future_rows:
        print(row["date"], row["data_source"])

    print("\n=== 当前 weather_daily 总记录数 ===")
    print(count_weather_daily_rows())


#  ======分区6：site_info  survey_batch相关 ======相关=====
def get_site_by_name_and_coords(
    site_name: str,
    lat: float,
    lon: float,
) -> dict[str, Any] | None:
    """
    根据 site_name + lat + lon 查询 site_info。
    """
    sql = """
    SELECT *
    FROM site_info
    WHERE site_name = ?
      AND lat = ?
      AND lon = ?
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (str(site_name).strip(), float(lat), float(lon)),
        ).fetchone()

    return dict(row) if row else None

def insert_site_info_row(row: dict[str, Any]) -> int:
    """
    插入一条 site_info，返回 site_id。
    若 site_name + lat + lon 已存在，则直接返回已有 site_id。
    """
    required_fields = ["site_name", "lat", "lon"]
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"site_info 入库缺少字段: {field_name}")

    existing = get_site_by_name_and_coords(
        site_name=row["site_name"],
        lat=row["lat"],
        lon=row["lon"],
    )
    if existing:
        return int(existing["site_id"])

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = """
    INSERT INTO site_info (
        province,
        city,
        site_name,
        lat,
        lon,
        elevation,
        location_id,
        is_active,
        created_at,
        updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ;
    """

    params = (
        None if row.get("province") is None else str(row["province"]).strip(),
        None if row.get("city") is None else str(row["city"]).strip(),
        str(row["site_name"]).strip(),
        float(row["lat"]),
        float(row["lon"]),
        None if row.get("elevation") is None else float(row["elevation"]),
        row.get("location_id"),
        1,
        now_str,
        now_str,
    )

    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid)
        
def get_survey_batch_by_site_and_name(
    site_id: int,
    batch_name: str,
) -> dict[str, Any] | None:
    """
    根据 site_id + batch_name 查询 survey_batch。
    """
    sql = """
    SELECT *
    FROM survey_batch
    WHERE site_id = ?
      AND batch_name = ?
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (int(site_id), str(batch_name).strip()),
        ).fetchone()

    return dict(row) if row else None

def insert_survey_batch_row(row: dict[str, Any]) -> int:
    """
    插入一条 survey_batch，返回 batch_id。
    若 site_id + batch_name 已存在，则直接返回已有 batch_id。
    """
    required_fields = ["site_id", "batch_name"]
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"survey_batch 入库缺少字段: {field_name}")

    existing = get_survey_batch_by_site_and_name(
        site_id=row["site_id"],
        batch_name=row["batch_name"],
    )
    if existing:
        return int(existing["batch_id"])

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = """
    INSERT INTO survey_batch (
        site_id,
        batch_name,
        batch_code,
        crop_variety,
        sowing_date,
        survey_start_date,
        survey_end_date,
        is_active,
        created_at,
        updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ;
    """

    params = (
        int(row["site_id"]),
        str(row["batch_name"]).strip(),
        row.get("batch_code"),
        None if row.get("crop_variety") is None else str(row["crop_variety"]).strip(),
        None if row.get("sowing_date") is None else normalize_date_str(row["sowing_date"]),
        row.get("survey_start_date"),
        row.get("survey_end_date"),
        1,
        now_str,
        now_str,
    )

    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid)
        
def get_all_active_sites() -> list[dict[str, Any]]:
    """
    读取所有启用中的点位，返回 site_id / site_name / lat / lon / elevation。
    只返回经纬度完整的点位。
    """
    sql = """
    SELECT
        site_id,
        site_name,
        province,
        city,
        lat,
        lon,
        elevation,
        is_active
    FROM site_info
    WHERE is_active = 1
      AND lat IS NOT NULL
      AND lon IS NOT NULL
    ORDER BY site_id ASC
    ;
    """

    with closing(get_connection()) as conn:
        rows = conn.execute(sql).fetchall()

    return [dict(row) for row in rows]

def get_site_batch_by_names(
    site_name: str,
    batch_name: str,
) -> dict[str, Any] | None:
    """
    根据 site_name + batch_name 联查 site_info 和 survey_batch，
    返回 site_id / batch_id 等信息。
    """
    sql = """
    SELECT
        s.site_id,
        s.site_name,
        b.batch_id,
        b.batch_name,
        b.crop_variety
    FROM survey_batch b
    JOIN site_info s
      ON b.site_id = s.site_id
    WHERE s.site_name = ?
      AND b.batch_name = ?
    LIMIT 1
    ;
    """

    with closing(get_connection()) as conn:
        row = conn.execute(
            sql,
            (str(site_name).strip(), str(batch_name).strip()),
        ).fetchone()

    return dict(row) if row else None
def get_weather_data(site_id: int, days: int = 7) -> list[dict[str, Any]]:
    """
    从weather_daily表获取指定站点未来几天的天气数据
    """
    # 计算今天的日期
    today = datetime.now().strftime('%Y-%m-%d')

    sql = """
          SELECT date, temp_avg_c, relative_humidity,precip_sum
          FROM weather_daily
          WHERE site_id = ? AND date >= ?
          ORDER BY date ASC
              LIMIT ?
          ; \
          """


    with closing(get_connection()) as conn:
        rows = conn.execute(sql, (site_id, normalize_date_str(today), days)).fetchall()

    weather_data = []
    for row in rows:
        date, temp, humidity ,rain= row
        # 根据湿度确定天气类型
        if rain >=0.1:
            weather_icon = '☂'  # 雨天
        elif humidity >=70:
            weather_icon = '☁'  # 多云
        else:
            weather_icon = '☀'  # 晴天
        weather_data.append({
            'date': date,
            'temp': temp,
            'humidity': humidity,
            'icon': weather_icon
        })

    return weather_data
