from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any
import j_online_rolling_forecast as forecast10
import k_weather_data_storage as storage
from k_weather_data_storage import get_data_staleness_threshold
import l_history_padding_and_prediction_runner as runner13
from pathlib import Path
import a_config as cfg


def normalize_date_str(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()[:10]


def resolve_today_date(today_date: date | str | None = None) -> date:
    """
    统一解析 today_date：
    - None: 使用系统当前日期
    - date: 直接返回
    - str: 解析 YYYY-MM-DD
    """
    if today_date is None:
        return datetime.now().date()

    if isinstance(today_date, date):
        return today_date

    return datetime.strptime(str(today_date).strip()[:10], "%Y-%m-%d").date()


def get_yesterday_date(today_date: date) -> date:
    return today_date - timedelta(days=1)


def get_most_recent_prediction(
    site_id: int,
    batch_id: int,
    model_type: str,
) -> dict[str, Any] | None:
    """
    获取最近一次预测记录，不限制日期。
    """
    sql = """
    SELECT *
    FROM disease_prediction
    WHERE site_id = ?
      AND batch_id = ?
      AND model_type = ?
      AND is_current = 1
    ORDER BY predict_date DESC
    LIMIT 1
    ;
    """
    with storage.closing(storage.get_connection()) as conn:
        row = conn.execute(
            sql,
            (site_id, batch_id, model_type),
        ).fetchone()
    return dict(row) if row else None


def build_last_observed_for_prediction_start(
    site_id: int,
    batch_id: int,
    yesterday_date: date,
) -> tuple[dict[str, dict[str, float] | None], str, float]:
    """
    决定 online prediction 的起点 previous_targets 来源，并确定 stage_code。

    previous_targets 来源：
    1. 昨天真实值 observation
    2. 昨天 current 预测值 prediction
    3. 最近一次预测值 prediction（需在7天内）
    4. zero_init

    stage_code 来源：
    从 site_id + batch_id 在 yesterday 及以前最近一次真实调查记录中读取 growth_stage。
    例如 growth_stage=V10，则 stage_code=10.0。
    """
    yesterday_str = yesterday_date.strftime("%Y-%m-%d")

    stage_code = storage.get_latest_stage_code_on_or_before_date(
        site_id=site_id,
        batch_id=batch_id,
        survey_date=yesterday_str,
    )

    observation_row = storage.get_latest_observation_on_or_before_date(
        site_id=site_id,
        batch_id=batch_id,
        survey_date=yesterday_str,
    )

    if observation_row and normalize_date_str(observation_row["survey_date"]) == yesterday_str:
        return (
            storage.build_last_observed_by_disease_from_observation(observation_row),
            "observation",
            stage_code,
        )

    prediction_row = storage.get_current_prediction_by_date(
        site_id=site_id,
        batch_id=batch_id,
        predict_date=yesterday_str,
        model_type=cfg.ONLINE_MODEL_TYPE,
    )

    if prediction_row:
        return (
            storage.build_last_observed_by_disease_from_prediction(prediction_row),
            "prediction",
            stage_code,
        )

    # 昨天真实值和预测值都不存在，查找最近一次预测值
    recent_prediction = get_most_recent_prediction(
        site_id=site_id,
        batch_id=batch_id,
        model_type=cfg.ONLINE_MODEL_TYPE,
    )

    if recent_prediction:
        recent_predict_date = datetime.strptime(
            normalize_date_str(recent_prediction["predict_date"]), "%Y-%m-%d"
        ).date()
        days_diff = (yesterday_date - recent_predict_date).days
        threshold = get_data_staleness_threshold()
        if threshold is not None and days_diff > threshold:
            raise ValueError("数据过久，请及时更新数据")
        return (
            storage.build_last_observed_by_disease_from_prediction(recent_prediction),
            "recent_prediction",
            stage_code,
        )

    return (
        {
            "gray": None,
            "blight": None,
            "white": None,
        },
        "zero_init",
        stage_code,
    )


def load_weather_context_for_online_prediction(
        site_id: int,
        today_date: date,
        forecast_days: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str, str]:
    """
    为 today ~ today+forecast_days-1 的在线预测准备天气上下文：
    - history_daily_rows: 截止 yesterday 的最近21天历史天气
    - forecast_daily_rows: today ~ today+6 的未来预报
    - predict_dates: 待预测日期列表
    - history_end_date_str
    - forecast_end_date_str
    """
    yesterday_date = get_yesterday_date(today_date)
    history_end_date_str = yesterday_date.strftime("%Y-%m-%d")
    forecast_start_date_str = today_date.strftime("%Y-%m-%d")
    forecast_end_date_str = (today_date + timedelta(days=forecast_days - 1)).strftime("%Y-%m-%d")

    history_daily_rows = runner13.ensure_min_history_days(
        site_id=site_id,
        end_date_str=history_end_date_str,
        min_days=runner13.MIN_HISTORY_DAYS,
    )

    forecast_daily_rows = runner13.get_future_forecast_daily_rows_from_db(
        site_id=site_id,
        start_date_str=forecast_start_date_str,
        end_date_str=forecast_end_date_str,
    )

    predict_dates = runner13.build_predict_dates_from_forecast_rows(forecast_daily_rows)

    return (
        history_daily_rows,
        forecast_daily_rows,
        predict_dates,
        history_end_date_str,
        forecast_end_date_str,
    )


def run_online_prediction_for_today(
        site_id: int,
        model_type: str,
        batch_id: int,
        today_date: date | str | None = None,
        forecast_days: int = 7,
) -> dict[str, Any]:
    """
    界面点击“预测”时调用的统一入口。

    逻辑：
    1. today 默认取系统当前日期
    2. 预测窗口：today ~ today+6
    3. previous_targets 来源优先级：
       昨天真实值 > 昨天current预测值 > zero_init
    4. 最终调用 10_online_rolling_forecast.run_all_diseases_prediction_and_save
    """
    today_date = resolve_today_date(today_date)
    yesterday_date = get_yesterday_date(today_date)

    (
        history_daily_rows,
        forecast_daily_rows,
        predict_dates,
        history_end_date_str,
        forecast_end_date_str,
    ) = load_weather_context_for_online_prediction(
        site_id=site_id,
        today_date=today_date,
        forecast_days=forecast_days,
    )

    last_observed_by_disease, start_source_type, stage_code = build_last_observed_for_prediction_start(
        site_id=site_id,
        batch_id=batch_id,
        yesterday_date=yesterday_date,
    )

    model_type_text = str(model_type or "").strip().lower()
    is_xgboost = ("xgboost" in model_type_text) and ("lstm" not in model_type_text)
    is_fusion = ("fusion" in model_type_text) or ("融合" in model_type_text) or (
        ("lstm" in model_type_text) and ("xgboost" in model_type_text)
    )

    if is_xgboost:
        import x_xgboost_prediction as xgb_prediction

        all_output = xgb_prediction.run_all_diseases_prediction_and_save(
            site_id=site_id,
            batch_id=batch_id,
            model_type=model_type,
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            predict_dates=predict_dates,
            history_end_date_str=history_end_date_str,
            last_observed_by_disease=last_observed_by_disease,
        )
    elif is_fusion:
        import importlib
        fusion_prediction = importlib.import_module("f_fusion_predict")

        run_fn = getattr(
            fusion_prediction,
            "run_all_diseases_prediction_and_save",
            getattr(fusion_prediction, "call_run_all_diseases_prediction_and_save", None),
        )
        if run_fn is None:
            raise AttributeError("f_fusion_predict 缺少预测入口函数")

        all_output = run_fn(
            site_id=site_id,
            batch_id=batch_id,
            model_type=model_type,
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            predict_dates=predict_dates,
            history_end_date_str=history_end_date_str,
            last_observed_by_disease=last_observed_by_disease,
        )
    else:
        all_output = forecast10.run_all_diseases_prediction_and_save(
            site_id=site_id,
            batch_id=batch_id,
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            predict_dates=predict_dates,
            history_end_date_str=history_end_date_str,
            stage_code=stage_code,
            last_observed_by_disease=last_observed_by_disease,
        )

    return {
        "site_id": site_id,
        "batch_id": batch_id,
        "model_type": model_type,
        "today_date": today_date.strftime("%Y-%m-%d"),
        "yesterday_date": yesterday_date.strftime("%Y-%m-%d"),
        "forecast_end_date": forecast_end_date_str,
        "predict_dates": predict_dates,
        "start_source_type": start_source_type,
        "stage_code": stage_code,
        "prediction_run_id": all_output["prediction_run_id"],
        "results_by_disease": all_output["results_by_disease"],
    }
