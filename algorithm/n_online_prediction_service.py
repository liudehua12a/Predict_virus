from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any
import j_online_rolling_forecast as forecast10
import k_weather_data_storage as storage
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


def build_last_observed_for_prediction_start(
        site_id: int,
        batch_id: int,
        yesterday_date: date,
) -> tuple[dict[str, dict[str, float] | None], str]:
    """
    决定 online prediction 的起点 previous_targets 来源：

    优先级：
    1. 昨天真实值（observation）
    2. 昨天 current 预测值（prediction）
    3. zero_init

    返回：
    - last_observed_by_disease
    - start_source_type: observation / prediction / zero_init
    """
    yesterday_str = yesterday_date.strftime("%Y-%m-%d")

    # 1) 优先查昨天真实值
    observation_row = storage.get_latest_observation_on_or_before_date(
        site_id=site_id,
        batch_id=batch_id,
        survey_date=yesterday_str,
    )

    if observation_row and normalize_date_str(observation_row["survey_date"]) == yesterday_str:
        return (
            storage.build_last_observed_by_disease_from_observation(observation_row),
            "observation",
        )

    # 2) 没有昨天真实值，再查昨天 current 预测值
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
        )

    # 3) 都没有，则 zero_init
    return (
        {
            "gray": None,
            "blight": None,
            "white": None,
        },
        "zero_init",
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

    last_observed_by_disease, start_source_type = build_last_observed_for_prediction_start(
        site_id=site_id,
        batch_id=batch_id,
        yesterday_date=yesterday_date,
    )

    all_output = forecast10.run_all_diseases_prediction_and_save(
        site_id=site_id,
        batch_id=batch_id,
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        predict_dates=predict_dates,
        history_end_date_str=history_end_date_str,
        last_observed_by_disease=last_observed_by_disease,
    )

    return {
        "site_id": site_id,
        "batch_id": batch_id,
        "model_type": cfg.ONLINE_MODEL_TYPE,
        "today_date": today_date.strftime("%Y-%m-%d"),
        "yesterday_date": yesterday_date.strftime("%Y-%m-%d"),
        "forecast_end_date": forecast_end_date_str,
        "predict_dates": predict_dates,
        "start_source_type": start_source_type,
        "prediction_run_id": all_output["prediction_run_id"],
        "results_by_disease": all_output["results_by_disease"],
    }
