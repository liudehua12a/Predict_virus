'''
    第一天用 0 或最近调查值
    第二天用前一天预测值
    一天一天滚动到第 7 天
'''
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from pathlib import Path

import joblib
import numpy as np
import a_config as cfg
import d_model_training_testing as mt
import i_online_prediction_preparation as opp
import l_history_padding_and_prediction_runner as runner13
import k_weather_data_storage as storage


def parse_date_str(value: Any) -> str:
    """
    统一转成 YYYY-MM-DD 字符串。
    """
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text[:10]


def get_initial_previous_targets(
    disease_key: str,
    last_observed_values: dict[str, float] | None,
) -> np.ndarray:
    """
    根据病害类型，确定滚动预测的初始 previous_targets。
    如果 last_observed_values 为空，则默认全 0。
    """
    if not last_observed_values:
        return np.asarray([0.0, 0.0], dtype=np.float32)

    disease_cfg = cfg.DISEASE_CONFIGS[disease_key]
    target_1 = disease_cfg["targets"][0]
    target_2 = disease_cfg["targets"][1]

    value_1 = float(last_observed_values.get(target_1, 0.0)) / 100.0
    value_2 = float(last_observed_values.get(target_2, 0.0)) / 100.0

    return np.asarray([value_1, value_2], dtype=np.float32)


def build_prediction_result_row(
    disease_key: str,
    row: dict[str, Any],
    prev_targets: np.ndarray,
    pred_targets: np.ndarray,
    model_type: str = "LSTM",
) -> dict[str, Any]:
    """
    将单日预测结果整理成便于前端/数据库使用的结构。
    pred_targets / prev_targets 均为 0~1，输出时转回 0~100。
    """
    disease_cfg = cfg.DISEASE_CONFIGS[disease_key]
    target_1 = disease_cfg["targets"][0]
    target_2 = disease_cfg["targets"][1]

    pred_value_1 = float(pred_targets[0] * 100.0)
    pred_value_2 = float(pred_targets[1] * 100.0)

    prev_value_1 = float(prev_targets[0] * 100.0)
    prev_value_2 = float(prev_targets[1] * 100.0)

    overall_risk = cfg.combine_risk(pred_value_1, pred_value_2)

    return {
        "date": parse_date_str(row["date"]),
        "site_id": row["site_id"],
        "disease_key": disease_key,
        "disease_cn": disease_cfg["cn_name"],
        "model_type": model_type,

        "prev_target_1_name": target_1,
        "prev_target_2_name": target_2,
        "prev_target_1_value": round(prev_value_1, 4),
        "prev_target_2_value": round(prev_value_2, 4),

        "pred_target_1_name": target_1,
        "pred_target_2_name": target_2,
        "pred_target_1_value": round(pred_value_1, 4),
        "pred_target_2_value": round(pred_value_2, 4),

        "pred_target_1_risk": cfg.classify_risk(pred_value_1),
        "pred_target_2_risk": cfg.classify_risk(pred_value_2),
        "pred_overall_risk": overall_risk,
    }


def rolling_forecast_next_n_days(
    bundle: dict[str, Any],
    disease_key: str,
    future_rows: list[dict[str, Any]],
    last_observed_values: dict[str, float] | None = None,
    model_type: str = "XGBoost",
) -> list[dict[str, Any]]:
    """
    对未来多天样本做逐日滚动预测。
    - 第一天：使用 last_observed_values；若为空则用 0
    - 第二天及以后：使用前一天预测值
    """
    if not future_rows:
        return []

    previous_targets = get_initial_previous_targets(
        disease_key=disease_key,
        last_observed_values=last_observed_values,
    )

    prediction_results: list[dict[str, Any]] = []

    for row in future_rows:
        pred_targets = mt.predict_row(
            bundle=bundle,
            row=row,
            previous_targets=previous_targets,
        )

        result_row = build_prediction_result_row(
            disease_key=disease_key,
            row=row,
            prev_targets=previous_targets,
            pred_targets=pred_targets,
            model_type=model_type,
        )
        prediction_results.append(result_row)

        previous_targets = np.asarray(pred_targets, dtype=np.float32)

    return prediction_results


def prepare_and_rolling_forecast(
    bundle: dict[str, Any],
    disease_key: str,
    history_daily_rows: list[dict[str, Any]],
    forecast_daily_rows: list[dict[str, Any]],
    site_id: int,
    predict_dates: list[str],
    last_observed_values: dict[str, float] | None = None,
    model_type: str = "XGBoost",
) -> list[dict[str, Any]]:
    """
    一站式函数：
    1. 构造未来样本行
    2. 做逐日滚动预测
    """
    future_rows = opp.build_future_prediction_rows(
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        site_id=site_id,
        predict_dates=predict_dates,
    )

    return rolling_forecast_next_n_days(
        bundle=bundle,
        disease_key=disease_key,
        future_rows=future_rows,
        last_observed_values=last_observed_values,
        model_type=model_type,
    )


def save_prediction_results_for_one_disease(
    prediction_results: list[dict[str, Any]],
    batch_id: int,
    base_observation_date: str | None,
    base_source_type: str,
    base_source_id: str | None = None,
    prediction_run_id: str | None = None,
    allow_insert: bool = False,
    model_type: str = "LSTM",
) -> str:
    """
    将当前单病害滚动预测结果写入 disease_prediction 宽表。
    如果 prediction_run_id 未传，则自动生成新的 run_id。
    """
    if not prediction_results:
        raise ValueError("prediction_results 不能为空")

    first_row = prediction_results[0]
    site_id = int(first_row["site_id"])
    disease_key = str(first_row["disease_key"])

    if not prediction_run_id:
        prediction_run_id = storage.build_prediction_run_id()

    storage.insert_or_update_prediction_rows_for_one_disease(
        prediction_run_id=prediction_run_id,
        site_id=site_id,
        batch_id=batch_id,
        model_type=model_type,
        disease_key=disease_key,
        prediction_results=prediction_results,
        base_observation_date=base_observation_date,
        base_source_type=base_source_type,
        base_source_id=base_source_id,
        allow_insert=allow_insert,
    )

    return prediction_run_id


def run_single_disease_prediction_and_save(
    site_id: int,
    batch_id: int,
    disease_key: str,
    history_daily_rows: list[dict[str, Any]],
    forecast_daily_rows: list[dict[str, Any]],
    predict_dates: list[str],
    history_end_date_str: str,
    prediction_run_id: str,
    last_observed_values: dict[str, float] | None = None,
    allow_insert: bool = False,
    model_type: str = "XGBoost",
) -> list[dict[str, Any]]:
    """
    执行单个病害的：
    1. 加载模型
    2. 滚动预测
    3. 落库到 disease_prediction 宽表对应列
    """
    bundle = load_full_bundle_for_disease(disease_key, model_type=model_type)

    prediction_results = prepare_and_rolling_forecast(
        bundle=bundle,
        disease_key=disease_key,
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        site_id=site_id,
        predict_dates=predict_dates,
        last_observed_values=last_observed_values,
        model_type=model_type,
    )

    if last_observed_values:
        base_source_type = "observation"
    else:
        base_source_type = "zero_init"

    save_prediction_results_for_one_disease(
        prediction_results=prediction_results,
        batch_id=batch_id,
        base_observation_date=history_end_date_str,
        base_source_type=base_source_type,
        base_source_id=None,
        prediction_run_id=prediction_run_id,
        allow_insert=allow_insert,
        model_type=model_type,
    )

    return prediction_results


def run_all_diseases_prediction_and_save(
    site_id: int,
    batch_id: int,
    history_daily_rows: list[dict[str, Any]],
    forecast_daily_rows: list[dict[str, Any]],
    predict_dates: list[str],
    history_end_date_str: str,
    last_observed_by_disease: dict[str, dict[str, float] | None] | None = None,
    model_type: str = "XGBoost",
) -> dict[str, list[dict[str, Any]]]:
    """
    顺序执行三种病害预测，并共用同一个 prediction_run_id。
    最终会把三种病害写到同一批宽表记录中。
    """
    if last_observed_by_disease is None:
        last_observed_by_disease = {}

    prediction_run_id = storage.build_prediction_run_id()
    # 🔥 关键：只在整批 run 开始时做一次版本失效
    storage.disable_current_prediction_rows(
        site_id=site_id,
        batch_id=batch_id,
        model_type=model_type,
        predict_dates=predict_dates,
    )
    all_results: dict[str, list[dict[str, Any]]] = {}

    disease_order = ["gray", "blight", "white"]

    for idx, disease_key in enumerate(disease_order):
        disease_last_observed = last_observed_by_disease.get(disease_key)
        allow_insert = (idx == 0)   # 只有第一个病害允许插入新行

        disease_results = run_single_disease_prediction_and_save(
            site_id=site_id,
            batch_id=batch_id,
            disease_key=disease_key,
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            predict_dates=predict_dates,
            history_end_date_str=history_end_date_str,
            prediction_run_id=prediction_run_id,
            last_observed_values=disease_last_observed,
            allow_insert=allow_insert,
            model_type=model_type,
        )

        all_results[disease_key] = disease_results

    storage.validate_prediction_run_completeness(
        prediction_run_id=prediction_run_id,
        site_id=site_id,
        batch_id=batch_id,
    )

    return {
        "prediction_run_id": prediction_run_id,
        "results_by_disease": all_results,
    }


def load_full_bundle_for_disease(disease_key: str, model_type: str = "XGBoost") -> dict[str, Any]:
    """
    加载某个病害对应的 full bundle。
    """
    compact = str(model_type or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    load_as_xgboost = False

    if ("xgboost" in compact) and ("lstm" not in compact) and ("fusion" not in compact) and ("融合" not in compact):
        model_dir = getattr(cfg, "MODEL_DIR_XGBOOST", cfg.MODULE_DIR / "models" / "Xgboost")
        candidates = [
            model_dir / f"xg_full_bundle_{disease_key}.pt",
        ]
        load_as_xgboost = True
    elif ("fusion" in compact) or ("融合" in compact) or (("lstm" in compact) and ("xgboost" in compact)):
        model_dir = getattr(cfg, "MODEL_DIR_LSTM_XGBOOST", cfg.MODULE_DIR / "models" / "lstm+xgboost")
        candidates = [
            model_dir / f"fus_full_bundle_{disease_key}.pt",
        ]
    else:
        model_dir = getattr(cfg, "MODEL_DIR_LSTM", getattr(cfg, "MODEL_DIR", cfg.MODULE_DIR / "models" / "lstm"))
        candidates = [
            model_dir / f"lstm_full_bundle_{disease_key}.pt",
            model_dir / f"full_bundle_{disease_key}.pt",
        ]

    bundle_path = next((p for p in candidates if p.exists()), None)
    if bundle_path is None:
        raise FileNotFoundError(
            f"未找到模型文件: disease_key={disease_key}, model_type={model_type!r}, candidates={[str(p) for p in candidates]}"
        )

    print(f"[ModelSelect] model_type={model_type!r} -> 使用模型: {bundle_path}")
    if load_as_xgboost:
        return joblib.load(str(bundle_path), mmap_mode=None)

    return mt.load_bundle(bundle_path)

def rebuild_predictions_after_observation(
    site_id: int,
    batch_id: int,
    observation_date_str: str,
    forecast_horizon_days: int = 7,
    model_type: str = "XGBoost",
) -> dict[str, Any]:
    """
    当某条真实调查值入库后：
    1. 读取该真实值
    2. 判断是否与同日 current 预测冲突
    3. 若冲突，则从 observation_date + 1 开始重算未来 forecast_horizon_days 天
    4. 写入新的 prediction_run_id
    """
    observation_row = storage.get_latest_observation_on_or_before_date(
        site_id=site_id,
        batch_id=batch_id,
        survey_date=observation_date_str,
    )
    if not observation_row:
        raise ValueError(
            f"未找到真实调查记录: site_id={site_id}, batch_id={batch_id}, survey_date<={observation_date_str}"
        )

    current_prediction_same_day = storage.get_current_prediction_by_date(
        site_id=site_id,
        batch_id=batch_id,
        predict_date=observation_row["survey_date"],
        model_type=model_type,
    )

    has_conflict = storage.observation_conflicts_with_current_prediction(
        observation_row=observation_row,
        prediction_row=current_prediction_same_day,
    )

    if not has_conflict:
        return {
            "recalculated": False,
            "reason": "observation_matches_current_prediction",
            "observation_date": observation_row["survey_date"],
            "prediction_run_id": None,
        }

    next_start_date = (
        datetime.strptime(observation_row["survey_date"], "%Y-%m-%d").date()
        + timedelta(days=1)
    )
    next_end_date = next_start_date + timedelta(days=forecast_horizon_days - 1)

    forecast_start_date_str = next_start_date.strftime("%Y-%m-%d")
    forecast_end_date_str = next_end_date.strftime("%Y-%m-%d")

    # Step 1: 历史补齐到真实调查日
    history_daily_rows = runner13.ensure_min_history_days(
        site_id=site_id,
        end_date_str=observation_row["survey_date"],
        min_days=runner13.MIN_HISTORY_DAYS,
    )

    # Step 2: 读取未来预报日表
    forecast_daily_rows = runner13.get_future_forecast_daily_rows_from_db(
        site_id=site_id,
        start_date_str=forecast_start_date_str,
        end_date_str=forecast_end_date_str,
    )

    predict_dates = runner13.build_predict_dates_from_forecast_rows(forecast_daily_rows)

    # Step 3: 用这条真实调查值作为新的预测起点
    last_observed_by_disease = storage.build_last_observed_by_disease_from_observation(
        observation_row
    )

    # Step 4: 先把从 next_start_date 开始的旧 current 版本失效
    storage.disable_current_predictions_from_date(
        site_id=site_id,
        batch_id=batch_id,
        model_type=model_type,
        start_date_str=forecast_start_date_str,
    )

    # Step 5: 重新生成未来预测
    all_output = run_all_diseases_prediction_and_save(
        site_id=site_id,
        batch_id=batch_id,
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        predict_dates=predict_dates,
        history_end_date_str=observation_row["survey_date"],
        last_observed_by_disease=last_observed_by_disease,
        model_type=model_type,
    )

    return {
        "recalculated": True,
        "reason": "observation_conflicts_with_prediction",
        "observation_date": observation_row["survey_date"],
        "rebuild_start_date": forecast_start_date_str,
        "rebuild_end_date": forecast_end_date_str,
        "prediction_run_id": all_output["prediction_run_id"],
        "results_by_disease": all_output["results_by_disease"],
    }
