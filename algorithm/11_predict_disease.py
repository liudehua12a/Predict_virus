from __future__ import annotations

from typing import Any
from pathlib import Path

import joblib
import numpy as np

import a_config as cfg
import d_model_training_testing as mt
import i_online_prediction_preparation as opp
import k_weather_data_storage as storage


BASE_DIR = Path(__file__).resolve().parent


def _parse_date_str(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()[:10]


def _model_type_compact(model_type: str) -> str:
    return (
        str(model_type or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("－", "-")
        .replace("—", "-")
    )


def load_full_bundle_for_disease(disease_key: str, model_type: str = "XGBoost"):
    """
    在线滚动预测使用的模型加载。

    当前规则：
    - XGBoost -> xg_full_bundle_{disease}.pt
    - LSTM -> lstm_full_bundle_{disease}.pt（若不存在则回退 full_bundle）
    - 融合模型 -> fus_full_bundle_{disease}.pt（若不存在则回退 full_bundle）
    """
    model_type_compact = _model_type_compact(model_type)

    if (
        ("xgboost" in model_type_compact)
        and ("lstm" not in model_type_compact)
        and ("fusion" not in model_type_compact)
        and ("融合" not in model_type_compact)
    ):
        model_dir = getattr(cfg, "MODEL_DIR_XGBOOST", getattr(cfg, "MODEL_DIR", BASE_DIR))
        candidate_names = [f"xg_full_bundle_{disease_key}.pt", f"full_bundle_{disease_key}.pt"]

    elif ("fusion" in model_type_compact) or ("融合" in model_type_compact) or (
        ("lstm" in model_type_compact) and ("xgboost" in model_type_compact)
    ):
        model_dir = getattr(cfg, "MODEL_DIR_FUSION", getattr(cfg, "MODEL_DIR", BASE_DIR))
        candidate_names = [
            f"fus_full_bundle_{disease_key}.pt",
            f"fusion_full_bundle_{disease_key}.pt",
            f"full_bundle_{disease_key}.pt",
        ]

    elif "lstm" in model_type_compact:
        model_dir = getattr(cfg, "MODEL_DIR_LSTM", getattr(cfg, "MODEL_DIR", BASE_DIR))
        candidate_names = [
            f"lstm_full_bundle_{disease_key}.pt",
            f"full_bundle_{disease_key}.pt",
        ]

    else:
        raise ValueError(f"不支持的 model_type: {model_type!r}")

    model_dir = Path(model_dir)
    bundle_path = None
    for name in candidate_names:
        p = model_dir / name
        if p.exists():
            bundle_path = p
            break

    if bundle_path is None:
        raise FileNotFoundError(
            f"未找到模型文件: disease_key={disease_key}, model_type={model_type!r}, "
            f"search_dir={model_dir}, candidates={candidate_names}"
        )

    print(f"[ModelSelect] model_type={model_type!r} -> 使用模型: {bundle_path}")
    return joblib.load(bundle_path)


def _get_initial_previous_targets(
    disease_key: str,
    last_observed_values: dict | None,
    init_alpha: float = 0.8,
    init_cap: float = 0.7,
):
    """
    首日 previous_targets 初始化（0~1口径）
    - 有真实值: min(真实值, init_cap) * init_alpha
    - 无真实值: 0
    """
    if not last_observed_values:
        return np.asarray([0.0, 0.0], dtype=np.float32)

    init_alpha = float(np.clip(init_alpha, 0.0, 1.0))
    init_cap = float(np.clip(init_cap, 0.0, 1.0))

    disease_cfg = cfg.DISEASE_CONFIGS[disease_key]
    target_1, target_2 = disease_cfg["targets"]

    value_1 = float(last_observed_values.get(target_1, 0.0)) / 100.0
    value_2 = float(last_observed_values.get(target_2, 0.0)) / 100.0

    value_1 = min(max(value_1, 0.0), init_cap) * init_alpha
    value_2 = min(max(value_2, 0.0), init_cap) * init_alpha

    return np.asarray([value_1, value_2], dtype=np.float32)


def _build_prediction_result_row(
    disease_key: str,
    row: dict,
    prev_targets: np.ndarray,
    pred_targets: np.ndarray,
):
    disease_cfg = cfg.DISEASE_CONFIGS[disease_key]
    target_1, target_2 = disease_cfg["targets"]

    pred_value_1 = float(pred_targets[0] * 100.0)
    pred_value_2 = float(pred_targets[1] * 100.0)
    prev_value_1 = float(prev_targets[0] * 100.0)
    prev_value_2 = float(prev_targets[1] * 100.0)

    return {
        "date": _parse_date_str(row["date"]),
        "site_id": row["site_id"],
        "disease_key": disease_key,
        "disease_cn": disease_cfg["cn_name"],
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
        "pred_overall_risk": cfg.combine_risk(pred_value_1, pred_value_2),
    }


def _rolling_forecast_next_n_days(
    bundle: dict,
    disease_key: str,
    future_rows: list[dict],
    last_observed_values: dict | None = None,
    init_alpha: float = 0.8,
    init_cap: float = 0.7,
):
    """
    逐日滚动预测：
    - 第一天：软初始化后的上一次真实值；没有则 0
    - 第二天及以后：前一天预测结果作为下一天输入
    """
    if not future_rows:
        return []

    previous_targets = _get_initial_previous_targets(
        disease_key=disease_key,
        last_observed_values=last_observed_values,
        init_alpha=init_alpha,
        init_cap=init_cap,
    )

    prediction_results = []
    for row in future_rows:
        pred_targets = mt.predict_row(
            bundle=bundle,
            row=row,
            previous_targets=previous_targets,
        )
        pred_targets = np.asarray(pred_targets, dtype=np.float32)

        prediction_results.append(
            _build_prediction_result_row(
                disease_key=disease_key,
                row=row,
                prev_targets=previous_targets,
                pred_targets=pred_targets,
            )
        )

        previous_targets = np.asarray(pred_targets, dtype=np.float32)

    return prediction_results


def run_all_diseases_prediction_and_save(
    site_id: int,
    batch_id: int,
    model_type: str,
    history_daily_rows: list[dict],
    forecast_daily_rows: list[dict],
    predict_dates: list[str],
    history_end_date_str: str,
    last_observed_by_disease: dict | None = None,
    init_alpha: float = 0.8,
    init_cap: float = 0.7,
):
    """
    在线滚动预测主入口。

    参数含义与 prediction.py 保持一致：
    - history_daily_rows: 历史逐日特征行
    - forecast_daily_rows: 未来天气逐日行
    - predict_dates: 需要预测的日期列表
    - history_end_date_str: 历史截止日期
    - last_observed_by_disease: 每种病害最近一次真实观测
    """
    if last_observed_by_disease is None:
        last_observed_by_disease = {}

    prediction_run_id = storage.build_prediction_run_id()

    storage.disable_current_prediction_rows(
        site_id=site_id,
        batch_id=batch_id,
        predict_dates=predict_dates,
    )

    all_results = {}
    disease_order = ["gray", "blight", "white"]

    print(
        f"[ForecastParams] init_alpha={init_alpha}, init_cap={init_cap}, monotonic_output=False"
    )

    future_rows = opp.build_future_prediction_rows(
        history_daily_rows=history_daily_rows,
        forecast_daily_rows=forecast_daily_rows,
        site_id=site_id,
        predict_dates=predict_dates,
    )

    for idx, disease_key in enumerate(disease_order):
        bundle = load_full_bundle_for_disease(disease_key, model_type=model_type)

        disease_last_observed = last_observed_by_disease.get(disease_key)

        disease_results = _rolling_forecast_next_n_days(
            bundle=bundle,
            disease_key=disease_key,
            future_rows=future_rows,
            last_observed_values=disease_last_observed,
            init_alpha=init_alpha,
            init_cap=init_cap,
        )

        allow_insert = (idx == 0)
        base_source_type = "observation" if disease_last_observed else "zero_init"

        storage.insert_or_update_prediction_rows_for_one_disease(
            prediction_run_id=prediction_run_id,
            site_id=site_id,
            batch_id=batch_id,
            disease_key=disease_key,
            prediction_results=disease_results,
            base_observation_date=history_end_date_str,
            base_source_type=base_source_type,
            base_source_id=None,
            allow_insert=allow_insert,
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
        "params": {
            "init_alpha": init_alpha,
            "init_cap": init_cap,
            "monotonic_output": False,
        },
    }