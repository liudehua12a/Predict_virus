from __future__ import annotations

from typing import Any

import joblib
import numpy as np

import a_config as cfg
import d_model_training_testing as mt
import i_online_prediction_preparation as opp
import k_weather_data_storage as storage


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
    model_type_compact = _model_type_compact(model_type)

    is_xgboost_model = (
            ("xgboost" in model_type_compact)
            and ("lstm" not in model_type_compact)
            and ("fusion" not in model_type_compact)
            and ("融合" not in model_type_compact)
    )

    if not is_xgboost_model:
        raise ValueError(
            f"prediction.py 仅支持 XGBoost 在线滚动预测，当前 model_type={model_type!r}"
        )

    bundle_path = cfg.MODEL_DIR_XGBOOST / f"xg_full_bundle_{disease_key}.pt"
    print(f"[ModelSelect] model_type={model_type!r} -> 使用XGBoost模型: {bundle_path}")

    if not bundle_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {bundle_path}")

    # 1. 安全地将 PosixPath 转换为字符串并加载
    bundle = joblib.load(str(bundle_path),mmap_mode=None)

    # 2. 直接返回加载好的字典对象
    return bundle


def _get_initial_previous_targets(
    disease_key: str,
    last_observed_values: dict | None,
    init_alpha: float = 1.0,
    init_cap: float = 1.0,
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


def _normalize_xgb_output_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 2:
        return values
    max_val = float(np.max(values))
    if 1.5 < max_val <= 100.0:
        return values / 100.0
    return values


def _apply_white_smoothing(
    pred_targets: np.ndarray,
    prev_targets: np.ndarray,
    max_growth: float = 0.05,
    smooth_alpha: float = 0.6,
) -> np.ndarray:
    pred_targets = np.asarray(pred_targets, dtype=np.float32)
    prev_targets = np.asarray(prev_targets, dtype=np.float32)
    delta = np.clip(pred_targets - prev_targets, -max_growth, max_growth)
    smoothed = prev_targets + delta
    return smooth_alpha * smoothed + (1.0 - smooth_alpha) * pred_targets


def _build_prediction_result_row(
    disease_key: str,
    row: dict,
    prev_targets: np.ndarray,
    pred_targets: np.ndarray,
    model_type: str,
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
        "pred_overall_risk": cfg.combine_risk(pred_value_1, pred_value_2),
    }


def _rolling_forecast_next_n_days(
    bundle: dict,
    disease_key: str,
    future_rows: list[dict],
    observed_by_date: dict[str, dict[str, Any]] | None = None,
    last_observed_values: dict | None = None,
    init_alpha: float = 0.8,
    init_cap: float = 0.7,
    model_type: str = "XGBoost",
):
    """
    逐日滚动预测：
    - 第一天：软初始化后的上一次真实值；没有则 0
    - 第二天及以后：前一天预测结果作为下一天输
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
        date_str = _parse_date_str(row.get("date"))
        observed_row = observed_by_date.get(date_str) if observed_by_date else None

        if observed_row:
            disease_cfg = cfg.DISEASE_CONFIGS[disease_key]
            target_1, target_2 = disease_cfg["targets"]
            obs_1 = observed_row.get(target_1)
            obs_2 = observed_row.get(target_2)

            if obs_1 is not None and obs_2 is not None:
                pred_targets = np.asarray(
                    [float(obs_1) / 100.0, float(obs_2) / 100.0],
                    dtype=np.float32,
                )
            else:
                pred_targets = mt.predict_row(
                    bundle=bundle,
                    row=row,
                    previous_targets=previous_targets,
                )
                pred_targets = np.asarray(pred_targets, dtype=np.float32)
        else:
            pred_targets = mt.predict_row(
                bundle=bundle,
                row=row,
                previous_targets=previous_targets,
            )
            pred_targets = np.asarray(pred_targets, dtype=np.float32)

        pred_targets = _normalize_xgb_output_scale(pred_targets)

        # 如果是观测日，直接以观测值作为预测结果（不强制单调），
        # 否则对模型输出应用单调不降限制，确保预测值不会比前一天降低。
        if disease_key == "white" and not observed_row:
            pred_targets = _apply_white_smoothing(pred_targets, previous_targets)

        if not observed_row:
            pred_targets = np.maximum(pred_targets, previous_targets)

        prediction_results.append(
            _build_prediction_result_row(
                disease_key=disease_key,
                row=row,
                prev_targets=previous_targets,
                pred_targets=pred_targets,
                model_type=model_type,
            )
        )

        previous_targets = np.asarray(pred_targets, dtype=np.float32)

    return prediction_results


def _build_future_rows_daily(
    history_daily_rows: list[dict],
    forecast_daily_rows: list[dict],
    site_id: int,
    batch_id: int,
    predict_dates: list[str],
) -> list[dict]:
    """
    按预测日期逐日构造 future_row。
    每一天使用该日最新生育期编码，避免整段复用同一个 stage_code。
    """
    rows: list[dict] = []
    for date_str in predict_dates:
        stage_code = storage.get_latest_stage_code_on_or_before_date(
            site_id=site_id,
            batch_id=batch_id,
            survey_date=date_str,
        )
        day_rows = opp.build_future_prediction_rows(
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            site_id=site_id,
            predict_dates=[date_str],
            stage_code=stage_code,
        )
        if day_rows:
            rows.append(day_rows[0])
    return rows




def run_all_diseases_prediction_and_save(
    site_id: int,
    batch_id: int,
    model_type: str,
    history_daily_rows: list[dict],
    forecast_daily_rows: list[dict],
    predict_dates: list[str],
    history_end_date_str: str,
    last_observed_by_disease: dict | None = None,
    init_alpha: float = 1.0,
    init_cap: float = 1.0,
):
    if last_observed_by_disease is None:
        last_observed_by_disease = {}

    prediction_run_id = storage.build_prediction_run_id()

    storage.disable_current_prediction_rows(
        site_id=site_id,
        batch_id=batch_id,
        model_type=model_type,
        predict_dates=predict_dates,
    )

    all_results = {}
    disease_order = ["gray", "blight", "white"]

    observed_by_date: dict[str, dict[str, Any]] = {}
    if predict_dates:
        observation_rows = storage.get_observation_rows_between_dates(
            site_id=site_id,
            batch_id=batch_id,
            start_date=predict_dates[0],
            end_date=predict_dates[-1],
        )
        for obs in observation_rows:
            observed_by_date[_parse_date_str(obs.get("survey_date"))] = obs

    print(
        f"[ForecastParams] init_alpha={init_alpha}, init_cap={init_cap}, monotonic_output=False"
    )

    for idx, disease_key in enumerate(disease_order):
        bundle = load_full_bundle_for_disease(disease_key, model_type=model_type)

        future_rows = _build_future_rows_daily(
            history_daily_rows=history_daily_rows,
            forecast_daily_rows=forecast_daily_rows,
            site_id=site_id,
            batch_id=batch_id,
            predict_dates=predict_dates,
        )

        for row in future_rows:
            stage_code = storage.get_latest_stage_code_on_or_before_date(
                site_id=site_id,
                batch_id=batch_id,
                survey_date=_parse_date_str(row.get("date")),
            )
            row["stage_code"] = stage_code
            row["growth_stage_code"] = stage_code


    

        disease_last_observed = last_observed_by_disease.get(disease_key)

        disease_results = _rolling_forecast_next_n_days(
            bundle=bundle,
            disease_key=disease_key,
            future_rows=future_rows,
            observed_by_date=observed_by_date,
            last_observed_values=disease_last_observed,
            init_alpha=init_alpha,
            init_cap=init_cap,
            model_type=model_type,
        )

        for row in disease_results:
            if isinstance(row, dict) and "model_type" not in row:
                row["model_type"] = model_type

        allow_insert = (idx == 0)
        base_source_type = "observation" if disease_last_observed else "zero_init"

        storage.insert_or_update_prediction_rows_for_one_disease(
            prediction_run_id=prediction_run_id,
            site_id=site_id,
            batch_id=batch_id,
            model_type=model_type,
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