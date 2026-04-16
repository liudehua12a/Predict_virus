from __future__ import annotations

import random
from copy import deepcopy
from typing import Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
import a_config as cfg
import c_feature_engineering as fe
import warnings


class DiseaseLSTM(nn.Module):
    """双分支网络：LSTM 处理时序，MLP 处理过程特征，最终联合输出双目标增量。"""

    def __init__(self, seq_dim: int, tab_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=20, batch_first=True)
        self.tab_net = nn.Sequential(nn.Linear(tab_dim, 24), nn.ReLU(), nn.Dropout(0.05))
        self.head = nn.Sequential(nn.Linear(44, 24), nn.ReLU(), nn.Linear(24, 2))

    def forward(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self.lstm(seq_input)
        seq_hidden = seq_output[:, -1, :]
        tab_hidden = self.tab_net(tab_input)
        return self.head(torch.cat([seq_hidden, tab_hidden], dim=1))


class DiseaseLSTMFusion(nn.Module):
    """
    融合结构：
    先提取倒数第二层特征，再由线性头输出；
    在线推理时将 extract_penultimate 的输出与 tab_scaled 拼接，送入 xgb_models 做最终融合。
    """

    def __init__(self, seq_dim: int, tab_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=20, batch_first=True)
        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.fusion = nn.Sequential(
            nn.Linear(44, 24),
            nn.ReLU(),
        )
        self.head = nn.Linear(24, 2)

    def extract_penultimate(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self.lstm(seq_input)
        seq_hidden = seq_output[:, -1, :]
        tab_hidden = self.tab_net(tab_input)
        return self.fusion(torch.cat([seq_hidden, tab_hidden], dim=1))

    def forward(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        fusion_hidden = self.extract_penultimate(seq_input, tab_input)
        return self.head(fusion_hidden)


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weighted = loss * mask
    denominator = mask.sum().clamp(min=1.0)
    return weighted.sum() / denominator


def split_train_validation_sites(site_ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    shuffled = site_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.2)))
    validation = sorted(shuffled[:val_count])
    training = sorted(shuffled[val_count:])
    if not training:
        training, validation = validation, training
    return training, validation


def train_model(
    site_rows: dict[int, list[dict[str, Any]]],
    train_site_ids: list[int],
    validation_site_ids: list[int],
    targets: list[str],
    seed: int,
) -> dict[str, Any]:
    train_seq, train_tab, train_y, train_mask, _ = fe.build_training_arrays(site_rows, train_site_ids, targets)
    val_seq, val_tab, val_y, val_mask, val_prev = fe.build_training_arrays(site_rows, validation_site_ids, targets)

    scalers = fe.fit_scalers(train_seq, train_tab, train_y)
    train_seq_scaled, train_tab_scaled = fe.apply_scalers(train_seq, train_tab, scalers)
    val_seq_scaled, val_tab_scaled = fe.apply_scalers(val_seq, val_tab, scalers)
    train_y_scaled = fe.scale_targets(train_y, scalers)
    val_y_scaled = fe.scale_targets(val_y, scalers)

    model = DiseaseLSTM(seq_dim=train_seq_scaled.shape[-1], tab_dim=train_tab_scaled.shape[-1]).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_seq_tensor = torch.tensor(train_seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_tab_tensor = torch.tensor(train_tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32, device=cfg.DEVICE)

    val_seq_tensor = torch.tensor(val_seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_tab_tensor = torch.tensor(val_tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_y_tensor = torch.tensor(val_y_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.float32, device=cfg.DEVICE)

    rng = np.random.default_rng(seed)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience = 20
    patience_left = patience
    batch_size = len(train_seq_scaled)

    for _ in range(120):
        permutation = rng.permutation(len(train_seq_scaled))
        model.train()
        for start in range(0, len(permutation), batch_size):
            batch_ids = permutation[start : start + batch_size]
            pred = model(train_seq_tensor[batch_ids], train_tab_tensor[batch_ids])
            loss = masked_smooth_l1(pred, train_y_tensor[batch_ids], train_mask_tensor[batch_ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq_tensor, val_tab_tensor)
            val_loss = float(masked_smooth_l1(val_pred, val_y_tensor, val_mask_tensor).cpu().item())

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_delta_scaled = model(val_seq_tensor, val_tab_tensor).cpu().numpy()
    val_delta = fe.unscale_targets(val_delta_scaled, scalers)
    val_actual = val_prev + val_y

    output_alphas = []
    monotonic_flags = []
    for output_index, target_name in enumerate(targets):
        is_monotonic = target_name in cfg.MONOTONIC_TARGETS
        monotonic_flags.append(is_monotonic)
        valid_mask = val_mask[:, output_index] > 0.5
        if not np.any(valid_mask):
            output_alphas.append(0.0)
            continue
        best_alpha = 0.0
        best_rmse = float("inf")
        for alpha in cfg.ALPHA_GRID:
            pred_values = np.clip(val_prev[valid_mask, output_index] + alpha * val_delta[valid_mask, output_index], 0.0, 1.0)
            if is_monotonic:
                pred_values = np.maximum(pred_values, val_prev[valid_mask, output_index])
            actual_values = np.clip(val_actual[valid_mask, output_index], 0.0, 1.0)
            rmse = float(np.sqrt(np.mean((pred_values - actual_values) ** 2)))
            if rmse < best_rmse - 1e-8:
                best_rmse = rmse
                best_alpha = float(alpha)
        output_alphas.append(best_alpha)

    return {
        "model": model,
        "scalers": scalers,
        "targets": targets,
        "best_val_loss": best_val_loss,
        "output_alphas": np.asarray(output_alphas, dtype=np.float32),
        "monotonic_flags": monotonic_flags,
        "xgb_models": [],
    }


def train_full_model(site_rows: dict[int, list[dict[str, Any]]], targets: list[str], seed: int) -> dict[str, Any]:
    all_sites = sorted(site_rows.keys())
    train_sites, val_sites = split_train_validation_sites(all_sites, seed)
    return train_model(site_rows, train_sites, val_sites, targets, seed)


def save_bundle(bundle: dict[str, Any], save_path: str | Path) -> None:
    save_path = Path(save_path)
    model = bundle["model"]

    payload = {
        "seq_dim": int(model.lstm.input_size),
        "tab_dim": int(model.tab_net[0].in_features),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "xgb_models": bundle.get("xgb_models", []),
        "scalers": bundle["scalers"],
        "targets": list(bundle["targets"]),
        "best_val_loss": float(bundle["best_val_loss"]),
        "output_alphas": np.asarray(bundle["output_alphas"], dtype=np.float32),
        "monotonic_flags": list(bundle["monotonic_flags"]),
        "seq_features": list(cfg.SEQ_FEATURES),
        "base_model_features": list(cfg.BASE_MODEL_FEATURES),
        "lookback_days": int(cfg.LOOKBACK_DAYS),
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def load_bundle(save_path: str | Path) -> dict[str, Any]:
    save_path = Path(save_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*If you are loading a serialized model.*")
        payload = torch.load(save_path, map_location=cfg.DEVICE, weights_only=False)
    state_dict = payload["model_state_dict"]

    is_fusion_style = (
        "fusion.0.weight" in state_dict
        and "fusion.0.bias" in state_dict
        and "head.weight" in state_dict
        and "head.bias" in state_dict
    )

    is_lstm_style = (
        "head.0.weight" in state_dict
        and "head.0.bias" in state_dict
        and "head.2.weight" in state_dict
        and "head.2.bias" in state_dict
    )

    if is_fusion_style:
        model = DiseaseLSTMFusion(
            seq_dim=int(payload["seq_dim"]),
            tab_dim=int(payload["tab_dim"]),
        ).to(cfg.DEVICE)
    elif is_lstm_style:
        model = DiseaseLSTM(
            seq_dim=int(payload["seq_dim"]),
            tab_dim=int(payload["tab_dim"]),
        ).to(cfg.DEVICE)
    else:
        raise ValueError(
            f"无法识别模型结构，缺少关键权重字段。示例 keys: {list(state_dict.keys())[:10]}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "scalers": payload["scalers"],
        "targets": payload["targets"],
        "best_val_loss": payload["best_val_loss"],
        "output_alphas": np.asarray(payload["output_alphas"], dtype=np.float32),
        "monotonic_flags": payload["monotonic_flags"],
        "xgb_models": payload.get("xgb_models", []),
    }


def predict_row(bundle: dict[str, Any], row: dict[str, Any], previous_targets: np.ndarray) -> np.ndarray:
    #  XGBoost bundle
    if isinstance(bundle, dict) and "models" in bundle and "imputers" in bundle and "scalers" in bundle:
        return _predict_row_xgboost_bundle(bundle, row, previous_targets)

    tab_values = [fe.fill_none(row[name]) for name in cfg.BASE_MODEL_FEATURES] + previous_targets.tolist()
    seq = row["weather_seq_21"].astype(np.float32)
    tab = np.asarray(tab_values, dtype=np.float32)

    seq_scaled, tab_scaled = fe.apply_scalers(seq[None, ...], tab[None, ...], bundle["scalers"])
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32, device=cfg.DEVICE)

    with torch.no_grad():
        prediction = bundle["model"](seq_tensor, tab_tensor)
        if hasattr(bundle["model"], "extract_penultimate"):
            penultimate = bundle["model"].extract_penultimate(seq_tensor, tab_tensor).cpu().numpy()
        else:
            penultimate = None

    delta_values = fe.unscale_targets(prediction.cpu().numpy(), bundle["scalers"])[0]
    lstm_values = previous_targets + bundle["output_alphas"] * delta_values
    lstm_values = np.asarray(lstm_values, dtype=np.float32)

    xgb_models = bundle.get("xgb_models", [])
    use_xgb_fusion = (
        penultimate is not None
        and isinstance(xgb_models, (list, tuple))
        and len(xgb_models) == len(bundle["targets"])
        and any(model is not None for model in xgb_models)
    )

    if use_xgb_fusion:
        xgb_features = np.concatenate([penultimate[0], tab_scaled[0]], axis=0).astype(np.float32)[None, :]
        xgb_values = []
        for output_index, model in enumerate(xgb_models):
            if model is None:
                xgb_values.append(float(lstm_values[output_index]))
            else:
                pred = model.predict(xgb_features)
                xgb_values.append(float(np.asarray(pred, dtype=np.float32).reshape(-1)[0]))
        fusion_alpha = 0.2
        values = (1.0 - fusion_alpha) * lstm_values + fusion_alpha * np.asarray(xgb_values, dtype=np.float32)
    else:
        values = lstm_values.astype(np.float32)

    for output_index, is_monotonic in enumerate(bundle["monotonic_flags"]):
        if is_monotonic:
            values[output_index] = max(values[output_index], previous_targets[output_index])
    return np.clip(values, 0.0, 1.0)


def _extract_seq_last(row: dict[str, Any]) -> np.ndarray:
    seq_2d = np.asarray(row.get("weather_seq_21"), dtype=np.float32)
    if seq_2d.ndim != 2 or seq_2d.shape[1] != len(cfg.SEQ_FEATURES):
        return np.zeros(len(cfg.SEQ_FEATURES), dtype=np.float32)
    return seq_2d[-1].astype(np.float32)


def _extract_day_of_year(row: dict[str, Any]) -> float:
    date_value = row.get("date")
    if hasattr(date_value, "timetuple"):
        return float(date_value.timetuple().tm_yday)
    return float(datetime.strptime(str(date_value)[:10], "%Y-%m-%d").timetuple().tm_yday)


def _infer_disease_key_from_bundle(bundle: dict[str, Any]) -> str | None:
    targets = bundle.get("targets") or []
    if not targets:
        return None
    first = str(targets[0]).lower()
    if first.startswith("gray"):
        return "gray"
    if first.startswith("blight"):
        return "blight"
    if first.startswith("white"):
        return "white"
    return None


def _is_prev_feature_for_target(col_name: str, disease_key: str | None, target_type: str) -> bool:
    text = str(col_name).strip().lower()

    lag_markers = ["prev_", "previous_", "lag1", "lag_1", "t-1", "前一日", "前1日", "上一日", "上一天"]
    if not any(marker in text for marker in lag_markers):
        return False

    if target_type == "rate":
        target_tokens = ["发病株率", "发病率", "incidence", "rate"]
    else:
        target_tokens = ["病情指数", "index"]
    if not any(token in text for token in target_tokens):
        return False

    disease_aliases = {
        "gray": ["gray", "灰斑"],
        "blight": ["blight", "大斑"],
        "white": ["white", "白斑"],
    }
    all_aliases = [alias for aliases in disease_aliases.values() for alias in aliases]
    current_aliases = disease_aliases.get(disease_key or "", [])

    has_current_disease = any(alias in text for alias in current_aliases)
    has_other_disease = any(alias in text for alias in all_aliases if alias not in current_aliases)
    return has_current_disease or (not has_other_disease)


def _resolve_xgboost_feature_columns(bundle: dict[str, Any], model_key: str, imputer: Any) -> list[str]:
    feature_columns = bundle.get("feature_columns")

    if isinstance(feature_columns, dict):
        cols = feature_columns.get(model_key)
        if cols:
            return list(cols)

    if isinstance(feature_columns, list) and feature_columns:
        return list(feature_columns)

    if hasattr(imputer, "feature_names_in_"):
        cols = list(imputer.feature_names_in_)
        if cols:
            return cols

    raise ValueError(
        f"XGBoost bundle 缺少 {model_key} 的 feature_columns，且 imputer 也没有 feature_names_in_，无法严格对齐输入特征"
    )


def _build_feature_row_by_columns(
    row: dict[str, Any],
    previous_targets: np.ndarray,
    feature_columns: list[str],
    disease_key: str | None,
) -> pd.DataFrame:
    seq_last = _extract_seq_last(row)
    seq_last_map = {name: float(seq_last[idx]) for idx, name in enumerate(cfg.SEQ_FEATURES)}
    day_of_year = _extract_day_of_year(row)

    feature_values: dict[str, float] = {}
    for col in feature_columns:
        if col in row:
            feature_values[col] = float(fe.fill_none(row.get(col)))
            continue

        if col in seq_last_map:
            feature_values[col] = float(fe.fill_none(seq_last_map[col]))
            continue

        text = str(col).strip().lower()
        if text in {"day_of_year", "dayofyear", "doy"}:
            feature_values[col] = float(day_of_year)
            continue

        if _is_prev_feature_for_target(col, disease_key, "rate"):
            feature_values[col] = float(previous_targets[0])
            continue

        if _is_prev_feature_for_target(col, disease_key, "index"):
            feature_values[col] = float(previous_targets[1])
            continue

        feature_values[col] = 0.0

    return pd.DataFrame([[feature_values[col] for col in feature_columns]], columns=feature_columns)


def _predict_row_xgboost_bundle(
    bundle: dict[str, Any],
    row: dict[str, Any],
    previous_targets: np.ndarray,
) -> np.ndarray:
    models = bundle.get("models") or {}
    imputers = bundle.get("imputers") or {}
    scalers = bundle.get("scalers") or {}

    model_rate = (
        models.get("rate")
        or models.get("incidence")
        or models.get("target_1")
        or next(iter(models.values()), None)
    )
    model_index = (
        models.get("index")
        or models.get("target_2")
        or (list(models.values())[1] if len(models) > 1 else None)
    )
    if model_rate is None or model_index is None:
        raise ValueError("XGBoost bundle 缺少 rate/index 子模型")

    disease_key = _infer_disease_key_from_bundle(bundle)

    def _predict_one(model_key: str, model_obj: Any) -> float:
        imputer = imputers.get(model_key)
        scaler = scalers.get(model_key)
        feature_columns = _resolve_xgboost_feature_columns(bundle, model_key, imputer)

        x_df = _build_feature_row_by_columns(
            row=row,
            previous_targets=previous_targets,
            feature_columns=feature_columns,
            disease_key=disease_key,
        )

        x_cur: Any = x_df
        if imputer is not None and hasattr(imputer, "transform"):
            x_cur = imputer.transform(x_cur)
        if scaler is not None and hasattr(scaler, "transform"):
            x_cur = scaler.transform(x_cur)

        pred = model_obj.predict(x_cur)
        return float(np.asarray(pred, dtype=np.float32).reshape(-1)[0])

    rate_key = "rate" if "rate" in models else ("incidence" if "incidence" in models else "target_1")
    index_key = "index" if "index" in models else "target_2"

    pred_rate = _predict_one(rate_key, model_rate)
    pred_index = _predict_one(index_key, model_index)
    values = np.asarray([pred_rate, pred_index], dtype=np.float32)

    return np.clip(values, 0.0, 1.0)


def add_risk_fields(row_dict: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    incidence_target = targets[0]
    index_target = targets[1]

    actual_incidence = row_dict.get(f"actual_{incidence_target}")
    pred_incidence = row_dict.get(f"pred_{incidence_target}")
    actual_index = row_dict.get(f"actual_{index_target}")
    pred_index = row_dict.get(f"pred_{index_target}")

    row_dict[f"actual_{incidence_target}_risk"] = cfg.classify_risk(actual_incidence)
    row_dict[f"pred_{incidence_target}_risk"] = cfg.classify_risk(pred_incidence)
    row_dict[f"actual_{index_target}_risk"] = cfg.classify_risk(actual_index)
    row_dict[f"pred_{index_target}_risk"] = cfg.classify_risk(pred_index)
    row_dict["actual_overall_risk"] = cfg.combine_risk(actual_incidence, actual_index)
    row_dict["pred_overall_risk"] = cfg.combine_risk(pred_incidence, pred_index)
    return row_dict


def rolling_predictions(
    site_rows: dict[int, list[dict[str, Any]]],
    bundle: dict[str, Any],
    targets: list[str],
    mode_name: str,
    use_actual_previous: bool = True,
) -> list[dict[str, Any]]:
    prediction_rows = []

    for site_id in sorted(site_rows):
        rows = site_rows[site_id]
        sub_sequences = fe.split_rows_by_replicate(rows)

        for subset in sub_sequences:
            if not subset:
                continue

            replicate_id = subset[0].get("replicate_id_same_day", 1)

            previous_targets = np.array(
                [fe.fill_none(subset[0][targets[0]]) / 100.0, fe.fill_none(subset[0][targets[1]]) / 100.0],
                dtype=np.float32,
            )

            baseline_row = {
                "mode": mode_name,
                "site_id": site_id,
                "site_alias": cfg.SITE_ALIAS[site_id],
                "site_name": subset[0]["site_name"],
                "date_str": subset[0]["date_str"],
                "record_id": subset[0]["record_id"],
                "replicate_id_same_day": replicate_id,
                f"actual_{targets[0]}": subset[0][targets[0]],
                f"actual_{targets[1]}": subset[0][targets[1]],
                f"pred_{targets[0]}": subset[0][targets[0]],
                f"pred_{targets[1]}": subset[0][targets[1]],
                "is_baseline": 1,
            }
            prediction_rows.append(add_risk_fields(baseline_row, targets))

            for row_index in range(1, len(subset)):
                row = subset[row_index]

                if use_actual_previous:
                    prev_row = subset[row_index - 1]
                    previous_targets = np.array(
                        [fe.fill_none(prev_row[targets[0]]) / 100.0, fe.fill_none(prev_row[targets[1]]) / 100.0],
                        dtype=np.float32,
                    )

                current_pred = predict_row(bundle, row, previous_targets)

                pred_row = {
                    "mode": mode_name,
                    "site_id": site_id,
                    "site_alias": cfg.SITE_ALIAS[site_id],
                    "site_name": row["site_name"],
                    "date_str": row["date_str"],
                    "record_id": row["record_id"],
                    "replicate_id_same_day": replicate_id,
                    f"actual_{targets[0]}": row[targets[0]],
                    f"actual_{targets[1]}": row[targets[1]],
                    f"pred_{targets[0]}": float(current_pred[0] * 100.0),
                    f"pred_{targets[1]}": float(current_pred[1] * 100.0),
                    "is_baseline": 0,
                }
                prediction_rows.append(add_risk_fields(pred_row, targets))

                if not use_actual_previous:
                    previous_targets = current_pred.astype(np.float32)

    return prediction_rows


def merge_prediction_tables(prediction_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for rows in prediction_sets:
        for row in rows:
            key = (
                row["mode"],
                row["site_id"],
                row["date_str"],
                row.get("replicate_id_same_day", 1),
                row.get("record_id", 0),
            )
            if key not in merged:
                merged[key] = {
                    "mode": row["mode"],
                    "site_id": row["site_id"],
                    "site_alias": row["site_alias"],
                    "site_name": row["site_name"],
                    "date_str": row["date_str"],
                    "record_id": row.get("record_id", 0),
                    "replicate_id_same_day": row.get("replicate_id_same_day", 1),
                    "is_baseline": row["is_baseline"],
                }
            merged[key].update({k: v for k, v in row.items() if k.startswith("actual_") or k.startswith("pred_")})
    return sorted(
        merged.values(),
        key=lambda item: (
            item["mode"],
            item["site_id"],
            item["date_str"],
            item.get("replicate_id_same_day", 1),
            item.get("record_id", 0),
        ),
    )


def compute_metrics(prediction_rows: list[dict[str, Any]], mode_name: str) -> list[dict[str, Any]]:
    metrics = []
    target_names = list(cfg.TARGET_LABELS.keys())
    filtered = [row for row in prediction_rows if row["mode"] == mode_name and row["is_baseline"] == 0]
    for target in target_names:
        actual_values = []
        predicted_values = []
        for row in filtered:
            actual = row.get(f"actual_{target}")
            pred = row.get(f"pred_{target}")
            if actual is None or pred is None:
                continue
            actual_values.append(float(actual))
            predicted_values.append(float(pred))
        actual_arr = np.asarray(actual_values, dtype=np.float64)
        pred_arr = np.asarray(predicted_values, dtype=np.float64)
        if len(actual_arr) == 0:
            continue
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
        denominator = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
        r2 = float(1.0 - np.sum((pred_arr - actual_arr) ** 2) / denominator) if denominator > 1e-9 else float("nan")
        metrics.append(
            {
                "mode": mode_name,
                "target": target,
                "target_cn": cfg.TARGET_LABELS[target],
                "n": int(len(actual_arr)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return metrics
