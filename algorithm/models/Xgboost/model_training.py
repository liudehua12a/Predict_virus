"""
玉米病害 XGBoost 模型训练模块（修正版）
重点：
1. 使用 GroupKFold，按“地点 + 品种”分组交叉验证
2. 训练目标统一到 0~1，与在线滚动预测口径一致
3. 自动保存项目约定的 xg_full_bundle_*.pt
4. bundle 内显式保存 feature_columns，供在线推理严格按列名对齐
5. 输出每折与总体的分组交叉验证结果
6. 生成“按地点+品种”的预测图与预测明细表（基于折外验证预测）
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class CornDiseasePredictor:
    def __init__(self, data_file: str = "data/processed_data.csv", outputs_root: str = "outputs"):
        self.data_file = data_file
        self.outputs_root = outputs_root

        self.data: pd.DataFrame | None = None
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, Any] = {}
        self.imputers: dict[str, Any] = {}
        self.feature_columns_by_model: dict[str, list[str]] = {}
        self.best_params_by_model: dict[str, dict[str, Any]] = {}
        self.cv_summary_by_model: dict[str, dict[str, Any]] = {}

        self.figures_dir = os.path.join(self.outputs_root, "figures")
        self.tables_dir = os.path.join(self.outputs_root, "tables")
        self.tables_model_eval_dir = os.path.join(self.tables_dir, "model_evaluation")
        self.figures_prediction_results_dir = os.path.join(self.figures_dir, "prediction_results")
        self.group_plot_root = os.path.join(self.figures_prediction_results_dir, "by_group")
        self.group_table_root = os.path.join(self.tables_model_eval_dir, "by_group")

        for p in [
            self.figures_dir,
            self.tables_dir,
            self.tables_model_eval_dir,
            self.figures_prediction_results_dir,
            self.group_plot_root,
            self.group_table_root,
            "models",
            os.path.join("models", "Xgboost"),
        ]:
            os.makedirs(p, exist_ok=True)

        self.diseases = {
            "灰斑病": {"rate": "灰斑病发病株率", "index": "灰斑病病情指数"},
            "大斑病": {"rate": "大斑病发病株率", "index": "大斑病病情指数"},
            "白斑病": {"rate": "白斑病发病株率", "index": "白斑病病情指数"},
        }

        self.bundle_targets = {
            "灰斑病": ["gray_incidence", "gray_index"],
            "大斑病": ["blight_incidence", "blight_index"],
            "白斑病": ["white_incidence", "white_index"],
        }

        self.bundle_file_keys = {
            "灰斑病": "gray",
            "大斑病": "blight",
            "白斑病": "white",
        }

        # 训练特征必须与在线推理同口径（a_config.BASE_MODEL_FEATURES）
        self.core_feature_candidates = [
            "gdd_cum",
            "rain_21d_sum", "rain_7d_sum", "rain_14d_sum",
            "rainy_streak_days", "rain_gap_days",

            "temp_21d_mean", "temp_7d_mean", "temp_14d_mean",
            "temp_range_24h_c",

            "rh_21d_mean", "rh_7d_mean", "rh_14d_mean",
            "humidity_range_daily",

            "soil_rel_humidity_14d_mean", "soil_rel_humidity_7d_mean", "soil_rel_humidity_21d_mean",

            "wind_7d_mean",
            "is_weak_wind_day",
            "weak_wind_streak_days",

            "radiation_7d_mean", "low_radiation_streak_days",

            "hot_streak_days", "cold_streak_days", "optimal_temp_streak_days",

            "high_humidity_streak_days", "high_humidity_7d_count",

            "heavy_rain_7d_count", "heavy_rain_streak_days", "max_single_day_rain_7d",

            "hot_humid_streak_days", "optimal_temp_humid_streak_days", "weak_wind_humid_streak_days",
        ]

    def load_data(self) -> pd.DataFrame:
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_file, encoding="utf-8-sig")
        print(f"数据加载完成，共 {len(self.data)} 条记录")
        return self.data

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for col in ["date", "调查日期", "时间", "日期"]:
            if col in out.columns:
                out["__date__"] = pd.to_datetime(out[col], errors="coerce")
                break
        else:
            raise ValueError("数据中未找到可用日期列（date / 调查日期 / 时间 / 日期）")

        if "地点" not in out.columns:
            raise ValueError("数据缺少 '地点' 列")
        out["__site__"] = out["地点"].astype(str).str.strip()

        if "品种" in out.columns:
            out["__variety__"] = out["品种"].fillna("未知品种").astype(str).str.strip()
            out["__variety__"] = out["__variety__"].replace("", "未知品种")
        else:
            out["__variety__"] = "未知品种"

        out["__group__"] = out["__site__"] + "__" + out["__variety__"]
        out = out.dropna(subset=["__date__"]).copy()
        out = out.sort_values(["__group__", "__date__"]).reset_index(drop=True)
        return out

    def _build_prev_targets(self, df: pd.DataFrame, disease_name: str) -> pd.DataFrame:
        out = df.copy()
        rate_col = self.diseases[disease_name]["rate"]
        index_col = self.diseases[disease_name]["index"]

        out["prev_rate"] = (
            out.groupby("__group__")[rate_col]
            .shift(1)
            .fillna(0)
            .astype(float) / 100.0
        )
        out["prev_index"] = (
            out.groupby("__group__")[index_col]
            .shift(1)
            .fillna(0)
            .astype(float) / 100.0
        )

        out["month"] = out["__date__"].dt.month.astype(int)
        out["day_of_year"] = out["__date__"].dt.dayofyear.astype(int)
        return out

    def prepare_features_for_disease(self, disease_name: str) -> tuple[pd.DataFrame, list[str]]:
        if self.data is None:
            raise ValueError("请先 load_data()")

        df = self._ensure_date_column(self.data)
        df = self._build_prev_targets(df, disease_name)

        available = [c for c in self.core_feature_candidates if c in df.columns]

        # 回退机制：若在线同名特征缺失过多，则自动补充其它数值列，防止退化为4列模型
        if len(available) < 10:
            exclude_cols = {
                "序号", "时间", "地点", "品种", "date", "调查日期", "日期",
                "__date__", "__site__", "__variety__", "__group__", "__target__",
                "灰斑病发病株率", "灰斑病病情指数",
                "大斑病发病株率", "大斑病病情指数",
                "白斑病发病株率", "白斑病病情指数",
                "灰斑病抗性", "大斑病抗性", "白斑病抗性",
                "生育期", "LAT(degrees_north)", "LON(degrees_east)",
                "prev_rate", "prev_index", "month", "day_of_year",
            }
            numeric_candidates = [
                c for c in df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            merged_candidates: list[str] = []
            seen = set()
            for c in (available + numeric_candidates):
                if c not in seen:
                    merged_candidates.append(c)
                    seen.add(c)
            available = merged_candidates

        feature_columns = available + ["prev_rate", "prev_index", "month", "day_of_year"]

        dedup: list[str] = []
        seen = set()
        for c in feature_columns:
            if c not in seen:
                dedup.append(c)
                seen.add(c)

        print(
            f"[TrainFeatureSelect] {disease_name}: core_available={len([c for c in self.core_feature_candidates if c in df.columns])}, "
            f"final_features={len(dedup)}"
        )
        return df, dedup

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0:
            return float("nan")
        if float(np.var(y_true)) < 1e-12:
            return float("nan")
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def _diag_stats(name: str, y: pd.Series) -> dict[str, Any]:
        s = pd.Series(y).dropna()
        if len(s) == 0:
            return {"subset": name, "n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "nonzero_ratio": np.nan}
        return {
            "subset": name,
            "n": int(len(s)),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "nonzero_ratio": float((s > 0.001).mean()),
        }

    @staticmethod
    def _build_sample_weight(y_train: pd.Series) -> np.ndarray:
        y = np.asarray(y_train, dtype=float)
        w = np.ones_like(y, dtype=float)
        w[y >= 0.10] = 2.0
        w[y >= 0.20] = 3.0
        w[y >= 0.40] = 4.0
        return w

    def _candidate_params(self) -> list[dict[str, Any]]:
        return [
            {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 600, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 5.0},
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 600, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 5.0},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 400, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 8.0},
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 8.0},
            {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.5, "reg_lambda": 3.0},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 10.0},
        ]

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        bad = '\\/:*?"<>|'
        out = "".join("_" if ch in bad else ch for ch in str(name))
        out = out.replace(" ", "_")
        return out[:180]

    def _save_group_prediction_outputs(self, disease_name: str, target_type: str, oof_df: pd.DataFrame) -> tuple[str, str]:
        model_key = f"{disease_name}_{target_type}"
        group_table_dir = os.path.join(self.group_table_root, model_key)
        group_plot_dir = os.path.join(self.group_plot_root, model_key)
        os.makedirs(group_table_dir, exist_ok=True)
        os.makedirs(group_plot_dir, exist_ok=True)

        oof_df = oof_df.copy().sort_values(["group", "date"]).reset_index(drop=True)

        merged_csv = os.path.join(group_table_dir, f"{model_key}_oof_predictions_all_groups.csv")
        oof_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

        for group_name, gdf in oof_df.groupby("group"):
            gdf = gdf.sort_values("date").copy()
            safe_name = self._sanitize_filename(group_name)

            group_csv = os.path.join(group_table_dir, f"{safe_name}.csv")
            gdf.to_csv(group_csv, index=False, encoding="utf-8-sig")

            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(gdf["date"], gdf["actual"], marker="o", linewidth=1.8, label="实际值")
            ax.plot(gdf["date"], gdf["pred"], marker="s", linewidth=1.8, label="预测值")
            ax.set_title(f"{group_name} - {disease_name}{'发病株率' if target_type == 'rate' else '病情指数'}")
            ax.set_xlabel("日期")
            ax.set_ylabel("数值(0~1)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.legend(frameon=False)
            plt.xticks(rotation=25)
            plt.tight_layout()

            plot_path = os.path.join(group_plot_dir, f"{safe_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

        return merged_csv, group_plot_dir

    def train_model(self, disease_name: str, target_type: str = "rate", random_state: int = 42, n_splits: int = 5) -> dict[str, Any]:
        print("\\n" + "=" * 70)
        print(f"开始训练 {disease_name} - {'发病株率' if target_type == 'rate' else '病情指数'}（GroupKFold: 地点+品种）")
        print("=" * 70)

        df, feature_columns = self.prepare_features_for_disease(disease_name)
        target_col = self.diseases[disease_name][target_type]

        df = df[df[target_col].notna()].copy()
        if len(df) == 0:
            raise ValueError(f"{disease_name}-{target_type} 没有有效样本")

        df["__target__"] = pd.to_numeric(df[target_col], errors="coerce").astype(float) / 100.0
        df = df.dropna(subset=["__target__"]).copy()

        groups = df["__group__"].copy()
        unique_groups = groups.nunique()
        if unique_groups < 3:
            raise ValueError(f"{disease_name}-{target_type} 的 地点+品种 组合太少，仅 {unique_groups} 组，无法做 GroupKFold")

        actual_splits = min(n_splits, unique_groups)
        gkf = GroupKFold(n_splits=actual_splits)

        X_all = df[feature_columns].copy().reset_index(drop=True)
        y_all = df["__target__"].copy().reset_index(drop=True)
        groups_all = groups.reset_index(drop=True)
        dates_all = df["__date__"].reset_index(drop=True)

        candidate_params = self._candidate_params()

        print("\\n整体目标分布诊断（0~1口径）:")
        print(pd.DataFrame([self._diag_stats("all", y_all)]).to_string(index=False))
        print(f"地点+品种 分组数: {unique_groups}")

        best_params = None
        best_cv_rmse = float("inf")
        best_cv_mae = float("inf")
        best_fold_details: list[dict[str, Any]] = []
        best_oof_frames: list[pd.DataFrame] = []

        print(f"\\n开始搜索参数，共 {len(candidate_params)} 组...")
        for idx, params in enumerate(candidate_params, start=1):
            fold_rows: list[dict[str, Any]] = []
            oof_frames_this_param: list[pd.DataFrame] = []

            for fold, (train_idx, valid_idx) in enumerate(gkf.split(X_all, y_all, groups_all), start=1):
                X_train = X_all.iloc[train_idx].copy()
                X_valid = X_all.iloc[valid_idx].copy()
                y_train = y_all.iloc[train_idx].copy()
                y_valid = y_all.iloc[valid_idx].copy()

                train_groups = groups_all.iloc[train_idx].nunique()
                valid_groups = groups_all.iloc[valid_idx].nunique()

                imputer = SimpleImputer(strategy="median")
                X_train_imp = imputer.fit_transform(X_train)
                X_valid_imp = imputer.transform(X_valid)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imp)
                X_valid_scaled = scaler.transform(X_valid_imp)

                sample_weight = self._build_sample_weight(y_train)

                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                    **params,
                )

                fit_kwargs = {
                    "X": X_train_scaled,
                    "y": y_train,
                    "sample_weight": sample_weight,
                    "eval_set": [(X_valid_scaled, y_valid)],
                    "verbose": False,
                }
                try:
                    model.fit(**fit_kwargs, early_stopping_rounds=50)
                except TypeError:
                    model.fit(**fit_kwargs)

                pred_valid = np.clip(model.predict(X_valid_scaled), 0.0, 1.0)

                rmse = self._rmse(y_valid, pred_valid)
                mae = float(mean_absolute_error(y_valid, pred_valid))
                r2v = self._safe_r2(y_valid, pred_valid)

                nz_mask = y_valid > 0.05
                rmse_nz = self._rmse(y_valid[nz_mask], pred_valid[nz_mask]) if nz_mask.any() else np.nan
                mae_nz = float(mean_absolute_error(y_valid[nz_mask], pred_valid[nz_mask])) if nz_mask.any() else np.nan

                fold_rows.append({
                    "param_index": idx,
                    "fold": fold,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2v,
                    "rmse_nonzero": rmse_nz,
                    "mae_nonzero": mae_nz,
                    "n_valid": int(len(y_valid)),
                    "group_count_train": int(train_groups),
                    "group_count_valid": int(valid_groups),
                    "valid_mean": float(y_valid.mean()),
                    "valid_std": float(y_valid.std(ddof=0)),
                    "valid_nonzero_ratio": float((y_valid > 0.001).mean()),
                    "params": str(params),
                })

                oof_frames_this_param.append(pd.DataFrame({
                    "fold": fold,
                    "date": dates_all.iloc[valid_idx].values,
                    "group": groups_all.iloc[valid_idx].values,
                    "actual": y_valid.values,
                    "pred": pred_valid,
                }))

            fold_df = pd.DataFrame(fold_rows)
            mean_rmse = float(fold_df["rmse"].mean())
            mean_mae = float(fold_df["mae"].mean())
            print(f"  参数组 {idx:02d}/{len(candidate_params)} CV-RMSE={mean_rmse:.4f}, CV-MAE={mean_mae:.4f} params={params}")

            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_cv_mae = mean_mae
                best_params = params
                best_fold_details = fold_rows
                best_oof_frames = oof_frames_this_param

        if best_params is None:
            raise RuntimeError("未找到可用参数")

        print(f"  ✓ 最佳 CV-RMSE: {best_cv_rmse:.4f}")
        print(f"  ✓ 最佳 CV-MAE : {best_cv_mae:.4f}")
        print(f"  ✓ 最佳参数     : {best_params}")

        final_imputer = SimpleImputer(strategy="median")
        X_all_imp = final_imputer.fit_transform(X_all)

        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X_all_imp)

        sample_weight_all = self._build_sample_weight(y_all)

        final_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            **best_params,
        )
        final_model.fit(X_all_scaled, y_all, sample_weight=sample_weight_all, verbose=False)
        pred_all = np.clip(final_model.predict(X_all_scaled), 0.0, 1.0)

        all_rmse = self._rmse(y_all, pred_all)
        all_mae = float(mean_absolute_error(y_all, pred_all))
        all_r2 = self._safe_r2(y_all, pred_all)

        nz_mask_all = y_all > 0.05
        all_rmse_nz = self._rmse(y_all[nz_mask_all], pred_all[nz_mask_all]) if nz_mask_all.any() else np.nan
        all_mae_nz = float(mean_absolute_error(y_all[nz_mask_all], pred_all[nz_mask_all])) if nz_mask_all.any() else np.nan

        model_key = f"{disease_name}_{target_type}"
        self.models[model_key] = final_model
        self.scalers[model_key] = final_scaler
        self.imputers[model_key] = final_imputer
        self.feature_columns_by_model[model_key] = feature_columns
        self.best_params_by_model[model_key] = best_params

        joblib.dump(final_model, f"models/model_{disease_name}_{target_type}.pkl")
        joblib.dump(final_scaler, f"models/scaler_{disease_name}_{target_type}.pkl")
        joblib.dump(final_imputer, f"models/imputer_{disease_name}_{target_type}.pkl")

        fold_df = pd.DataFrame(best_fold_details)
        fold_csv = os.path.join(self.tables_model_eval_dir, f"groupkfold_{model_key}.csv")
        fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")

        oof_df = pd.concat(best_oof_frames, ignore_index=True).sort_values(["group", "date"]).reset_index(drop=True)
        group_csv, group_plot_dir = self._save_group_prediction_outputs(disease_name, target_type, oof_df)

        dist_df = pd.DataFrame([self._diag_stats("all", y_all)])
        dist_df.to_csv(
            os.path.join(self.tables_model_eval_dir, f"distribution_{model_key}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        summary = {
            "model": final_model,
            "imputer": final_imputer,
            "scaler": final_scaler,
            "feature_columns": feature_columns,
            "best_params": best_params,
            "cv_rmse": best_cv_rmse,
            "cv_mae": best_cv_mae,
            "cv_rmse_nonzero": float(fold_df["rmse_nonzero"].dropna().mean()) if fold_df["rmse_nonzero"].notna().any() else np.nan,
            "cv_mae_nonzero": float(fold_df["mae_nonzero"].dropna().mean()) if fold_df["mae_nonzero"].notna().any() else np.nan,
            "all_rmse": all_rmse,
            "all_mae": all_mae,
            "all_r2": all_r2,
            "all_rmse_nonzero": all_rmse_nz,
            "all_mae_nonzero": all_mae_nz,
            "fold_details_path": fold_csv,
            "group_oof_csv": group_csv,
            "group_plot_dir": group_plot_dir,
            "y_all": y_all,
            "y_all_pred": pred_all,
        }
        self.cv_summary_by_model[model_key] = summary

        print("\\n交叉验证结果（按地点+品种分组）:")
        print(f"CV - RMSE: {summary['cv_rmse']:.4f}, MAE: {summary['cv_mae']:.4f}")
        print(f"CV(非零样本) - RMSE: {summary['cv_rmse_nonzero']}, MAE: {summary['cv_mae_nonzero']}")
        print(f"按地点+品种明细表: {summary['group_oof_csv']}")
        print(f"按地点+品种预测图目录: {summary['group_plot_dir']}")
        print("\\n全量重训练拟合结果（仅用于检查是否过拟合，不代表泛化）:")
        print(f"ALL - RMSE: {summary['all_rmse']:.4f}, MAE: {summary['all_mae']:.4f}, R²: {summary['all_r2']}")

        self.plot_prediction_results(disease_name, target_type, summary)
        return summary

    def plot_prediction_results(self, disease_name: str, target_type: str, results: dict[str, Any]) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        y_true = np.asarray(results["y_all"], dtype=float)
        y_pred = np.asarray(results["y_all_pred"], dtype=float)

        ax.scatter(y_true, y_pred, alpha=0.6)
        low = min(float(np.min(y_true)), float(np.min(y_pred)))
        high = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([low, high], [low, high], "r--", lw=2)
        ax.set_xlabel("实际值(0~1)")
        ax.set_ylabel("预测值(0~1)")
        ax.set_title(
            f"{disease_name} - {'发病株率' if target_type == 'rate' else '病情指数'}\\n"
            f"CV-RMSE={results['cv_rmse']:.4f}, ALL-RMSE={results['all_rmse']:.4f}"
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_file = os.path.join(self.figures_prediction_results_dir, f"prediction_results_{disease_name}_{target_type}.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

    def save_evaluation_summary(self) -> None:
        rows = []
        for model_key, res in self.cv_summary_by_model.items():
            rows.append({
                "model_key": model_key,
                "cv_rmse": res["cv_rmse"],
                "cv_mae": res["cv_mae"],
                "cv_rmse_nonzero": res["cv_rmse_nonzero"],
                "cv_mae_nonzero": res["cv_mae_nonzero"],
                "all_rmse": res["all_rmse"],
                "all_mae": res["all_mae"],
                "all_r2": res["all_r2"],
                "all_rmse_nonzero": res["all_rmse_nonzero"],
                "all_mae_nonzero": res["all_mae_nonzero"],
                "best_params": str(res["best_params"]),
                "fold_details_path": res["fold_details_path"],
                "group_oof_csv": res["group_oof_csv"],
                "group_plot_dir": res["group_plot_dir"],
            })

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(self.tables_model_eval_dir, "model_evaluation_summary.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\\n✓ 模型评估汇总已保存: {out_path}")

    def save_xgboost_full_bundle(self, disease_name: str) -> Path:
        rate_key = f"{disease_name}_rate"
        index_key = f"{disease_name}_index"

        if rate_key not in self.models or index_key not in self.models:
            raise ValueError(f"{disease_name} 的 rate/index 子模型不完整，无法保存 bundle")

        bundle = {
            "models": {
                "rate": self.models[rate_key],
                "index": self.models[index_key],
            },
            "imputers": {
                "rate": self.imputers[rate_key],
                "index": self.imputers[index_key],
            },
            "scalers": {
                "rate": self.scalers[rate_key],
                "index": self.scalers[index_key],
            },
            "targets": self.bundle_targets[disease_name],
            "feature_columns": {
                "rate": self.feature_columns_by_model[rate_key],
                "index": self.feature_columns_by_model[index_key],
            },
        }

        bundle_path = Path("models") / "Xgboost" / f"xg_full_bundle_{self.bundle_file_keys[disease_name]}.pt"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, bundle_path)
        print(f"✓ 已保存 XGBoost full bundle: {bundle_path}")
        print(f"  └─ rate feature_columns 数量: {len(bundle['feature_columns']['rate'])}")
        print(f"  └─ index feature_columns 数量: {len(bundle['feature_columns']['index'])}")
        return bundle_path

    def save_all_xgboost_full_bundles(self) -> None:
        for disease_name in self.diseases.keys():
            self.save_xgboost_full_bundle(disease_name)

    def train_all_models(self, random_state: int = 42, n_splits: int = 5) -> dict[str, dict[str, Any]]:
        if self.data is None:
            raise ValueError("请先 load_data()")

        print("\\n" + "=" * 70)
        print("开始训练所有 XGBoost 模型（修正版：GroupKFold 地点+品种 + 按组预测图 + 保存feature_columns）")
        print("=" * 70)

        all_results: dict[str, dict[str, Any]] = {}

        for disease_name in self.diseases.keys():
            for target_type in ["rate", "index"]:
                model_key = f"{disease_name}_{target_type}"
                try:
                    all_results[model_key] = self.train_model(
                        disease_name,
                        target_type=target_type,
                        random_state=random_state,
                        n_splits=n_splits,
                    )
                except Exception as exc:
                    print(f"\\n✗ {model_key} 训练失败: {exc}")

        self.save_evaluation_summary()
        self.save_all_xgboost_full_bundles()

        print("\\n" + "=" * 70)
        print("所有模型训练完成")
        print("=" * 70)
        return all_results


if __name__ == "__main__":
    predictor = CornDiseasePredictor("data/processed_data.csv", outputs_root="outputs")
    predictor.load_data()
    predictor.train_all_models()
