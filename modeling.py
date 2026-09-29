from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from youtube_analytics.config import LOGGER, MODEL_PATH
from youtube_analytics.data import MODEL_NUMERIC_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    category = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3))])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, MODEL_NUMERIC_FEATURES),
            ("category", category, ["category_name"]),
            ("title", TfidfVectorizer(stop_words="english", max_features=2500, min_df=3, ngram_range=(1, 2)), "title_text"),
        ],
        remainder="drop",
        sparse_threshold=0.35,
    )


def _classifier_metrics(y_true: pd.Series, prediction: np.ndarray, probabilities: np.ndarray, name: str) -> dict[str, Any]:
    return {
        "task": "Trend classification", "model": name,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else 0.0,
        "mae": np.nan, "rmse": np.nan, "r2": np.nan,
    }


def train_models(frame: pd.DataFrame, output_path: Path = MODEL_PATH, include_xgboost: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train models with a chronological holdout; only publish-time features are predictors."""
    start = time.perf_counter()
    ordered = frame.sort_values("published_at").reset_index(drop=True)
    split_index = max(int(len(ordered) * 0.8), 1)
    train, test = ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()
    if train["trending"].nunique() < 2 or test["trending"].nunique() < 2:
        raise ValueError("A time-aware split needs both trending and non-trending examples in train and test periods.")

    for part in (train, test):
        part["title_text"] = part["title"].fillna("")
    preprocessor = _build_preprocessor()
    x_train = preprocessor.fit_transform(train)
    x_test = preprocessor.transform(test)
    y_train = train["trending"].astype(int)
    y_test = test["trending"].astype(int)
    classifiers: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=700, class_weight="balanced", solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            classifiers["XGBoost"] = XGBClassifier(
                n_estimators=220, max_depth=5, learning_rate=0.06, subsample=0.85,
                colsample_bytree=0.85, eval_metric="logloss", tree_method="hist",
                n_jobs=-1, random_state=42,
            )
        except ImportError:
            LOGGER.warning("XGBoost is unavailable; continuing with sklearn classifiers")

    fitted_classifiers: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    for name, model in classifiers.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]
        metrics.append(_classifier_metrics(y_test, prediction, probability, name))
        fitted_classifiers[name] = model

    regressor_templates: dict[str, Any] = {
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=42),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
            regressor_templates["XGBoost"] = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.85,
                colsample_bytree=0.85, objective="reg:squarederror", tree_method="hist",
                n_jobs=-1, random_state=42,
            )
        except ImportError:
            pass

    regressors: dict[str, dict[str, Any]] = {"View prediction": {}, "Engagement prediction": {}, "Trend duration": {}}
    duration_train = train.loc[train["trending"].eq(1) & train["trend_duration_days"].gt(0)]
    duration_test = test.loc[test["trending"].eq(1) & test["trend_duration_days"].gt(0)]
    for task, target, inverse in [("View prediction", "views", np.expm1), ("Engagement prediction", "engagement_rate", None)]:
        target_train = train[target].astype(float)
        target_test = test[target].astype(float)
        transform = np.log1p if task == "View prediction" else None
        for name, template in regressor_templates.items():
            model = template.__class__(**template.get_params())
            model.fit(x_train, transform(target_train) if transform else target_train)
            estimate = model.predict(x_test)
            actual = target_test.to_numpy()
            if inverse:
                estimate = inverse(estimate)
            rmse = float(np.sqrt(mean_squared_error(actual, estimate)))
            metrics.append({
                "task": task, "model": name, "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan, "roc_auc": np.nan,
                "mae": float(mean_absolute_error(actual, estimate)), "rmse": rmse,
                "r2": float(r2_score(actual, estimate)),
            })
            regressors[task][name] = model

    duration_models: dict[str, Any] = {}
    if len(duration_train) >= 20 and len(duration_test) >= 5:
        train_positions = train.index.get_indexer(duration_train.index)
        test_positions = test.index.get_indexer(duration_test.index)
        x_duration_train = x_train[train_positions]
        x_duration_test = x_test[test_positions]
        duration_y_train = duration_train["trend_duration_days"].astype(float)
        duration_y_test = duration_test["trend_duration_days"].astype(float)
        for name, template in regressor_templates.items():
            model = template.__class__(**template.get_params())
            model.fit(x_duration_train, duration_y_train)
            estimate = np.maximum(model.predict(x_duration_test), 0)
            metrics.append({
                "task": "Trend duration", "model": name, "accuracy": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan,
                "mae": float(mean_absolute_error(duration_y_test, estimate)),
                "rmse": float(np.sqrt(mean_squared_error(duration_y_test, estimate))),
                "r2": float(r2_score(duration_y_test, estimate)),
            })
            regressors["Trend duration"][name] = model

    metric_frame = pd.DataFrame(metrics)
    model_bundle = {
        "version": 1,
        "preprocessor": preprocessor,
        "classifiers": fitted_classifiers,
        "regressors": regressors,
        "duration_models": regressors["Trend duration"],
        "feature_names": list(preprocessor.get_feature_names_out()),
        "metrics": metric_frame,
        "cutoff": str(test["published_at"].min()),
        "train_rows": len(train),
        "test_rows": len(test),
        "trained_seconds": time.perf_counter() - start,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_path, compress=3)
    LOGGER.info("Trained model bundle in %.1fs; time split cutoff %s", model_bundle["trained_seconds"], model_bundle["cutoff"])
    return model_bundle, metric_frame


def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except (OSError, ValueError, EOFError) as error:
        LOGGER.exception("Could not load model bundle")
        raise RuntimeError(f"Could not load saved model bundle: {error}") from error


def predict_video(bundle: dict[str, Any], row: pd.DataFrame) -> dict[str, float]:
    prepared = row.copy()
    if "title_text" not in prepared:
        prepared["title_text"] = prepared["title"].fillna("")
    matrix = bundle["preprocessor"].transform(prepared)
    classifier = bundle["classifiers"].get("XGBoost") or bundle["classifiers"].get("Random Forest") or next(iter(bundle["classifiers"].values()))
    probability = float(classifier.predict_proba(matrix)[0, 1])
    view_models = bundle["regressors"]["View prediction"]
    view_model = view_models.get("XGBoost") or next(iter(view_models.values()))
    expected_views = max(float(np.expm1(view_model.predict(matrix)[0])), 0)
    engagement_models = bundle["regressors"]["Engagement prediction"]
    engagement_model = engagement_models.get("XGBoost") or next(iter(engagement_models.values()))
    engagement = max(float(engagement_model.predict(matrix)[0]), 0)
    duration_model = next(iter(bundle["duration_models"].values()), None)
    duration = max(float(duration_model.predict(matrix)[0]), 0) if duration_model is not None else 0.0
    return {"trend_probability": probability, "expected_views": expected_views, "engagement_rate": engagement, "trend_duration_days": duration}


def explain_model(bundle: dict[str, Any], sample: pd.DataFrame, maximum_rows: int = 80) -> pd.DataFrame:
    """Compute SHAP values for a bounded sample; fall back to tree impurity importance."""
    model = bundle["classifiers"].get("XGBoost") or bundle["classifiers"].get("Random Forest")
    matrix = bundle["preprocessor"].transform(sample.head(maximum_rows))
    names = np.asarray(bundle["feature_names"], dtype=object)
    try:
        import shap
        if matrix.shape[1] <= 8000:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(matrix)
            if isinstance(values, list):
                values = values[-1]
            importance = np.abs(np.asarray(values)).mean(axis=0)
            return pd.DataFrame({"feature": names[:len(importance)], "importance": importance}).nlargest(30, "importance")
    except (ImportError, ValueError, TypeError, AttributeError, RuntimeError) as error:
        LOGGER.warning("SHAP explanation unavailable; using model importance: %s", error)
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": names[:len(importance)], "importance": importance}).nlargest(30, "importance")
