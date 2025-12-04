import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.features import split_data, preprocess_text


DATA_PATH = "data/vozhsd_filtered_full.csv"
XGB_MODEL_PATH = os.path.join("models", "xgb_model.pkl")
CB_MODEL_PATH = os.path.join("models", "catboost_model.pkl")
TFIDF_PATH = os.path.join("models", "tfidf_vectorizer.pkl")
ENSEMBLE_CONFIG_PATH = os.path.join("models", "ensemble_config.json")


def load_models_and_vectorizer():
    """Load trained XGBoost, CatBoost and TF-IDF vectorizer."""
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"Missing XGBoost model at {XGB_MODEL_PATH}")
    if not os.path.exists(CB_MODEL_PATH):
        raise FileNotFoundError(f"Missing CatBoost model at {CB_MODEL_PATH}")
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(f"Missing TF-IDF vectorizer at {TFIDF_PATH}")

    xgb_clf = joblib.load(XGB_MODEL_PATH)
    cb_clf = joblib.load(CB_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return xgb_clf, cb_clf, tfidf


def prepare_val_test_features(df: pd.DataFrame, tfidf):
    """
    Split data into train/val/test (same split as training),
    then build TF-IDF features for val and test using the loaded vectorizer.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_val_clean = [preprocess_text(t) for t in X_val]
    X_test_clean = [preprocess_text(t) for t in X_test]

    X_val_tfidf = tfidf.transform(X_val_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    return X_val_tfidf, X_test_tfidf, y_val, y_test


def tune_soft_ensemble(
    p_xgb_val: np.ndarray,
    p_cb_val: np.ndarray,
    y_val: np.ndarray,
    weight_grid=None,
    thr_grid=None,
    optimize_for: str = "macro",
):
    """
    Grid search over weights and thresholds on validation set.
    optimize_for: "macro" -> F1_macro, "hate" -> F1 for class 1.
    """
    if weight_grid is None:
        weight_grid = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    if thr_grid is None:
        thr_grid = np.linspace(0.3, 0.7, 9)  # 0.30, 0.35, ..., 0.70

    best_score = -1.0
    best_w = None
    best_thr = None
    results = []

    for w in weight_grid:
        p_ens = w * p_xgb_val + (1.0 - w) * p_cb_val

        for thr in thr_grid:
            y_pred = (p_ens >= thr).astype(int)

            f1_macro = f1_score(y_val, y_pred, average="macro")
            f1_hate = f1_score(y_val, y_pred, pos_label=1)

            results.append((w, thr, f1_macro, f1_hate))

            if optimize_for == "hate":
                score = f1_hate
            else:
                score = f1_macro

            if score > best_score:
                best_score = score
                best_w = w
                best_thr = thr

    return best_w, best_thr, best_score, results


def run_ensemble():
    """Run soft-voting ensemble between XGBoost and CatBoost."""
    # 1) Load models + vectorizer
    xgb_clf, cb_clf, tfidf = load_models_and_vectorizer()

    # 2) Load filtered data and prepare val/test features
    df = pd.read_csv(DATA_PATH)
    X_val_tfidf, X_test_tfidf, y_val, y_test = prepare_val_test_features(df, tfidf)

    # 3) Get probabilities on validation
    p_xgb_val = xgb_clf.predict_proba(X_val_tfidf)[:, 1]
    p_cb_val = cb_clf.predict_proba(X_val_tfidf)[:, 1]

    # 4) Grid search weights + threshold on validation
    best_w, best_thr, best_score, results = tune_soft_ensemble(
        p_xgb_val,
        p_cb_val,
        y_val,
        optimize_for="macro",
    )

    print("Best on VAL:")
    print("  w_xgb =", best_w, "w_cb =", 1.0 - best_w)
    print("  threshold =", best_thr)
    print("  F1_macro(val) =", best_score)

    # Optionally save ensemble config
    os.makedirs(os.path.dirname(ENSEMBLE_CONFIG_PATH), exist_ok=True)
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "w_xgb": best_w,
                "w_cb": 1.0 - best_w,
                "threshold": best_thr,
                "best_F1_macro_val": best_score,
            },
            f,
            indent=2,
        )
    print(f"Saved ensemble config to: {ENSEMBLE_CONFIG_PATH}")

    # 5) Evaluate on TEST using best weight + threshold
    p_xgb_test = xgb_clf.predict_proba(X_test_tfidf)[:, 1]
    p_cb_test = cb_clf.predict_proba(X_test_tfidf)[:, 1]

    p_ens_test = best_w * p_xgb_test + (1.0 - best_w) * p_cb_test
    y_ens_test = (p_ens_test >= best_thr).astype(int)

    print("\n=== Ensemble (soft vote tuned) - TEST ===")
    print(classification_report(y_test, y_ens_test, digits=3))
    print("F1_macro (test):", f1_score(y_test, y_ens_test, average="macro"))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_ens_test))


if __name__ == "__main__":
    run_ensemble()
