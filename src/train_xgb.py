import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from src.features import split_data, build_tfidf


TOTAL_ROUNDS = 400        # total number of trees
CHUNK_ROUNDS = 50         # trees added per chunk

DATA_PATH = "data/vozhsd_filtered_full.csv"
CHECKPOINT_PATH = os.path.join("models", "xgb_checkpoint.pkl")
FINAL_MODEL_PATH = os.path.join("models", "xgb_model.pkl")
TFIDF_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def eval_model(
    name: str,
    model: xgb.XGBClassifier,
    X_tr,
    y_tr,
    X_te,
    y_te,
) -> None:
    """Evaluate XGBoost classifier on train and test."""
    print(f"\n=== {name} - TRAIN ===")
    pred_tr = model.predict(X_tr)
    print(classification_report(y_tr, pred_tr, digits=3))
    print("F1_macro (train):", f1_score(y_tr, pred_tr, average="macro"))

    print(f"\n=== {name} - TEST ===")
    pred_te = model.predict(X_te)
    print(classification_report(y_te, pred_te, digits=3))
    print("F1_macro (test):", f1_score(y_te, pred_te, average="macro"))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_te, pred_te))


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced binary classification."""
    neg, pos = np.bincount(y_train)
    print("neg, pos:", neg, pos)
    scale_pos_weight = neg / pos
    print("scale_pos_weight:", scale_pos_weight)
    return scale_pos_weight


def train_xgb(
    data_path: str = DATA_PATH,
    total_rounds: int = TOTAL_ROUNDS,
    chunk_rounds: int = CHUNK_ROUNDS,
) -> Tuple[xgb.XGBClassifier, object]:
    """
    Train XGBoost with TF-IDF features and checkpointing.
    Returns the trained model and the fitted TF-IDF vectorizer.
    """
    # 1) Load filtered data
    df = pd.read_csv(data_path)

    # 2) Split and build TF-IDF
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf = build_tfidf(
        X_train,
        X_val,
        X_test,
    )

    # 3) Compute scale_pos_weight
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # 4) Load checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        state = joblib.load(CHECKPOINT_PATH)
        rounds_done = state["rounds"]
        prev_model = state["model"]
        print(
            f"\nLoaded checkpoint from {CHECKPOINT_PATH}, "
            f"rounds_done = {rounds_done}"
        )
    else:
        rounds_done = 0
        prev_model = None
        print("\nNo checkpoint found, training from scratch.")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # 5) Incremental training loop
    while rounds_done < total_rounds:
        rounds_to_train = min(chunk_rounds, total_rounds - rounds_done)
        print(
            f"\nTraining extra {rounds_to_train} trees "
            f"(from {rounds_done} to {rounds_done + rounds_to_train})"
        )

        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=rounds_to_train,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )

        xgb_clf.fit(
            X_train_tfidf,
            y_train,
            eval_set=[(X_val_tfidf, y_val)],
            verbose=50,
            xgb_model=prev_model,
        )

        rounds_done += rounds_to_train
        state = {
            "model": xgb_clf,
            "rounds": rounds_done,
        }
        joblib.dump(state, CHECKPOINT_PATH)
        print(
            f"Checkpoint saved at {CHECKPOINT_PATH} "
            f"with {rounds_done} trees."
        )

        prev_model = xgb_clf

    print("\nTraining finished.")
    final_model = prev_model

    # 6) Save final model and TF-IDF vectorizer
    joblib.dump(final_model, FINAL_MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)
    print(f"Saved final XGBoost model to: {FINAL_MODEL_PATH}")
    print(f"Saved TF-IDF vectorizer to: {TFIDF_PATH}")

    # 7) Final evaluation
    eval_model(
        "XGBoost (checkpointed)",
        final_model,
        X_train_tfidf,
        y_train,
        X_test_tfidf,
        y_test,
    )

    return final_model, tfidf


if __name__ == "__main__":
    train_xgb()
