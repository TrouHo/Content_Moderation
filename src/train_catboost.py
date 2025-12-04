import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from src.features import split_data, build_tfidf


TOTAL_ROUNDS = 400       # total CatBoost iterations
CHUNK_ROUNDS = 50        # iterations per chunk

DATA_PATH = "data/vozhsd_filtered_full.csv"
CHECKPOINT_PATH = os.path.join("models", "catboost_checkpoint.pkl")
FINAL_MODEL_PATH = os.path.join("models", "catboost_model.pkl")
TFIDF_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def eval_cb(name, model, pool_tr, y_tr, pool_te, y_te) -> None:
    """Evaluate CatBoost classifier on train and test pools."""
    print(f"\n=== {name} - TRAIN ===")
    pred_tr = model.predict(pool_tr).astype(int).ravel()
    print(classification_report(y_tr, pred_tr, digits=3))
    print("F1_macro (train):", f1_score(y_tr, pred_tr, average="macro"))

    print(f"\n=== {name} - TEST ===")
    pred_te = model.predict(pool_te).astype(int).ravel()
    print(classification_report(y_te, pred_te, digits=3))
    print("F1_macro (test):", f1_score(y_te, pred_te, average="macro"))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_te, pred_te))


def train_catboost(
    data_path: str = DATA_PATH,
    total_rounds: int = TOTAL_ROUNDS,
    chunk_rounds: int = CHUNK_ROUNDS,
) -> None:
    """Train CatBoost with TF-IDF features and checkpointing."""
    # 1) Load filtered data
    df = pd.read_csv(data_path)

    # 2) Split and build TF-IDF
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf = build_tfidf(
        X_train, X_val, X_test
    )

    # 3) Build CatBoost pools
    train_pool = Pool(X_train_tfidf, y_train)
    val_pool = Pool(X_val_tfidf, y_val)
    test_pool = Pool(X_test_tfidf, y_test)

    # 4) Compute class weights (inverse frequency)
    class_counts = np.bincount(y_train)
    total = class_counts.sum()
    class_weights = [total / (2 * c) for c in class_counts]
    print("class_counts:", class_counts)
    print("class_weights:", class_weights)

    # 5) Load checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        state = joblib.load(CHECKPOINT_PATH)
        cb_rounds_done = state["rounds"]
        cb_prev_model = state["model"]
        print(
            f"[CatBoost] Loaded checkpoint from {CHECKPOINT_PATH}, "
            f"rounds_done = {cb_rounds_done}"
        )
    else:
        cb_rounds_done = 0
        cb_prev_model = None
        print("[CatBoost] No checkpoint found, training from scratch.")

    # Ensure models/ exists
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # 6) Incremental training loop
    while cb_rounds_done < total_rounds:
        rounds_to_train = min(chunk_rounds, total_rounds - cb_rounds_done)
        print(
            f"\n[CatBoost] Training extra {rounds_to_train} iterations "
            f"(from {cb_rounds_done} to {cb_rounds_done + rounds_to_train})"
        )

        cb_clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            learning_rate=0.1,
            depth=6,
            iterations=rounds_to_train,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=50,
            class_weights=class_weigh
