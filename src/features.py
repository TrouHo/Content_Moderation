import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(s: str) -> str:
    """
    Basic text normalization: lowercase, remove URLs and mentions, collapse spaces.
    """
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " USER ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_data(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a filtered DataFrame into train/val/test with stratification.
    """
    X = df[text_col].astype(str).values
    y = df[label_col].values

    # first split off (val + test)
    temp_size = val_size + test_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=random_state,
    )

    # split temp into val and test
    relative_test_size = test_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Val size  : {len(X_val)}")
    print(f"Test size : {len(X_test)}")

    print("\nTrain label distribution:", np.bincount(y_train))

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tfidf(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    ngram_range=(1, 2),
    min_df: int = 5,
    max_df: float = 0.9,
    sublinear_tf: bool = True,
):
    """
    Preprocess text and fit/apply a TfidfVectorizer on train/val/test.
    Returns TF-IDF matrices and the fitted vectorizer.
    """
    X_train_clean = [preprocess_text(t) for t in X_train]
    X_val_clean = [preprocess_text(t) for t in X_val]
    X_test_clean = [preprocess_text(t) for t in X_test]

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )

    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_val_tfidf = tfidf.transform(X_val_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    print("TF-IDF shapes:")
    print("Train:", X_train_tfidf.shape)
    print("Val  :", X_val_tfidf.shape)
    print("Test :", X_test_tfidf.shape)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf
