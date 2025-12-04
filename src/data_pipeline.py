from datasets import load_dataset
import pandas as pd
import os

# Default filter thresholds
MIN_PROB_POS = 0.6   # for label=1 (HATE)
MIN_PROB_NEG = 0.9   # for label=0 (CLEAN)
MIN_LEN_WORD = 3


def load_full_voz_hsd() -> pd.DataFrame:
    """
    Load the full VOZ-HSD train split into a pandas DataFrame.
    """
    ds = load_dataset("tarudesu/VOZ-HSD")
    train_ds = ds["train"]

    df = train_ds.to_pandas()
    df = df.rename(columns={"texts": "text", "labels": "label", "probs": "prob"})

    df["len_char"] = df["text"].astype(str).str.len()
    df["len_word"] = df["text"].astype(str).str.split().str.len()

    return df


def filter_voz_hsd(
    df: pd.DataFrame,
    min_prob_pos: float = MIN_PROB_POS,
    min_prob_neg: float = MIN_PROB_NEG,
    min_len_word: int = MIN_LEN_WORD,
) -> pd.DataFrame:
    """
    Apply probability-based filtering and length filtering.
    """
    mask_pos = (df["label"] == 1) & (df["prob"] >= min_prob_pos)
    mask_neg = (df["label"] == 0) & (df["prob"] >= min_prob_neg)

    df_filtered = df[mask_pos | mask_neg].copy()
    df_filtered = df_filtered[df_filtered["len_word"] >= min_len_word]

    return df_filtered


def run_pipeline(output_path: str = "data/vozhsd_filtered_full.csv") -> None:
    """
    Run the full pipeline: load VOZ-HSD, filter, and save to CSV.
    """
    df = load_full_voz_hsd()
    print("Raw shape:", df.shape)
    print("\nLabel distribution (before filter):")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True))

    df_filtered = filter_voz_hsd(df)
    print("\nFiltered shape:", df_filtered.shape)
    print("\nLabel distribution (after filter):")
    print(df_filtered["label"].value_counts())
    print(df_filtered["label"].value_counts(normalize=True))

    print("\nProb stats (after filter):")
    print(df_filtered["prob"].describe())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    print(f"\nSaved filtered dataset to: {output_path}")


if __name__ == "__main__":
    run_pipeline()
