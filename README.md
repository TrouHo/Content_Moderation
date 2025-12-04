# Vietnamese Content Moderation (VOZ-HSD)

This repository contains a classic ML pipeline for Vietnamese content moderation using the **VOZ-HSD** dataset (Hugging Face: `tarudesu/VOZ-HSD`).  
The goal is to automatically classify user comments as:

- `0` – clean / acceptable content  
- `1` – hateful / toxic content  

The project uses:

- Text preprocessing + TF–IDF features
- **XGBoost** and **CatBoost** as main classifiers
- An optional **soft-voting ensemble** on top of both models


## 1. Dataset

- Source: Hugging Face, dataset ID: `tarudesu/VOZ-HSD`
- Split used: `train`
- Columns (renamed inside the pipeline):
  - `text` – raw comment
  - `label` – 0 (clean) or 1 (hate)
  - `prob` – weak-label probability from the original dataset

The data pipeline:

- Loads the full VOZ-HSD training set
- Renames columns (`texts → text`, `labels → label`, `probs → prob`)
- Adds simple length features (`len_char`, `len_word`)
- Filters examples based on:
  - Probability thresholds (asymmetric for class 0 vs class 1)
  - Minimum number of words

The filtered dataset is saved as a CSV file under `data/`.


## 2. Environment & Installation

Python version: 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
