import pandas as pd
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

# Input dataset. Change this path to the dataset you want to adapt.
datasetPath = "dataset/fake_or_real_news.csv"

# Output dataset (converted)
outputPath = "dataset/converted_dataset.csv"

# -----------------------------------------------------
# Column mapping
# change the values according to the new dataset's column names
# -----------------------------------------------------

COLUMN_MAPPING = {
    "title": "title",
    "text": "text",
    "is_fake": "label"
}

# ---------------------------------------------
# Label mapping
# change the values according to the new dataset's labals
# ---------------------------------------------

LABEL_MAPPING = {
    "FAKE": 1,
    "REAL": 0
}

# Optional: normalize labels to uppercase before mapping
NORMALIZE_LABELS = True

# =====================================================
# STANDARD SCHEMA (DO NOT CHANGE)
# =====================================================

STANDARD_COLUMNS = [
    "title",
    "text",
    "subject",
    "date",
    "is_fake"
]


# =====================================================
# ADAPTATION LOGIC
# =====================================================

def adapt_dataset():
    print(f"Loading dataset: {datasetPath}")
    df = pd.read_csv(datasetPath)

    new_df = pd.DataFrame()

    for standard_col in STANDARD_COLUMNS:

        if standard_col in COLUMN_MAPPING:
            original_col = COLUMN_MAPPING[standard_col]

            if original_col in df.columns:
                new_df[standard_col] = df[original_col]
                print(f"Mapped: {original_col} → {standard_col}")
            else:
                new_df[standard_col] = ""
                print(f"Missing column '{original_col}', filled empty.")

        else:
            new_df[standard_col] = ""
            print(f"No mapping for '{standard_col}', filled empty.")

    if "is_fake" in new_df.columns:
        new_df["is_fake"] = map_labels(new_df["is_fake"])

    Path(outputPath).parent.mkdir(parents=True, exist_ok=True)

    new_df.to_csv(outputPath, index=False)

    print("\nDataset successfully converted.")
    print(f"Saved to: {outputPath}")
    print("\nFinal columns:")
    print(list(new_df.columns))

def map_labels(series):

    if NORMALIZE_LABELS:
        series = series.astype(str).str.upper()

    mapped = series.map(LABEL_MAPPING)

    if mapped.isnull().any():
        unknown = series[mapped.isnull()].unique()
        print("\nWarning: unknown labels found:")
        print(unknown)

    return mapped.fillna(0).astype(int)

if __name__ == "__main__":
    adapt_dataset()
