"""Load the raw admission data, clean it, and split it into train/test sets.

Usage (from the project root):
    python src/prepare_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Build paths from this file's location so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "admission.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Rename the columns to snake_case so the API and the model use the same names.
COLUMN_RENAMES = {
    "GRE Score": "gre_score",
    "TOEFL Score": "toefl_score",
    "University Rating": "university_rating",
    "SOP": "sop",
    "LOR": "lor",
    "CGPA": "cgpa",
    "Research": "research",
    "Chance of Admit": "chance_of_admit",
}

TARGET = "chance_of_admit"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    print(f"raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Some column names in the CSV have trailing spaces ("LOR ", "Chance of Admit "),
    # so strip them first or the rename below will not match.
    df.columns = df.columns.str.strip()

    # "Serial No." is just a row number, it has no predictive value.
    df = df.drop(columns=["Serial No."])
    df = df.rename(columns=COLUMN_RENAMES)

    # The dataset is already clean, but drop missing values and duplicates just in case.
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"clean dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Fixed random_state so the split (and the metrics) are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    for name, data in datasets.items():
        # index=False: otherwise pandas saves the index as an extra column.
        data.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"saved {name} {data.shape} -> data/processed/{name}.csv")


if __name__ == "__main__":
    main()
