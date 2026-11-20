from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(r"D:\image-forensics-project")
DATASET_DIR = PROJECT_DIR / "final_dataset"
IMAGE_DIR = DATASET_DIR / "images"
MANIFEST = DATASET_DIR / "final_balanced_manifest.csv"
SPLIT_DIR = DATASET_DIR / "splits"

CLASSES = ["REAL", "TAMPERED", "SYNTHETIC"]
RANDOM_SEED = 42


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    required = {"class_label", "category", "sha256", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if len(df) != 120_000:
        raise ValueError(f"Expected 120,000 records; found {len(df)}")

    if df["sha256"].duplicated().any():
        raise ValueError("Duplicate SHA-256 hashes found. Stop and investigate.")

    if set(df["class_label"].unique()) != set(CLASSES):
        raise ValueError("Unexpected class labels found in manifest.")

    for label in CLASSES:
        count = int((df["class_label"] == label).sum())
        if count != 40_000:
            raise ValueError(f"{label} should have 40,000 records, found {count}")

    # Verify every image exists before creating the splits.
    for row in df.itertuples(index=False):
        path = DATASET_DIR / str(row.image_path).replace("/", "\\")
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

    train_parts = []
    validation_parts = []
    test_parts = []

    # Split each class separately and preserve category proportions.
    for label in CLASSES:
        class_df = df[df["class_label"] == label].copy()

        train_df, temp_df = train_test_split(
            class_df,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=class_df["category"],
        )

        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temp_df["category"],
        )

        train_parts.append(train_df)
        validation_parts.append(validation_df)
        test_parts.append(test_df)

    splits = {
        "train": pd.concat(train_parts).sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True),
        "validation": pd.concat(validation_parts).sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True),
        "test": pd.concat(test_parts).sample(
            frac=1, random_state=RANDOM_SEED
        ).reset_index(drop=True),
    }

    expected_totals = {
        "train": 96_000,
        "validation": 12_000,
        "test": 12_000,
    }

    for split_name, split_df in splits.items():
        if len(split_df) != expected_totals[split_name]:
            raise ValueError(f"Incorrect size for {split_name}")

        counts = split_df["class_label"].value_counts().to_dict()
        expected_per_class = (
            32_000 if split_name == "train" else 4_000
        )

        for label in CLASSES:
            if counts.get(label, 0) != expected_per_class:
                raise ValueError(
                    f"{split_name}: incorrect count for {label}"
                )

    # Check that no image hash appears in multiple splits.
    train_hashes = set(splits["train"]["sha256"])
    val_hashes = set(splits["validation"]["sha256"])
    test_hashes = set(splits["test"]["sha256"])

    if train_hashes & val_hashes or train_hashes & test_hashes:
        raise ValueError("Image hash overlap involving training set.")
    if val_hashes & test_hashes:
        raise ValueError("Image hash overlap between validation and test.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_df.to_csv(SPLIT_DIR / f"{split_name}.csv", index=False)

    class_report = pd.concat(
        [
            split_df.groupby("class_label").size().rename(split_name)
            for split_name, split_df in splits.items()
        ],
        axis=1,
    ).fillna(0).astype(int)

    class_report.to_csv(SPLIT_DIR / "split_class_counts.csv")

    category_report = (
        pd.concat(
            [
                split_df.groupby(["class_label", "category"])
                .size()
                .rename(split_name)
                for split_name, split_df in splits.items()
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
    )
    category_report.to_csv(SPLIT_DIR / "split_category_counts.csv")

    print("\nSPLIT CREATION SUCCESSFUL")
    for name, split_df in splits.items():
        print(f"{name}: {len(split_df):,} images")
        print(split_df["class_label"].value_counts().to_dict())

    print(f"\nCSV files saved to: {SPLIT_DIR}")
    print("No images have been moved yet.")


if __name__ == "__main__":
    main()
