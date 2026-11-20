from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\image-forensics-project")

INPUT_CSV = (
    PROJECT_DIR
    / "analysis"
    / "content_v3_clean"
    / "image_content_analysis_v3_deduplicated.csv"
)

OUTPUT_DIR = PROJECT_DIR / "analysis" / "final_dataset"

MANIFEST_PATH = OUTPUT_DIR / "final_balanced_manifest.csv"
CATEGORY_REPORT_PATH = OUTPUT_DIR / "final_category_counts.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "final_class_counts.csv"

RANDOM_SEED = 42

# Equal number of images per category in EACH authenticity class.
CATEGORY_QUOTAS = {
    "people and portraits": 13468,
    "animals and wildlife": 5324,
    "everyday objects and technology": 5178,
    "ground vehicles": 5017,
    "crowds, meetings, and public gatherings": 2927,
    "nature and landscapes": 2767,
    "food and prepared meals": 2486,
    "buildings and urban scenes": 1694,
    "aircraft and helicopters": 910,
    "plants, flowers, and crops": 229,
}

CLASSES = ["REAL", "TAMPERED", "SYNTHETIC"]

# ============================================================
# LOAD AND VALIDATE THE SOURCE DATA
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "class_label",
        "category",
        "sha256",
        "archive",
        "member",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Normalize labels for consistent filtering.
    df["class_label"] = (
        df["class_label"].astype(str).str.strip().str.upper()
    )

    df["category"] = (
        df["category"].astype(str).str.strip().str.lower()
    )

    df["sha256"] = (
        df["sha256"].astype(str).str.strip().str.lower()
    )

    # Check class labels.
    unexpected_classes = (
        set(df["class_label"].unique()) - set(CLASSES)
    )

    if unexpected_classes:
        raise ValueError(
            f"Unexpected class labels: {unexpected_classes}"
        )

    # Check categories.
    unexpected_categories = (
        set(df["category"].unique()) - set(CATEGORY_QUOTAS)
    )

    if unexpected_categories:
        raise ValueError(
            "Unexpected categories found in the CSV: "
            f"{unexpected_categories}"
        )

    # Do not silently use records without hashes.
    if df["sha256"].isna().any() or df["sha256"].eq("").any():
        raise ValueError("Missing SHA-256 hashes found.")

    # The source is expected to be deduplicated already.
    if df["sha256"].duplicated().any():
        duplicate_count = int(df["sha256"].duplicated().sum())
        raise ValueError(
            f"Source still contains {duplicate_count} "
            "repeated SHA-256 records. Recheck deduplication."
        )

    # Each hash should have exactly one class and category.
    class_conflicts = df.groupby("sha256")["class_label"].nunique()
    category_conflicts = df.groupby("sha256")["category"].nunique()

    if (class_conflicts > 1).any():
        raise ValueError("Conflicting class labels for image hashes.")

    if (category_conflicts > 1).any():
        raise ValueError("Conflicting categories for image hashes.")

    target_per_class = sum(CATEGORY_QUOTAS.values())

    print("\nSOURCE DATA")
    print(f"Records available: {len(df):,}")
    print(f"Unique SHA-256 hashes: {df['sha256'].nunique():,}")
    print(f"Target per class: {target_per_class:,}")
    print(f"Target total: {target_per_class * len(CLASSES):,}")

    # ========================================================
    # CHECK THAT EVERY QUOTA CAN BE MET
    # ========================================================

    print("\nAVAILABLE IMAGES BY CATEGORY AND CLASS")

    availability = (
        df.groupby(["category", "class_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=list(CATEGORY_QUOTAS),
            columns=CLASSES,
            fill_value=0,
        )
    )

    print(availability.to_string())

    for category, quota in CATEGORY_QUOTAS.items():
        for class_name in CLASSES:
            available = int(availability.loc[category, class_name])

            if available < quota:
                raise ValueError(
                    f"Not enough images for {category!r}, "
                    f"class {class_name}: need {quota:,}, "
                    f"have {available:,}. No manifest created."
                )

    # ========================================================
    # SELECT EXACT QUOTAS
    # ========================================================

    selected_parts = []

    for category, quota in CATEGORY_QUOTAS.items():
        for class_name in CLASSES:
            candidates = df[
                (df["category"] == category)
                & (df["class_label"] == class_name)
            ]

            selected = candidates.sample(
                n=quota,
                random_state=RANDOM_SEED,
                replace=False,
            )

            selected_parts.append(selected)

    final_df = pd.concat(selected_parts, ignore_index=True)

    # Shuffle final rows reproducibly.
    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Assign stable manifest IDs.
    final_df.insert(
        0,
        "sample_id",
        [f"IMG_{i:06d}" for i in range(1, len(final_df) + 1)],
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    actual_class_counts = final_df["class_label"].value_counts()

    if len(final_df) != target_per_class * len(CLASSES):
        raise AssertionError("Final dataset total is incorrect.")

    if final_df["sha256"].nunique() != len(final_df):
        raise AssertionError("Duplicate image hashes in final dataset.")

    for class_name in CLASSES:
        actual = int(actual_class_counts.get(class_name, 0))

        if actual != target_per_class:
            raise AssertionError(
                f"{class_name}: expected {target_per_class}, "
                f"got {actual}"
            )

    category_report = (
        final_df.groupby(["category", "class_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=list(CATEGORY_QUOTAS),
            columns=CLASSES,
            fill_value=0,
        )
    )

    category_report["quota_per_class"] = [
        CATEGORY_QUOTAS[category]
        for category in category_report.index
    ]

    class_report = (
        final_df["class_label"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .rename_axis("class_label")
        .reset_index(name="image_count")
    )

    # Save all output files.
    final_df.to_csv(MANIFEST_PATH, index=False)
    category_report.to_csv(CATEGORY_REPORT_PATH)
    class_report.to_csv(CLASS_REPORT_PATH, index=False)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========== FINAL DATASET CREATED ==========")
    print(f"Total images: {len(final_df):,}")
    print(f"Unique hashes: {final_df['sha256'].nunique():,}")

    print("\nCLASS COUNTS")
    print(class_report.to_string(index=False))

    print("\nCATEGORY COUNTS")
    print(category_report.to_string())

    print("\nOUTPUT FILES")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Category report: {CATEGORY_REPORT_PATH}")
    print(f"Class report: {CLASS_REPORT_PATH}")

    print("\nSUCCESS: All quotas and uniqueness checks passed.")
    print("Original CSV files and ZIP archives were not modified.")


if __name__ == "__main__":
    main()
