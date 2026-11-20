from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"D:\image-forensics-project")

INPUT_CSV = (
    PROJECT_DIR
    / "analysis"
    / "content_v3_clean"
    / "image_content_analysis_v3_unflagged.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "analysis"
    / "duplicate_check"
)

LABELS = ["REAL", "TAMPERED", "SYNTHETIC"]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"class_label", "sha256", "archive", "member"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["class_label"] = (
        df["class_label"].astype("string").str.strip().str.upper()
    )
    df["sha256"] = (
        df["sha256"].astype("string").str.strip().str.lower()
    )

    print(f"Input records: {len(df):,}")

    # Check whether hashes are missing or empty.
    missing_hash = df["sha256"].isna() | df["sha256"].eq("")
    missing_hash_df = df[missing_hash].copy()
    missing_hash_df.to_csv(
        OUTPUT_DIR / "missing_hashes.csv", index=False
    )

    # Use records with hashes for duplicate/conflict analysis.
    valid = df.loc[~missing_hash].copy()

    # A hash appearing with multiple authenticity labels is a conflict.
    labels_per_hash = valid.groupby("sha256")["class_label"].nunique()
    conflicting_hashes = labels_per_hash[
        labels_per_hash > 1
    ].index

    conflicting_df = valid[
        valid["sha256"].isin(conflicting_hashes)
    ].sort_values("sha256")

    conflicting_df.to_csv(
        OUTPUT_DIR / "conflicting_label_records.csv",
        index=False,
    )

    # Same hash repeated more than once, regardless of label.
    duplicate_mask = valid.duplicated("sha256", keep=False)
    duplicates_df = valid[duplicate_mask].sort_values("sha256")

    duplicates_df.to_csv(
        OUTPUT_DIR / "all_duplicate_hash_groups.csv",
        index=False,
    )

    # Repeated hashes where all copies have the same label.
    same_label_hashes = labels_per_hash[
        labels_per_hash == 1
    ].index

    same_label_df = valid[
        valid["sha256"].isin(same_label_hashes)
    ]

    same_label_duplicates = same_label_df[
        same_label_df.duplicated("sha256", keep=False)
    ].sort_values("sha256")

    same_label_duplicates.to_csv(
        OUTPUT_DIR / "same_label_duplicate_groups.csv",
        index=False,
    )

    # Report class counts in the input.
    class_counts = (
        df["class_label"]
        .value_counts(dropna=False)
        .rename_axis("class_label")
        .reset_index(name="record_count")
    )

    class_counts.to_csv(
        OUTPUT_DIR / "input_class_counts.csv",
        index=False,
    )

    print("\n========== CHECK RESULTS ==========")
    print(f"Input records: {len(df):,}")
    print(f"Records with missing/empty hash: {len(missing_hash_df):,}")
    print(f"Distinct hashes: {valid['sha256'].nunique():,}")
    print(f"Hashes with conflicting labels: {len(conflicting_hashes):,}")
    print(f"Records belonging to conflicting hashes: {len(conflicting_df):,}")
    print(f"Records in repeated-hash groups: {len(duplicates_df):,}")
    print(f"Records in same-label repeated-hash groups: {len(same_label_duplicates):,}")

    print("\nClass counts:")
    print(class_counts.to_string(index=False))

    print(f"\nReports saved to:\n{OUTPUT_DIR}")
    print("\nNo source CSV or ZIP archive was modified.")
    print("Review the reports before removing any records.")


if __name__ == "__main__":
    main()
