
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
    / "content_v3_clean"
)

OUTPUT_CSV = OUTPUT_DIR / "image_content_analysis_v3_deduplicated.csv"
DUPLICATES_CSV = OUTPUT_DIR / "removed_duplicate_records.csv"

# Read the unflagged working dataset
df = pd.read_csv(INPUT_CSV)

# Normalize hashes
df["sha256"] = df["sha256"].fillna("").astype(str).str.strip()

# Stop if a hash is missing
missing = df["sha256"].eq("")
if missing.any():
    raise ValueError(
        f"Found {missing.sum()} records without hashes. "
        "Review them before deduplication."
    )

# Safety check: ensure one hash does not have conflicting class labels
label_counts = df.groupby("sha256")["class_label"].nunique()
conflicting_hashes = label_counts[label_counts > 1].index

if len(conflicting_hashes) > 0:
    raise ValueError(
        f"Found {len(conflicting_hashes)} conflicting hashes. "
        "Resolve them before deduplication."
    )

# Keep one record per hash, preserving the first occurrence
deduplicated = df.drop_duplicates(subset="sha256", keep="first").copy()

# Save the extra occurrences separately for auditing
duplicate_records = df[df.duplicated(subset="sha256", keep="first")].copy()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
deduplicated.to_csv(OUTPUT_CSV, index=False)
duplicate_records.to_csv(DUPLICATES_CSV, index=False)

print("========== DEDUPLICATION RESULTS ==========")
print(f"Original records: {len(df):,}")
print(f"Unique records retained: {len(deduplicated):,}")
print(f"Extra duplicate records: {len(duplicate_records):,}")
print(f"Unique hashes remaining: {deduplicated['sha256'].nunique():,}")

print("\nClass counts after deduplication:")
print(deduplicated["class_label"].value_counts().to_string())

print("\nDeduplicated CSV:")
print(OUTPUT_CSV)

print("\nRemoved duplicate records report:")
print(DUPLICATES_CSV)

print("\nOriginal CSV and ZIP archives were not modified.")
