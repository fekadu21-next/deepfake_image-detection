from pathlib import Path
from collections import Counter
import hashlib
import warnings

import pandas as pd
from PIL import Image

PROJECT_DIR = Path(r"D:\image-forensics-project")
DATASET_DIR = PROJECT_DIR / "final_dataset"
SPLIT_DIR = DATASET_DIR / "splits"
MANIFEST_PATH = DATASET_DIR / "final_balanced_manifest.csv"

CLASSES = ["REAL", "TAMPERED", "SYNTHETIC"]
SPLITS = ["train", "validation", "test"]

EXPECTED = {
    "train": 32_000,
    "validation": 4_000,
    "test": 4_000,
}
OUTPUT_FAILURES = DATASET_DIR / "image_validation_failures.csv"
OUTPUT_SUMMARY = DATASET_DIR / "image_validation_summary.csv"


def calculate_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    required = {"class_label", "sha256", "image_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"Manifest needs these columns: {sorted(required)}"
        )

    if len(manifest) != 120_000:
        raise ValueError(
            f"Expected 120,000 manifest rows; found {len(manifest)}"
        )

    if manifest["sha256"].duplicated().any():
        raise ValueError("Duplicate hashes in manifest. Stop and investigate.")

    manifest_by_hash = manifest.set_index("sha256")["image_path"].to_dict()
    records = []
    failures = []
    seen_hashes = set()
    total = 0
    successful = 0
    counts = Counter()

    for split in SPLITS:
        csv_path = SPLIT_DIR / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing split CSV: {csv_path}. Do not continue."
            )

        df = pd.read_csv(csv_path)
        if not required.issubset(df.columns):
            raise ValueError(f"Required columns missing in {csv_path}")
        expected_total = 96_000 if split == "train" else 12_000
        if len(df) != expected_total:
            raise ValueError(
                f"{split}.csv has {len(df)} rows; expected {expected_total}"
            )

        for label in CLASSES:
            count = int((df["class_label"] == label).sum())
            if count != EXPECTED[split]:
                raise ValueError(
                    f"{split}/{label}: found {count}; "
                    f"expected {EXPECTED[split]}"
                )

        for row in df.itertuples(index=False):
            total += 1
            label = str(row.class_label)
            expected_hash = str(row.sha256).lower()
            relative_path = str(row.image_path).replace("\\", "/")
            path = DATASET_DIR / relative_path

            problems = []

            # Confirm that the path matches the intended split and class.
            if label not in CLASSES:
                problems.append("unexpected_class")
            if path.parent.name != label or path.parent.parent.name != split:
                problems.append("incorrect_folder")

            if expected_hash in seen_hashes:
                problems.append("duplicate_hash_across_splits")
            seen_hashes.add(expected_hash)

            if expected_hash not in manifest_by_hash:
                problems.append("hash_missing_from_manifest")
            elif str(manifest_by_hash[expected_hash]).replace("\\", "/") != relative_path:
                problems.append("manifest_path_mismatch")

            if not path.is_file():
                problems.append("missing_file")
            else:
                try:
                    # Decode and verify the image, not just its existence.
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with Image.open(path) as img:
                            img.verify()

                        with Image.open(path) as img:
                            img.load()
                            width, height = img.size
                            image_format = img.format

                    if width <= 0 or height <= 0:
                        problems.append("invalid_dimensions")

                    if calculate_sha256(path).lower() != expected_hash:
                        problems.append("sha256_mismatch")

                except Exception as exc:
                    width = None
                    height = None
                    image_format = None
                    problems.append(
                        f"decode_or_read_error: {type(exc).__name__}: {exc}"
                    )

            status = "PASS" if not problems else "FAIL"

            records.append({
                "split": split,
                "class_label": label,
                "image_path": relative_path,
                "sha256": expected_hash,
                "status": status,
                "width": locals().get("width") if path.is_file() else None,
                "height": locals().get("height") if path.is_file() else None,
                "format": locals().get("image_format") if path.is_file() else None,
                "problems": "; ".join(problems),
            })

            if status == "PASS":
                successful += 1
                counts[(split, label)] += 1
            else:
                failures.append(records[-1])

            if total % 1_000 == 0:
                print(
                    f"Checked {total:,}/120,000 | "
                    f"Passed: {successful:,} | Failed: {len(failures):,}"
                )

    # Save the complete report and a separate report of failures.
    full_report = pd.DataFrame(records)
    full_report.to_csv(DATASET_DIR / "image_validation_report.csv", index=False)
    pd.DataFrame(failures).to_csv(OUTPUT_FAILURES, index=False)

    summary = (
        full_report.groupby(["split", "class_label", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Total images checked: {total:,}")
    print(f"Passed: {successful:,}")
    print(f"Failed: {len(failures):,}")
    print(f"Full report: {DATASET_DIR / 'image_validation_report.csv'}")
    print(f"Failure report: {OUTPUT_FAILURES}")
    print(f"Summary: {OUTPUT_SUMMARY}")

    if len(failures) == 0 and successful == 120_000:
        print("\nSUCCESS: all 120,000 images passed validation.")
        print("The file hashes match the manifest.")
        print("You can proceed to the small-batch loading test.")
    else:
        print("\nWARNING: some images failed validation.")
        print("Do not delete the ZIP archives or start training yet.")
        print("Review image_validation_failures.csv first.")


if __name__ == "__main__":
    main()
