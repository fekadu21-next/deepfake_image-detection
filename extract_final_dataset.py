from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import shutil
import zipfile
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"D:\image-forensics-project")

SOURCE_ZIP_DIR = PROJECT / "raw_datasets" / "SECOUND"

SOURCE_MANIFEST = (
    PROJECT / "analysis" / "final_dataset"
    / "final_balanced_manifest.csv"
)

SOURCE_CLASS_REPORT = (
    PROJECT / "analysis" / "final_dataset"
    / "final_class_counts.csv"
)

SOURCE_CATEGORY_REPORT = (
    PROJECT / "analysis" / "final_dataset"
    / "final_category_counts.csv"
)

# This is your permanent, extracted dataset folder.
DESTINATION = PROJECT / "final_dataset"

IMAGES_DIR = DESTINATION / "images"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_archive_name(value):
    name = Path(str(value).replace("\\", "/")).name.lower()

    if name.endswith(".zip"):
        name = name[:-4]

    return name


def main():
    parser = argparse.ArgumentParser(
        description="Extract and verify the selected final dataset."
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="Actually extract images. Without this, run a dry run.",
    )
    args = parser.parse_args()

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {SOURCE_MANIFEST}"
        )
    if not SOURCE_ZIP_DIR.exists():
        raise FileNotFoundError(
            f"ZIP folder not found: {SOURCE_ZIP_DIR}"
        )
    df = pd.read_csv(SOURCE_MANIFEST)
    required = {
        "sample_id",
        "class_label",
        "archive",
        "member",
        "sha256",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    df["class_label"] = (
        df["class_label"].astype(str).str.strip().str.upper()
    )

    df["sha256"] = (
        df["sha256"].astype(str).str.strip().str.lower()
    )

    if len(df) != 120_000:
        raise ValueError(
            f"Expected 120,000 rows, found {len(df):,}."
        )

    if df["sha256"].duplicated().any():
        raise ValueError("Duplicate SHA-256 hashes in manifest.")

    if set(df["class_label"].unique()) != {
        "REAL", "TAMPERED", "SYNTHETIC"
    }:
        raise ValueError("Unexpected authenticity class labels.")

    expected_counts = {
        "REAL": 40_000,
        "TAMPERED": 40_000,
        "SYNTHETIC": 40_000,
    }
    actual_counts = df["class_label"].value_counts().to_dict()

    if actual_counts != expected_counts:
        raise ValueError(
            f"Incorrect class counts: {actual_counts}"
        )

    zip_paths = list(SOURCE_ZIP_DIR.glob("*.zip"))

    if not zip_paths:
        raise FileNotFoundError(
            f"No ZIP files found in {SOURCE_ZIP_DIR}"
        )

    zip_lookup = {}

    for path in zip_paths:
        zip_lookup[path.name.lower()] = path
        zip_lookup[path.stem.lower()] = path

    # Resolve every archive reference in the manifest.
    archive_paths = {}

    for value in df["archive"].dropna().unique():
        key = normalize_archive_name(value)

        if key not in zip_lookup:
            raise ValueError(
                f"Cannot match manifest archive value {value!r} "
                f"to a ZIP file in {SOURCE_ZIP_DIR}."
            )

        archive_paths[key] = zip_lookup[key]

    # Open ZIPs and inspect every selected member before extraction.
    zip_handles = {}
    zip_infos = {}

    try:
        for key, path in archive_paths.items():
            handle = zipfile.ZipFile(path, "r")
            zip_handles[key] = handle
            # A lookup by exact member path.
            zip_infos[key] = {
                info.filename: info
                for info in handle.infolist()
                if not info.is_dir()
            }

        # Preflight all requested images.
        total_uncompressed_bytes = 0
        missing_members = []

        for row in df.itertuples(index=False):
            archive_key = normalize_archive_name(row.archive)
            member = str(row.member)

            info = zip_infos[archive_key].get(member)

            if info is None:
                missing_members.append(
                    f"{row.archive}: {member}"
                )
                continue

            total_uncompressed_bytes += info.file_size

        if missing_members:
            print("Missing ZIP members (first 20):")
            for member in missing_members[:20]:
                print("  ", member)

            raise ValueError(
                f"{len(missing_members)} selected ZIP members "
                "could not be found. Nothing was extracted."
            )

        estimated_gb = total_uncompressed_bytes / (1024 ** 3)

        print("\n========== EXTRACTION PREFLIGHT ==========")
        print(f"Selected images: {len(df):,}")
        print(f"REAL: {actual_counts['REAL']:,}")
        print(f"TAMPERED: {actual_counts['TAMPERED']:,}")
        print(f"SYNTHETIC: {actual_counts['SYNTHETIC']:,}")
        print(f"Estimated uncompressed image size: {estimated_gb:.2f} GB")
        print(f"Destination: {DESTINATION}")

        DESTINATION.mkdir(parents=True, exist_ok=True)

        free_bytes = shutil.disk_usage(DESTINATION).free
        free_gb = free_bytes / (1024 ** 3)

        print(f"Free space on destination drive: {free_gb:.2f} GB")

        # Leave additional space for reports, temporary files,
        # and later training operations.
        if free_bytes < total_uncompressed_bytes + 2 * (1024 ** 3):
            raise OSError(
                "Not enough free disk space for extraction plus "
                "a 2 GB safety margin. Free space before continuing."
            )

        if not args.extract:
            print("\nDRY RUN ONLY: no images were extracted.")
            print("If the paths and size look correct, run:")
            print("python extract_final_dataset.py --extract")
            return

        # ====================================================
        # EXTRACT SELECTED IMAGES AND VERIFY SHA-256
        # ====================================================

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        verified = 0
        extracted = 0
        reused = 0
        class_counts = Counter()

        for index, row in enumerate(
            df.itertuples(index=False), start=1
        ):
            class_name = row.class_label
            archive_key = normalize_archive_name(row.archive)
            member = str(row.member)
            expected_hash = str(row.sha256).lower()

            info = zip_infos[archive_key][member]

            suffix = Path(member).suffix.lower()

            if not suffix or len(suffix) > 10:
                suffix = ".img"

            class_dir = IMAGES_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            output_path = class_dir / f"{row.sample_id}{suffix}"

            # Allow safe restart after an interrupted extraction.
            if output_path.exists():
                if sha256_file(output_path) == expected_hash:
                    reused += 1
                    verified += 1
                    class_counts[class_name] += 1
                    continue

                output_path.unlink()

            handle = zip_handles[archive_key]

            digest = hashlib.sha256()

            with handle.open(info, "r") as source:
                with output_path.open("wb") as target:
                    for chunk in iter(
                        lambda: source.read(1024 * 1024), b""
                    ):
                        digest.update(chunk)
                        target.write(chunk)

            actual_hash = digest.hexdigest()

            if actual_hash != expected_hash:
                output_path.unlink(missing_ok=True)

                raise ValueError(
                    f"SHA-256 mismatch for {row.sample_id}\n"
                    f"Expected: {expected_hash}\n"
                    f"Actual:   {actual_hash}"
                )

            extracted += 1
            verified += 1
            class_counts[class_name] += 1

            if index % 500 == 0 or index == len(df):
                print(
                    f"Progress: {index:,}/{len(df):,} | "
                    f"newly extracted: {extracted:,} | "
                    f"already verified: {reused:,}"
                )

        if verified != 120_000:
            raise AssertionError(
                f"Expected 120,000 verified images, got {verified:,}"
            )

        for class_name, expected in expected_counts.items():
            actual = len(list((IMAGES_DIR / class_name).iterdir()))

            if actual != expected:
                raise AssertionError(
                    f"{class_name}: expected {expected:,} files, "
                    f"found {actual:,}"
                )

        # Copy the exact manifest and reports into the final folder.
        final_manifest = DESTINATION / "final_balanced_manifest.csv"
        final_class_report = DESTINATION / "final_class_counts.csv"
        final_category_report = DESTINATION / "final_category_counts.csv"

        shutil.copy2(SOURCE_MANIFEST, final_manifest)

        if SOURCE_CLASS_REPORT.exists():
            shutil.copy2(SOURCE_CLASS_REPORT, final_class_report)

        if SOURCE_CATEGORY_REPORT.exists():
            shutil.copy2(SOURCE_CATEGORY_REPORT, final_category_report)

        # Save paths to extracted images in the final manifest.
        final_df = pd.read_csv(final_manifest)

        final_df["image_path"] = final_df.apply(
            lambda row: (
                f"images/{str(row['class_label']).upper()}/"
                f"{row['sample_id']}"
                f"{Path(str(row['member'])).suffix.lower() or '.img'}"
            ),
            axis=1,
        )

        final_df.to_csv(final_manifest, index=False)

        metadata = {
            "dataset_name": "Image Authenticity Dataset",
            "total_images": 120000,
            "class_counts": expected_counts,
            "unique_sha256_hashes": int(
                final_df["sha256"].nunique()
            ),
            "image_content_verified_against_sha256": True,
            "classes": ["REAL", "TAMPERED", "SYNTHETIC"],
            "source_manifest": str(SOURCE_MANIFEST),
            "random_seed_for_selection": 42,
        }

        with (DESTINATION / "dataset_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)

        print("\n========== EXTRACTION SUCCESS ==========")
        print(f"Images verified: {verified:,}")
        print(f"Newly extracted: {extracted:,}")
        print(f"Previously extracted and reverified: {reused:,}")
        print(f"Final folder: {DESTINATION}")
        print(f"Manifest: {final_manifest}")
        print("All extracted image SHA-256 hashes matched.")

    finally:
        for handle in zip_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
