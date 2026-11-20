from pathlib import Path
import csv
from collections import Counter, defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\image-forensics-project")
DATASET_DIR = PROJECT_DIR / "final_dataset"
SPLITS_DIR = DATASET_DIR / "splits"

CLASS_NAMES = ["REAL", "TAMPERED", "SYNTHETIC"]

EXPECTED_COUNTS = {
    "train": {
        "REAL": 32000,
        "TAMPERED": 32000,
        "SYNTHETIC": 32000,
    },
    "validation": {
        "REAL": 4000,
        "TAMPERED": 4000,
        "SYNTHETIC": 4000,
    },
    "test": {
        "REAL": 4000,
        "TAMPERED": 4000,
        "SYNTHETIC": 4000,
    },
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".tif", ".tiff"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_class(value):
    """Normalize class names from CSV files."""
    value = str(value).strip().upper()

    # Support common naming variations if present in a CSV.
    aliases = {
        "REAL": "REAL",
        "TAMPERED": "TAMPERED",
        "SYNTHETIC": "SYNTHETIC",
    }
    return aliases.get(value, value)

def read_csv_rows(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    print(f"\nCSV: {csv_path.name}")
    print("Columns:", columns)
    print("Rows:", f"{len(rows):,}")

    return rows, columns

def find_column(columns, candidates):
    """Find a column regardless of capitalization."""
    lookup = {column.strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def count_images(folder):
    if not folder.is_dir():
        return None

    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_image_path(value):
    """Resolve absolute or project-relative image paths."""
    path = Path(str(value).strip())

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    errors = []
    warnings = []

    print("=" * 65)
    print("FINAL IMAGE DATASET INTEGRITY CHECK")
    print("=" * 65)
    print("Project:", PROJECT_DIR)
    print("Dataset:", DATASET_DIR)
    print("Expected class order:", CLASS_NAMES)

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_DIR}"
        )

    # Store image hashes and paths to compare the splits.
    hashes_by_split = defaultdict(set)
    paths_by_split = defaultdict(set)
    all_csv_hashes = {}
    total_csv_rows = 0

    # --------------------------------------------------------
    # CHECK 1: Folder structure and actual image counts
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CHECK 1: FOLDER STRUCTURE AND IMAGE COUNTS")
    print("=" * 65)

    for split, class_expectations in EXPECTED_COUNTS.items():
        split_dir = DATASET_DIR / split

        if not split_dir.is_dir():
            errors.append(f"Missing split folder: {split_dir}")
            continue

        print(f"\n{split.upper()}")

        for class_name, expected in class_expectations.items():
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                errors.append(f"Missing class folder: {class_dir}")
                print(f"  ERROR: Missing {class_name} folder")
                continue

            actual = count_images(class_dir)

            print(
                f"  {class_name}: {actual:,} images "
                f"(expected {expected:,})"
            )

            if actual != expected:
                errors.append(
                    f"{split}/{class_name}: expected {expected}, "
                    f"found {actual}"
                )

        # Warn about unexpected subfolders directly inside a split.
        actual_class_folders = {
            p.name.upper()
            for p in split_dir.iterdir()
            if p.is_dir()
        }
        unexpected = actual_class_folders - set(CLASS_NAMES)

        if unexpected:
            warnings.append(
                f"Unexpected folders in {split}: {sorted(unexpected)}"
            )

    # --------------------------------------------------------
    # CHECK 2: CSV rows, paths, classes, and hashes
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CHECK 2: SPLIT CSV FILES")
    print("=" * 65)

    split_csv_files = {
        "train": SPLITS_DIR / "train.csv",
        "validation": SPLITS_DIR / "validation.csv",
        "test": SPLITS_DIR / "test.csv",
    }

    for split, csv_path in split_csv_files.items():
        try:
            rows, columns = read_csv_rows(csv_path)
        except (FileNotFoundError, OSError) as exc:
            errors.append(str(exc))
            continue

        total_csv_rows += len(rows)
        expected_total = sum(EXPECTED_COUNTS[split].values())

        if len(rows) != expected_total:
            errors.append(
                f"{split}.csv: expected {expected_total} rows, "
                f"found {len(rows)}"
            )

        class_col = find_column(
            columns, ["class_label", "class", "label"]
        )
        path_col = find_column(
            columns, ["image_path", "path", "filepath", "file_path"]
        )
        hash_col = find_column(
            columns, ["sha256", "sha_256", "hash"]
        )

        if class_col is None:
            errors.append(f"{csv_path.name} has no class column")
            continue

        if path_col is None:
            errors.append(f"{csv_path.name} has no image path column")
            continue

        if hash_col is None:
            errors.append(
                f"{csv_path.name} has no SHA-256 column; "
                "cannot verify split overlap from CSV hashes"
            )
            continue

        class_counts = Counter()
        missing_paths = 0
        wrong_split_paths = 0
        wrong_class_paths = 0
        missing_hashes = 0

        for row_number, row in enumerate(rows, start=2):
            class_name = normalize_class(row.get(class_col, ""))
            raw_path = row.get(path_col, "")
            image_hash = str(row.get(hash_col, "")).strip().lower()

            if class_name not in CLASS_NAMES:
                errors.append(
                    f"{csv_path.name}, row {row_number}: "
                    f"unexpected class {class_name!r}"
                )
                continue

            class_counts[class_name] += 1

            if not raw_path or not raw_path.strip():
                errors.append(
                    f"{csv_path.name}, row {row_number}: empty image path"
                )
                missing_paths += 1
            else:
                image_path = resolve_image_path(raw_path)
                normalized_path = str(image_path.resolve()).casefold()
                paths_by_split[split].add(normalized_path)

                if not image_path.is_file():
                    missing_paths += 1
                    errors.append(
                        f"{csv_path.name}, row {row_number}: "
                        f"image not found: {image_path}"
                    )
                else:
                    # Verify the CSV path points into the correct split
                    # and class directory.
                    try:
                        relative = image_path.resolve().relative_to(
                            DATASET_DIR.resolve()
                        )
                        parts = relative.parts

                        if len(parts) < 3:
                            wrong_split_paths += 1
                            errors.append(
                                f"Image is not inside a split/class folder: "
                                f"{image_path}"
                            )
                        else:
                            if parts[0].casefold() != split.casefold():
                                wrong_split_paths += 1
                                errors.append(
                                    f"{csv_path.name}, row {row_number}: "
                                    f"path belongs to split {parts[0]!r}"
                                )

                            if parts[1].upper() != class_name:
                                wrong_class_paths += 1
                                errors.append(
                                    f"{csv_path.name}, row {row_number}: "
                                    f"class says {class_name}, but path "
                                    f"folder is {parts[1]!r}"
                                )

                    except ValueError:
                        wrong_split_paths += 1
                        errors.append(
                            f"Image path is outside final_dataset: {image_path}"
                        )

            if not image_hash:
                missing_hashes += 1
                continue

            if len(image_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in image_hash
            ):
                errors.append(
                    f"{csv_path.name}, row {row_number}: "
                    "invalid SHA-256 value"
                )
                continue

            hashes_by_split[split].add(image_hash)

            previous = all_csv_hashes.get(image_hash)

            if previous is not None:
                previous_split, previous_path = previous

                # Repeated hashes inside one split also indicate duplicates.
                if previous_split == split:
                    errors.append(
                        f"Duplicate SHA-256 within {split}: {image_hash}"
                    )
                else:
                    # The cross-split summary below will also report this.
                    pass
            else:
                all_csv_hashes[image_hash] = (split, raw_path)

        print(f"\nClass counts in {csv_path.name}:")
        for class_name in CLASS_NAMES:
            actual = class_counts[class_name]
            expected = EXPECTED_COUNTS[split][class_name]
            print(
                f"  {class_name}: {actual:,} "
                f"(expected {expected:,})"
            )

            if actual != expected:
                errors.append(
                    f"{csv_path.name}: {class_name} expected {expected}, "
                    f"found {actual}"
                )

        print("Missing paths:", missing_paths)
        print("Wrong split paths:", wrong_split_paths)
        print("Wrong class paths:", wrong_class_paths)
        print("Missing hashes:", missing_hashes)

        if missing_hashes:
            errors.append(
                f"{csv_path.name} contains {missing_hashes} missing hashes"
            )

    print(f"\nTotal rows across split CSVs: {total_csv_rows:,}")
    if total_csv_rows != 120000:
        errors.append(
            f"Expected 120,000 total split CSV rows, found {total_csv_rows}"
        )

    # --------------------------------------------------------
    # CHECK 3: Cross-split overlap
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CHECK 3: CROSS-SPLIT OVERLAP")
    print("=" * 65)

    split_names = ["train", "validation", "test"]

    for index, first_split in enumerate(split_names):
        for second_split in split_names[index + 1:]:
            common_hashes = (
                hashes_by_split[first_split]
                & hashes_by_split[second_split]
            )
            common_paths = (
                paths_by_split[first_split]
                & paths_by_split[second_split]
            )

            print(
                f"{first_split} vs {second_split}: "
                f"{len(common_hashes)} shared hashes, "
                f"{len(common_paths)} shared paths"
            )

            if common_hashes:
                errors.append(
                    f"{len(common_hashes)} shared hashes between "
                    f"{first_split} and {second_split}"
                )

            if common_paths:
                errors.append(
                    f"{len(common_paths)} shared paths between "
                    f"{first_split} and {second_split}"
                )

    # --------------------------------------------------------
    # CHECK 4: Manifest row count and class distribution
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CHECK 4: FINAL MANIFEST")
    print("=" * 65)

    manifest_path = DATASET_DIR / "final_balanced_manifest.csv"

    try:
        manifest_rows, manifest_columns = read_csv_rows(manifest_path)

        if len(manifest_rows) != 120000:
            errors.append(
                f"Manifest expected 120,000 rows, found {len(manifest_rows)}"
            )

        manifest_class_col = find_column(
            manifest_columns, ["class_label", "class", "label"]
        )

        if manifest_class_col is None:
            errors.append("Manifest has no class column")
        else:
            manifest_counts = Counter(
                normalize_class(row.get(manifest_class_col, ""))
                for row in manifest_rows
            )

            print("Manifest class counts:")
            for class_name in CLASS_NAMES:
                actual = manifest_counts[class_name]
                print(f"  {class_name}: {actual:,}")

                if actual != 40000:
                    errors.append(
                        f"Manifest {class_name}: expected 40,000, "
                        f"found {actual}"
                    )

    except (FileNotFoundError, OSError) as exc:
        errors.append(str(exc))

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL RESULT")
    print("=" * 65)

    for warning in warnings:
        print("WARNING:", warning)

    if errors:
        print(f"\nFAILED: {len(errors)} issue(s) found.")
        for number, error in enumerate(errors[:50], start=1):
            print(f"{number}. {error}")

        if len(errors) > 50:
            print(f"... and {len(errors) - 50} additional issue(s).")

        print("\nDo not start full training until these issues are reviewed.")
        raise SystemExit(1)

    print("SUCCESS: all configured checks passed.")
    print("- Folder counts are correct.")
    print("- Split CSV class counts are correct.")
    print("- CSV paths point to existing images in the correct folders.")
    print("- Manifest contains 120,000 records with 40,000 per class.")
    print("- No shared SHA-256 hashes between train, validation, and test.")
    print("- No shared image paths between splits.")
    print("\nThe dataset configuration is ready for the model smoke test.")


if __name__ == "__main__":
    main()
