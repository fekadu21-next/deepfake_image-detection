
from pathlib import Path
import shutil

import pandas as pd

DATASET_DIR = Path(r"D:\image-forensics-project\final_dataset")
MANIFEST = DATASET_DIR / "final_balanced_manifest.csv"
SPLIT_DIR = DATASET_DIR / "splits"

SPLITS = ["train", "validation", "test"]
CLASSES = ["REAL", "TAMPERED", "SYNTHETIC"]

# First run: leave True to check everything without moving files.
# After a successful dry run, change this to False.
DRY_RUN = False


def relative_path(path):
    return path.relative_to(DATASET_DIR).as_posix()


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    manifest = pd.read_csv(MANIFEST)

    required = {"class_label", "sha256", "image_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(
            f"Manifest must contain columns: {sorted(required)}"
        )

    if len(manifest) != 120_000:
        raise ValueError(f"Expected 120,000 images; found {len(manifest)}")

    if manifest["sha256"].duplicated().any():
        raise ValueError("Duplicate hashes found. Nothing will be moved.")

    # Load split files and validate all assignments before moving anything.
    split_frames = {}
    all_hashes = set()
    planned_moves = []

    expected_per_class = {
        "train": 32_000,
        "validation": 4_000,
        "test": 4_000,
    }

    for split in SPLITS:
        csv_path = SPLIT_DIR / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run create_dataset_splits.py first."
            )

        df = pd.read_csv(csv_path)

        if not required.issubset(df.columns):
            raise ValueError(f"Required columns missing in {csv_path}")

        expected_total = 96_000 if split == "train" else 12_000
        if len(df) != expected_total:
            raise ValueError(
                f"{split}.csv should have {expected_total} rows; "
                f"found {len(df)}"
            )

        if df["sha256"].duplicated().any():
            raise ValueError(f"Duplicate hashes inside {split}.csv")

        for label in CLASSES:
            count = int((df["class_label"] == label).sum())
            if count != expected_per_class[split]:
                raise ValueError(
                    f"{split}: {label} has {count} rows, "
                    f"expected {expected_per_class[split]}"
                )

        hashes = set(df["sha256"])
        if all_hashes & hashes:
            raise ValueError("Images overlap between different splits.")
        all_hashes.update(hashes)

        split_frames[split] = df

        for row in df.itertuples(index=False):
            label = str(row.class_label)
            if label not in CLASSES:
                raise ValueError(f"Unexpected class: {label}")

            source = DATASET_DIR / str(row.image_path).replace("/", "\\")
            destination = DATASET_DIR / split / label / source.name

            if not source.is_file() and not destination.is_file():
                raise FileNotFoundError(
                    f"Image found in neither source nor destination:\n"
                    f"Source: {source}\nDestination: {destination}"
                )

            if source.is_file() and destination.exists() and source != destination:
                raise FileExistsError(
                    f"Destination already exists; refusing to overwrite:\n"
                    f"{destination}"
                )

            planned_moves.append(
                (split, row.sha256, source, destination)
            )

    if len(all_hashes) != 120_000:
        raise ValueError("The split files do not cover 120,000 unique images.")

    print(f"Preflight passed: {len(planned_moves):,} image assignments.")
    print("Counts and cross-split hash checks passed.")

    if DRY_RUN:
        print("\nDRY RUN ONLY: no images were moved.")
        print("Review the checks above.")
        print("If everything is correct, set DRY_RUN = False and rerun.")
        return

    # Move the images. If interrupted, rerunning can continue where possible.
    completed = 0

    for split, image_hash, source, destination in planned_moves:
        if source == destination:
            if not destination.is_file():
                raise FileNotFoundError(f"Expected image missing: {destination}")
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        elif not destination.is_file():
            raise FileNotFoundError(
                f"Image disappeared during moving: {source}"
            )

        completed += 1
        if completed % 5_000 == 0:
            print(f"Processed {completed:,} / 120,000 images")

    # Update CSV paths to match the new physical locations.
    new_path_by_hash = {
        image_hash: relative_path(destination)
        for _, image_hash, _, destination in planned_moves
    }

    for split, df in split_frames.items():
        df["image_path"] = df["sha256"].map(new_path_by_hash)

        if df["image_path"].isna().any():
            raise ValueError(f"Could not update every path in {split}.csv")

        df.to_csv(SPLIT_DIR / f"{split}.csv", index=False)

    manifest["image_path"] = manifest["sha256"].map(new_path_by_hash)

    if manifest["image_path"].isna().any():
        raise ValueError("Could not update every image path in the manifest.")

    manifest.to_csv(MANIFEST, index=False)

    # Final verification of folder counts and file existence.
    for split in SPLITS:
        for label in CLASSES:
            folder = DATASET_DIR / split / label
            actual_files = sum(1 for p in folder.iterdir() if p.is_file())
            expected = expected_per_class[split]

            if actual_files != expected:
                raise RuntimeError(
                    f"{folder}: found {actual_files} files; expected {expected}"
                )

    print("\nSUCCESS: all 120,000 images have been organized.")
    print("The original images folder has not been manually deleted.")
    print(f"Dataset location: {DATASET_DIR}")


if __name__ == "__main__":
    main()
