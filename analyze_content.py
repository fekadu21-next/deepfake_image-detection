from __future__ import annotations
import csv
import hashlib
import io
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\image-forensics-project")
ARCHIVE_DIR = PROJECT / "raw_datasets" / "SECOUND"
OUTPUT_DIR = PROJECT / "analysis" / "content"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff"
}

CLASS_ORDER = ["real", "tampered", "synthetic"]

# These categories describe image CONTENT, not forensic labels.
CATEGORIES = [
    "people and portraits",
    "animals and wildlife",
    "nature and landscapes",
    "cars, trucks, motorcycles, and other ground vehicles",
    "aircraft and helicopters",
    "buildings, houses, and architecture",
    "everyday objects, tools, furniture, and electronics",
    "food, fruit, and vegetables",
    "other subjects or unclear content",
]

PROMPTS = [
    f"a photograph showing {category}"
    for category in CATEGORIES
]

CSV_FIELDS = [
    "class_label",
    "archive",
    "member",
    "sha256",
    "category",
    "top_score",
    "width",
    "height",
    "status",
]


# ============================================================
# 2. FIND AND VALIDATE THE SOURCE ZIP FILES
# ============================================================

def find_archives():
    if not ARCHIVE_DIR.exists():
        raise FileNotFoundError(
            f"Archive directory does not exist:\n{ARCHIVE_DIR}"
        )
    zip_files = sorted(
        p for p in ARCHIVE_DIR.glob("*.zip")
        if p.is_file()
    )
    print("\nZIP FILES FOUND")
    print("-" * 75)
    for path in zip_files:
        size_gib = path.stat().st_size / (1024 ** 3)
        print(f"{path.name:45} {size_gib:8.2f} GiB")

    real_files = [
        p for p in zip_files
        if "real" in p.name.lower()
        and "mask" not in p.name.lower()
    ]
    tampered_files = [
        p for p in zip_files
        if "tampered" in p.name.lower()
        and "mask" not in p.name.lower()
    ]
    synthetic_files = sorted(
        [
            p for p in zip_files
            if "synthetic" in p.name.lower()
            and "mask" not in p.name.lower()
        ],
        key=lambda p: p.name.lower()
    )
    if len(real_files) != 1:
        raise RuntimeError(
            "Expected exactly one real ZIP. Found: "
            + str([p.name for p in real_files])
        )
    if len(tampered_files) != 1:
        raise RuntimeError(
            "Expected exactly one tampered ZIP. Found: "
            + str([p.name for p in tampered_files])
        )

    if len(synthetic_files) < 2:
        raise RuntimeError(
            "Expected two synthetic ZIP parts. Check that both ZIP "
            "downloads finished and have filenames ending in .zip."
        )

    if len(synthetic_files) > 2:
        print(
            "\nWARNING: More than two synthetic ZIPs found. "
            "All matching ZIPs will be analyzed."
        )

    return {
        "real": [real_files[0]],
        "tampered": [tampered_files[0]],
        "synthetic": synthetic_files,
    }


# ============================================================
# 3. READ IMAGE ENTRIES WITHOUT EXTRACTING THE ZIP
# ============================================================

def get_image_entries(zip_file):
    return [
        info for info in zip_file.infolist()
        if not info.is_dir()
        and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# 4. RESUME SUPPORT
# ============================================================

def load_done_keys(csv_path):
    """
    Read already processed archive entries.

    If the script is interrupted, previously recorded entries
    will be skipped when it runs again.
    """
    done = set()

    if not csv_path.exists():
        return done

    with csv_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        for row in csv.DictReader(file):
            done.add(
                (
                    row["class_label"],
                    row["archive"],
                    row["member"],
                )
            )

    return done


# ============================================================
# 5. WRITE RESULTS IN SMALL BATCHES
# ============================================================

def append_rows(csv_path, rows):
    if not rows:
        return

    needs_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=CSV_FIELDS
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# 6. CLASSIFY THE SUBJECT OF EACH IMAGE WITH CLIP
# ============================================================

def classify_batch(
    class_label,
    archive_name,
    images,
    members,
    error_rows,
    model,
    processor,
    output_csv,
):
    output_rows = list(error_rows)

    if images:
        inputs = processor(
            text=PROMPTS,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        with torch.inference_mode():
            outputs = model(**inputs)

            probabilities = (
                outputs.logits_per_image.softmax(dim=1)
            )

            scores, category_indices = probabilities.max(
                dim=1
            )

        for member_data, score, category_index in zip(
            members,
            scores.tolist(),
            category_indices.tolist(),
        ):
            member, digest, width, height = member_data

            output_rows.append({
                "class_label": class_label,
                "archive": archive_name,
                "member": member,
                "sha256": digest,
                "category": CATEGORIES[category_index],
                "top_score": f"{score:.6f}",
                "width": width,
                "height": height,
                "status": "ok",
            })

    append_rows(output_csv, output_rows)


# ============================================================
# 7. ANALYZE ONE CLASS AND ITS ZIP ARCHIVES
# ============================================================

def analyze_one_class(
    class_label,
    archive_paths,
    model,
    processor,
    output_csv,
):
    done = load_done_keys(output_csv)

    started = time.time()
    newly_classified = 0
    skipped_existing = 0
    invalid_images = 0

    for archive_path in archive_paths:
        print(f"\n{'=' * 70}")
        print(f"CLASS: {class_label.upper()}")
        print(f"ARCHIVE: {archive_path.name}")
        print(f"{'=' * 70}")

        with ZipFile(archive_path, "r") as zip_file:
            entries = get_image_entries(zip_file)

            print(
                f"Image entries in archive: {len(entries):,}"
            )

            images = []
            members = []
            error_rows = []

            for index, info in enumerate(entries, start=1):
                key = (
                    class_label,
                    archive_path.name,
                    info.filename,
                )

                if key in done:
                    skipped_existing += 1
                    continue

                try:
                    raw_bytes = zip_file.read(info)

                    with Image.open(
                        io.BytesIO(raw_bytes)
                    ) as image:
                        image.load()

                        width, height = image.size
                        rgb_image = image.convert("RGB")

                    digest = hashlib.sha256(
                        raw_bytes
                    ).hexdigest()

                    images.append(rgb_image)
                    members.append(
                        (
                            info.filename,
                            digest,
                            width,
                            height,
                        )
                    )

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    EOFError,
                ) as exc:
                    error_rows.append({
                        "class_label": class_label,
                        "archive": archive_path.name,
                        "member": info.filename,
                        "sha256": "",
                        "category": "review_invalid",
                        "top_score": "",
                        "width": "",
                        "height": "",
                        "status": (
                            f"decode_error:"
                            f"{type(exc).__name__}"
                        ),
                    })

                    invalid_images += 1

                # Process a small batch to limit RAM usage.
                if len(images) >= BATCH_SIZE:
                    batch_size = len(images)

                    classify_batch(
                        class_label,
                        archive_path.name,
                        images,
                        members,
                        error_rows,
                        model,
                        processor,
                        output_csv,
                    )

                    newly_classified += batch_size

                    # Mark both successful images and errors as done.
                    for member_data in members:
                        done.add((
                            class_label,
                            archive_path.name,
                            member_data[0],
                        ))

                    for row in error_rows:
                        done.add((
                            class_label,
                            archive_path.name,
                            row["member"],
                        ))

                    images = []
                    members = []
                    error_rows = []

                # Report progress every 5,000 archive entries.
                if index % 5000 == 0:
                    print(
                        f"Scanned {index:,}/"
                        f"{len(entries):,} entries | "
                        f"New valid images: "
                        f"{newly_classified:,}"
                    )

            # Save the final partial batch.
            if images or error_rows:
                batch_size = len(images)

                classify_batch(
                    class_label,
                    archive_path.name,
                    images,
                    members,
                    error_rows,
                    model,
                    processor,
                    output_csv,
                )

                newly_classified += batch_size

                for member_data in members:
                    done.add((
                        class_label,
                        archive_path.name,
                        member_data[0],
                    ))

                for row in error_rows:
                    done.add((
                        class_label,
                        archive_path.name,
                        row["member"],
                    ))

        print(
            f"\nFinished archive: {archive_path.name}"
        )

    elapsed_minutes = (time.time() - started) / 60

    print(f"\nFinished class: {class_label}")
    print(f"New valid images classified: {newly_classified:,}")
    print(f"Previously recorded entries skipped: {skipped_existing:,}")
    print(f"Images with decoding errors: {invalid_images:,}")
    print(f"Elapsed time: {elapsed_minutes:.1f} minutes")


# ============================================================
# 8. CREATE CATEGORY COUNTS AND DUPLICATE REPORTS
# ============================================================

def write_summaries(output_csv):
    if not output_csv.exists():
        print("No analysis CSV exists yet.")
        return

    with output_csv.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    valid_rows = [
        row for row in rows
        if row["status"] == "ok"
    ]

    counts = Counter(
        (row["class_label"], row["category"])
        for row in valid_rows
    )

    # Save counts for each subject category and forensic class.
    counts_path = (
        OUTPUT_DIR / "content_category_counts.csv"
    )

    with counts_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "class_label",
            "category",
            "count",
        ])
        for class_label in CLASS_ORDER:
            for category in CATEGORIES:
                writer.writerow([
                    class_label,
                    category,
                    counts[(class_label, category)],
                ])
    # Find exact byte-identical image files.
    hash_groups = defaultdict(list)

    for row in valid_rows:
        if row["sha256"]:
            hash_groups[row["sha256"]].append(row)

    duplicate_rows = []

    for digest, group in hash_groups.items():
        if len(group) > 1:
            for row in group:
                duplicate_rows.append({
                    "sha256": digest,
                    "class_label": row["class_label"],
                    "archive": row["archive"],
                    "member": row["member"],
                    "category": row["category"],
                })

    duplicates_path = (
        OUTPUT_DIR / "exact_duplicate_review.csv"
    )

    with duplicates_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "sha256",
            "class_label",
            "archive",
            "member",
            "category",
        ]

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(duplicate_rows)

    # Save a simple image-dimension and status report.
    status_counts = Counter(
        (
            row["class_label"],
            row["status"],
        )
        for row in rows
    )

    status_path = OUTPUT_DIR / "image_status_counts.csv"

    with status_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "class_label",
            "status",
            "count",
        ])

        for class_label in CLASS_ORDER:
            statuses = sorted({
                row["status"]
                for row in rows
                if row["class_label"] == class_label
            })

            for status in statuses:
                writer.writerow([
                    class_label,
                    status,
                    status_counts[(class_label, status)],
                ])

    print("\n" + "=" * 90)
    print("CONTENT CATEGORY COUNTS")
    print("=" * 90)

    print(
        f"{'Category':48}"
        f"{'Real':>10}"
        f"{'Tampered':>12}"
        f"{'Synthetic':>12}"
    )

    print("-" * 90)

    for category in CATEGORIES:
        print(
            f"{category:48}"
            f"{counts[('real', category)]:10,}"
            f"{counts[('tampered', category)]:12,}"
            f"{counts[('synthetic', category)]:12,}"
        )

    print("\nANALYSIS SUMMARY")
    print("-" * 50)
    print(f"Total CSV rows: {len(rows):,}")
    print(f"Valid images: {len(valid_rows):,}")
    print(f"Images needing decode review: {len(rows) - len(valid_rows):,}")
    print(f"Rows involved in exact duplicate groups: {len(duplicate_rows):,}")

    print("\nOUTPUT FILES")
    print(f"Detailed analysis: {output_csv}")
    print(f"Category counts:  {counts_path}")
    print(f"Duplicate review: {duplicates_path}")
    print(f"Image statuses:   {status_path}")

    print("\nIMPORTANT:")
    print("- Categories are estimates made by CLIP, not verified labels.")
    print("- top_score is a relative score, not a calibrated probability.")
    print("- Exact hashes find identical file bytes, not visually similar images.")
    print("- Inspect samples before selecting the final training dataset.")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archives = find_archives()

    output_csv = (
        OUTPUT_DIR / "image_content_analysis.csv"
    )

    print("\nThis program:")
    print("1. Reads images directly from ZIP archives.")
    print("2. Does not extract all images to disk.")
    print("3. Estimates each image's subject category using CLIP.")
    print("4. Records image dimensions and decoding errors.")
    print("5. Reports exact duplicates.")
    print("6. Saves progress in batches.")

    print(
        "\nIMPORTANT: Your CPU-only computer may take many hours "
        "to process all images."
    )

    input(
        "\nPress Enter to begin, or Ctrl+C to cancel..."
    )

    torch.set_num_threads(
        max(1, min(6, torch.get_num_threads()))
    )

    print(f"\nLoading model: {MODEL_NAME}")
    print("The first run requires internet to download the model.")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to("cpu")

    for class_label in CLASS_ORDER:
        analyze_one_class(
            class_label,
            archives[class_label],
            model,
            processor,
            output_csv,
        )

    write_summaries(output_csv)

    print("\nALL ANALYSIS STEPS FINISHED.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nStopped by user. Completed batches were saved. "
            "Run the script again to resume."
        )
        sys.exit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
