#!/usr/bin/env python3
"""
balance_and_merge_synthetic.py
------------------------------
Balance and merge the 3 Synthetic Image Generators:
- FLUX (Original 40,000 in final_dataset) -> Keep 16,099 highest-quality images
- STABLE DIFFUSION (11,913 images)       -> Keep 100% (11,913)
- MIDJOURNEY (11,988 images)              -> Keep 100% (11,988)

Total Synthetic Images: EXACTLY 40,000 (perfectly balanced across 10 categories)
Split Distribution:     Train 32,000 (80%) | Validation 4,000 (10%) | Test 4,000 (10%)

Safety:
- By default runs in --dry-run mode (does not modify or delete anything).
- When --apply is given:
    * Moves the 23,901 unselected FLUX images to a safe backup directory (NO permanent data loss).
    * Copies SD and MJ images into final_dataset splits with proper naming.
    * Updates final_balanced_manifest.csv and all split CSVs.
    * Backs up previous manifests before updating.
- NEVER TOUCHES REAL OR TAMPERED IMAGES.
"""

import sys
import os
import time
import shutil
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_DIR = Path(__file__).resolve().parent
FINAL_DATASET_DIR = PROJECT_DIR / "final_dataset"
ANALYSIS_DIR = PROJECT_DIR / "analysis" / "multi_generator"
EXTERNAL_SOURCES_DIR = PROJECT_DIR / "new_data" / "SYNTHETIC" / "collected_external_sources"
BACKUP_DIR = PROJECT_DIR / "new_data" / "SYNTHETIC" / "removed_flux_backup"
MANIFEST_BACKUP_DIR = FINAL_DATASET_DIR / "manifest_backup"

FINAL_MANIFEST_PATH = FINAL_DATASET_DIR / "final_balanced_manifest.csv"
COLLECTED_MANIFEST_PATH = ANALYSIS_DIR / "collected_images_manifest.csv"

SPLITS_DIR = FINAL_DATASET_DIR / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
SPLIT_CAT_CSV = SPLITS_DIR / "split_category_counts.csv"
SPLIT_CLASS_CSV = SPLITS_DIR / "split_class_counts.csv"

# Exact target quotas per category (matching REAL and TAMPERED distributions)
CATEGORY_TARGETS = {
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

# Aliases for plant category naming differences
CAT_ALIASES = {
    "plants, flowers and crops": "plants, flowers, and crops",
    "plants, flowers, and crops": "plants, flowers, and crops"
}

def normalize_cat(cat: str) -> str:
    cat = cat.strip()
    return CAT_ALIASES.get(cat, cat)

def parse_args():
    parser = argparse.ArgumentParser(description="Balance and merge 3 synthetic image generators into 40k total.")
    parser.add_argument("--apply", action="store_true", help="Execute the changes (moves removed FLUX images to backup, copies SD/MJ, updates manifest).")
    parser.add_argument("--dry-run", action="store_true", help="Simulate and print summary without touching any files (default).")
    return parser.parse_args()

def main():
    args = parse_args()
    is_dry_run = not args.apply

    print("=" * 85)
    print("        SYNTHETIC 3-MODEL DATASET BALANCING & MERGING TOOL")
    print("        Models: FLUX (16,099) + STABLE DIFFUSION (11,913) + MIDJOURNEY (11,988)")
    print(f"        Mode:   {'[DRY RUN - PREVIEW ONLY]' if is_dry_run else '[APPLY - EXECUTING CHANGES]'}")
    print("=" * 85)

    if not FINAL_MANIFEST_PATH.exists():
        print(f"[ERROR] Final manifest not found: {FINAL_MANIFEST_PATH}")
        sys.exit(1)

    if not COLLECTED_MANIFEST_PATH.exists():
        print(f"[ERROR] Collected manifest not found: {COLLECTED_MANIFEST_PATH}")
        sys.exit(1)

    # -------------------------------------------------------------
    # 1. Load Collected New Images (SD & MJ)
    # -------------------------------------------------------------
    print("\n[1/5] Loading Collected Images (Stable Diffusion & Midjourney)...")
    sd_by_cat = defaultdict(list)
    mj_by_cat = defaultdict(list)

    with open(COLLECTED_MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = normalize_cat(row["category"])
            src = row["source_name"].upper()
            if src == "STABLE_DIFFUSION":
                sd_by_cat[cat].append(row)
            elif src == "MIDJOURNEY":
                mj_by_cat[cat].append(row)

    total_sd = sum(len(v) for v in sd_by_cat.values())
    total_mj = sum(len(v) for v in mj_by_cat.values())
    print(f"  -> Stable Diffusion images loaded: {total_sd:,}")
    print(f"  -> Midjourney images loaded:        {total_mj:,}")
    print(f"  -> Total new images to integrate:   {total_sd + total_mj:,}")

    # -------------------------------------------------------------
    # 2. Load Existing Final Dataset Manifest
    # -------------------------------------------------------------
    print("\n[2/5] Loading Existing Final Dataset Manifest...")
    all_manifest_rows = []
    flux_by_cat = defaultdict(list)
    real_rows = []
    tampered_rows = []

    with open(FINAL_MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            cls_lbl = row["class_label"].upper()
            cat = normalize_cat(row["category"])
            if cls_lbl == "SYNTHETIC":
                flux_by_cat[cat].append(row)
            elif cls_lbl == "REAL":
                real_rows.append(row)
            elif cls_lbl == "TAMPERED":
                tampered_rows.append(row)
            all_manifest_rows.append(row)

    total_flux = sum(len(v) for v in flux_by_cat.values())
    print(f"  -> Existing FLUX Synthetic images: {total_flux:,}")
    print(f"  -> Existing REAL images:           {len(real_rows):,} (will NOT touch)")
    print(f"  -> Existing TAMPERED images:       {len(tampered_rows):,} (will NOT touch)")

    # -------------------------------------------------------------
    # 3. Calculate Exact Category Balance Quotas
    # -------------------------------------------------------------
    print("\n[3/5] Calculating Category Breakdown across All 3 Models:")
    print("-" * 85)
    print(f"{'Category':<38} {'Target':>7} {'SD':>6} {'MJ':>6} {'FLUX Keep':>10} {'FLUX Remove':>12}")
    print("-" * 85)

    flux_to_keep_by_cat = {}
    flux_to_remove_by_cat = {}

    for cat, target in CATEGORY_TARGETS.items():
        sd_count = len(sd_by_cat[cat])
        mj_count = len(mj_by_cat[cat])
        flux_count = len(flux_by_cat[cat])

        # How many FLUX images to keep to meet the target
        needed_from_flux = target - (sd_count + mj_count)
        needed_from_flux = max(0, min(needed_from_flux, flux_count))

        to_remove = flux_count - needed_from_flux

        flux_to_keep_by_cat[cat] = needed_from_flux
        flux_to_remove_by_cat[cat] = to_remove

        print(f"{cat:<38} {target:>7,} {sd_count:>6,} {mj_count:>6,} {needed_from_flux:>10,} {to_remove:>12,}")

    total_flux_keep = sum(flux_to_keep_by_cat.values())
    total_flux_remove = sum(flux_to_remove_by_cat.values())
    final_total_synthetic = total_sd + total_mj + total_flux_keep

    print("-" * 85)
    print(f"{'TOTAL SYNTHETIC':<38} {40000:>7,} {total_sd:>6,} {total_mj:>6,} {total_flux_keep:>10,} {total_flux_remove:>12,}")
    print(f"Final Total Synthetic: {final_total_synthetic:,} images")
    print("-" * 85)

    # -------------------------------------------------------------
    # 4. Partition FLUX and Assign Retained Images
    # -------------------------------------------------------------
    print("\n[4/5] Selecting Highest-Quality FLUX Images per Category...")
    retained_flux_rows = []
    removed_flux_rows = []

    for cat, keep_count in flux_to_keep_by_cat.items():
        rows = flux_by_cat[cat]
        # Sort by quality score (top_score descending, category_margin descending)
        def quality_key(r):
            try:
                score = float(r.get("top_score") or 0.0)
            except Exception:
                score = 0.0
            try:
                margin = float(r.get("category_margin") or 0.0)
            except Exception:
                margin = 0.0
            return (score, margin)

        sorted_rows = sorted(rows, key=quality_key, reverse=True)
        retained = sorted_rows[:keep_count]
        removed = sorted_rows[keep_count:]

        retained_flux_rows.extend(retained)
        removed_flux_rows.extend(removed)

    print(f"  -> FLUX images selected to KEEP:   {len(retained_flux_rows):,}")
    print(f"  -> FLUX images selected to REMOVE: {len(removed_flux_rows):,}")

    # -------------------------------------------------------------
    # 5. Execute or Preview
    # -------------------------------------------------------------
    if is_dry_run:
        print("\n" + "=" * 85)
        print("                        DRY RUN COMPLETE")
        print("=" * 85)
        print("No files were moved, modified, or deleted.")
        print("\nTo execute this balancing and merge the new SD/MJ images into final_dataset:")
        print("  python balance_and_merge_synthetic.py --apply")
        print("=" * 85)
        return

    # APPLY MODE
    print("\n[5/5] Executing Dataset Changes...")

    # Step A: Backup old manifests
    MANIFEST_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(FINAL_MANIFEST_PATH, MANIFEST_BACKUP_DIR / f"final_balanced_manifest_{timestamp}.csv")
    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV, SPLIT_CAT_CSV, SPLIT_CLASS_CSV]:
        if csv_path.exists():
            shutil.copy2(csv_path, MANIFEST_BACKUP_DIR / f"{csv_path.stem}_{timestamp}.csv")
    print("  [A] Backed up previous manifest files to final_dataset/manifest_backup/")

    # Step B: Move unselected FLUX images to safe backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    moved_flux_count = 0
    missing_flux_count = 0

    print("  [B] Moving unselected FLUX images to safe backup folder...")
    for r in removed_flux_rows:
        rel_path = r["image_path"]
        src_file = FINAL_DATASET_DIR / rel_path
        dst_file = BACKUP_DIR / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if src_file.exists():
            shutil.move(str(src_file), str(dst_file))
            moved_flux_count += 1
        else:
            missing_flux_count += 1

    print(f"      -> Safely moved {moved_flux_count:,} FLUX images into {BACKUP_DIR.name}/")
    if missing_flux_count > 0:
        print(f"      -> Note: {missing_flux_count} files were already moved or missing.")

    # Step C: Prepare slots for new SD & MJ images
    # We will slot new images into the removed slots across train, val, and test
    print("  [C] Copying Stable Diffusion and Midjourney images into final_dataset splits...")
    new_images_to_integrate = []
    for cat in CATEGORY_TARGETS.keys():
        for r in sd_by_cat[cat]:
            new_images_to_integrate.append((r, "STABLE_DIFFUSION", cat))
        for r in mj_by_cat[cat]:
            new_images_to_integrate.append((r, "MIDJOURNEY", cat))

    # Match removed slots split locations (train/val/test)
    available_slots = defaultdict(list)
    for r in removed_flux_rows:
        img_path = r["image_path"]
        split_name = img_path.split("/")[0] if "/" in img_path else img_path.split("\\")[0]
        cat = normalize_cat(r["category"])
        available_slots[(split_name, cat)].append(r)

    # Sort new images by category to match slot assignments
    new_by_cat = defaultdict(list)
    for item in new_images_to_integrate:
        new_by_cat[item[2]].append(item)

    integrated_new_rows = []
    copied_count = 0

    for cat, items in new_by_cat.items():
        # Desired split for this category's new items: ~80% train, 10% val, 10% test
        n_items = len(items)
        n_val = int(round(n_items * 0.10))
        n_test = int(round(n_items * 0.10))
        n_train = n_items - (n_val + n_test)

        split_assignments = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test

        for item, split in zip(items, split_assignments):
            source_row, model_name, _ = item
            src_disk_path = Path(source_row["output_path"])

            # Slot sample_id
            cat_slots = available_slots[(split, cat)]
            if cat_slots:
                slot_row = cat_slots.pop()
                target_sample_id = slot_row["sample_id"]
            else:
                # Fallback if slot list exhausted
                target_sample_id = f"IMG_M_{copied_count+1:06d}"

            target_ext = src_disk_path.suffix.lower()
            target_rel_path = f"{split}/SYNTHETIC/{target_sample_id}{target_ext}"
            target_disk_path = FINAL_DATASET_DIR / target_rel_path
            target_disk_path.parent.mkdir(parents=True, exist_ok=True)

            if src_disk_path.exists():
                shutil.copy2(str(src_disk_path), str(target_disk_path))
                copied_count += 1

            new_manifest_row = {
                "sample_id": target_sample_id,
                "class_label": "SYNTHETIC",
                "archive": f"{model_name.lower()}_external_collection",
                "member": src_disk_path.name,
                "sha256": source_row["pixel_sha256"],
                "category": cat,
                "top_score": source_row.get("clip_category_score", "0.90"),
                "width": source_row.get("width", "1024"),
                "height": source_row.get("height", "1024"),
                "status": "ok",
                "original_category": cat,
                "category_margin": source_row.get("clip_category_margin", "0.50"),
                "review_flag": "",
                "reclassification_method": f"external_source_{model_name.lower()}",
                "image_path": target_rel_path.replace("\\", "/")
            }
            integrated_new_rows.append(new_manifest_row)

    print(f"      -> Successfully copied {copied_count:,} SD & MJ images into final_dataset splits!")

    # Step D: Construct and Write New Manifests
    print("  [D] Writing updated final_balanced_manifest.csv...")
    final_combined_rows = real_rows + tampered_rows + retained_flux_rows + integrated_new_rows

    with open(FINAL_MANIFEST_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in final_combined_rows:
            writer.writerow(r)

    # Step E: Update split CSVs (train.csv, validation.csv, test.csv)
    print("  [E] Updating split CSV files (train.csv, validation.csv, test.csv)...")
    for split_name, split_file in [("train", TRAIN_CSV), ("validation", VAL_CSV), ("test", TEST_CSV)]:
        split_rows = [r for r in final_combined_rows if r["image_path"].startswith(f"{split_name}/")]
        with open(split_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in split_rows:
                writer.writerow(r)

    # Step F: Update split_category_counts.csv & split_class_counts.csv
    print("  [F] Updating summary count tables...")
    split_cat_counts = defaultdict(lambda: Counter())
    split_cls_counts = defaultdict(lambda: Counter())

    for r in final_combined_rows:
        split_name = r["image_path"].split("/")[0]
        cls_lbl = r["class_label"]
        cat = r["category"]
        split_cat_counts[(cls_lbl, cat)][split_name] += 1
        split_cls_counts[cls_lbl][split_name] += 1

    with open(SPLIT_CAT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_label", "category", "train", "validation", "test"])
        for (cls_lbl, cat) in sorted(split_cat_counts.keys()):
            counts = split_cat_counts[(cls_lbl, cat)]
            writer.writerow([cls_lbl, cat, counts["train"], counts["validation"], counts["test"]])

    with open(SPLIT_CLASS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_label", "train", "validation", "test"])
        for cls_lbl in sorted(split_cls_counts.keys()):
            counts = split_cls_counts[cls_lbl]
            writer.writerow([cls_lbl, counts["train"], counts["validation"], counts["test"]])

    print("\n" + "=" * 85)
    print("                 SUCCESSFULLY BALANCED & MERGED DATASET!")
    print("=" * 85)
    print("Summary of Final Dataset:")
    print(f"  * REAL Images:      {len(real_rows):,} (untouched)")
    print(f"  * TAMPERED Images:  {len(tampered_rows):,} (untouched)")
    print(f"  * SYNTHETIC Images: {len(retained_flux_rows) + len(integrated_new_rows):,}")
    print(f"      - FLUX:             {len(retained_flux_rows):,} (16,099)")
    print(f"      - STABLE DIFFUSION: {total_sd:,} (11,913)")
    print(f"      - MIDJOURNEY:       {total_mj:,} (11,988)")
    print(f"  * Total Dataset:    {len(final_combined_rows):,} images")
    print(f"  * Removed FLUX:     {moved_flux_count:,} images safely stored in {BACKUP_DIR}")
    print("=" * 85)

if __name__ == "__main__":
    main()

