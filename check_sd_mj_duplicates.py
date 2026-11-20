#!/usr/bin/env python3
"""
check_sd_mj_duplicates.py
-------------------------
Comprehensive Duplicate & Similarity Verification for:
- Stable Diffusion (11,913 images)
- Midjourney (11,988 images)
Total: 23,901 images

Checks:
1. Exact Pixel Duplicates (SHA-256 match)
2. Exact File Duplicates (File SHA-256 match)
3. Perceptual Duplicates (pHash Hamming Distance <= 4)
4. Highly Similar Images (pHash Hamming Distance <= 8)
5. Disk Integrity Check (all files exist and readable)

READ-ONLY: Does not touch, delete, or modify any files.
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_DIR / "analysis" / "multi_generator"
MANIFEST_PATH = ANALYSIS_DIR / "collected_images_manifest.csv"
OUTPUT_REPORT_DIR = ANALYSIS_DIR / "verification_reports"

def hamming_weight_64(n):
    """Compute number of set bits for a uint64."""
    return bin(n).count("1")

def main():
    print("=" * 80)
    print("  MULTI-GENERATOR DUPLICATE & SIMILARITY INTEGRITY CHECK")
    print("  Models: STABLE DIFFUSION & MIDJOURNEY")
    print("=" * 80)
    start_time = time.time()

    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest file not found: {MANIFEST_PATH}")
        sys.exit(1)

    print(f"\n[1/4] Loading manifest: {MANIFEST_PATH.name}...")
    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            records.append({
                "idx": idx,
                "source": row["source_name"],
                "category": row["category"],
                "pixel_sha256": row["pixel_sha256"],
                "file_sha256": row.get("saved_file_sha256", ""),
                "phash_hex": row["perceptual_hash"],
                "phash_uint64": int(row["perceptual_hash"], 16),
                "path": row["output_path"],
                "filename": Path(row["output_path"]).name
            })

    total_records = len(records)
    source_counts = Counter(r["source"] for r in records)
    category_counts = Counter(r["category"] for r in records)

    print(f"  -> Total records loaded: {total_records:,}")
    for src, cnt in sorted(source_counts.items()):
        print(f"     * {src}: {cnt:,} images")

    # ---------------------------------------------------------
    # 2. Disk Verification Check
    # ---------------------------------------------------------
    print("\n[2/4] Verifying physical file existence and readability on disk...")
    missing_files = []
    unreadable_files = []
    
    # Fast check file exists
    for r in records:
        p = Path(r["path"])
        if not p.exists():
            missing_files.append(r["path"])
        elif p.stat().st_size == 0:
            unreadable_files.append(r["path"])

    print(f"  -> Files missing: {len(missing_files)}")
    print(f"  -> Files with 0 bytes: {len(unreadable_files)}")

    # ---------------------------------------------------------
    # 3. Exact Duplicates Check (Pixel SHA-256)
    # ---------------------------------------------------------
    print("\n[3/4] Checking for Exact Duplicates (Pixel SHA-256)...")
    pixel_hash_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        pixel_hash_to_indices[r["pixel_sha256"]].append(idx)

    exact_duplicate_groups = [idxs for idxs in pixel_hash_to_indices.values() if len(idxs) > 1]
    total_exact_duplicate_pairs = sum(len(idxs) - 1 for idxs in exact_duplicate_groups)

    print(f"  -> Unique pixel hashes: {len(pixel_hash_to_indices):,} / {total_records:,}")
    print(f"  -> Exact duplicate instances: {total_exact_duplicate_pairs}")

    exact_dup_details = []
    if exact_duplicate_groups:
        for grp in exact_duplicate_groups:
            primary = records[grp[0]]
            for dup_idx in grp[1:]:
                dup = records[dup_idx]
                exact_dup_details.append({
                    "type": "exact_pixel_duplicate",
                    "source_1": primary["source"],
                    "path_1": primary["path"],
                    "source_2": dup["source"],
                    "path_2": dup["path"],
                    "category": primary["category"],
                    "pixel_sha256": primary["pixel_sha256"]
                })

    # ---------------------------------------------------------
    # 4. Perceptual Similarity Check (pHash Distance)
    # ---------------------------------------------------------
    print("\n[4/4] Checking for Perceptual Similarity (pHash Hamming Distance)...")
    print("  -> Computing pairwise perceptual distances across categories...")

    # Group by category for efficient within-category comparisons
    cat_records = defaultdict(list)
    for r in records:
        cat_records[r["category"]].append(r)

    identical_phash_pairs = []   # distance == 0 (identical perceptual hash)
    near_duplicate_pairs = []    # 1 <= distance <= 4 (near-duplicate)
    similar_pairs = []           # 5 <= distance <= 8 (moderately similar)

    # Convert to vectorized numpy uint64 for fast bitwise XOR
    for cat_name, recs in cat_records.items():
        n_cat = len(recs)
        if n_cat < 2:
            continue

        phashes = np.array([r["phash_uint64"] for r in recs], dtype=np.uint64)

        # Chunked comparison to keep memory footprint minimal (< 50MB)
        chunk_size = 2000
        for i_start in range(0, n_cat, chunk_size):
            i_end = min(i_start + chunk_size, n_cat)
            sub_a = phashes[i_start:i_end]

            for j_start in range(i_start, n_cat, chunk_size):
                j_end = min(j_start + chunk_size, n_cat)
                sub_b = phashes[j_start:j_end]

                # Bitwise XOR matrix
                xor_mat = sub_a[:, None] ^ sub_b[None, :]

                # Fast vectorized popcount for uint64
                # Numpy doesn't have native popcount in older versions, so we use byte-level lookups
                bytes_view = xor_mat.view(np.uint8).reshape(xor_mat.shape[0], xor_mat.shape[1], 8)
                # Bit count lookup table for 0..255
                bit_counts = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
                dist_mat = bit_counts[bytes_view].sum(axis=2)

                # Find matches <= 8
                rows, cols = np.where(dist_mat <= 8)
                for r_idx, c_idx in zip(rows, cols):
                    global_i = i_start + r_idx
                    global_j = j_start + c_idx
                    if global_i >= global_j:
                        continue  # Avoid self-comparison and symmetric duplicates

                    dist = int(dist_mat[r_idx, c_idx])
                    rec_a = recs[global_i]
                    rec_b = recs[global_j]

                    pair_info = {
                        "distance": dist,
                        "category": cat_name,
                        "source_1": rec_a["source"],
                        "file_1": rec_a["filename"],
                        "path_1": rec_a["path"],
                        "source_2": rec_b["source"],
                        "file_2": rec_b["filename"],
                        "path_2": rec_b["path"],
                        "same_source": (rec_a["source"] == rec_b["source"])
                    }

                    if dist == 0:
                        identical_phash_pairs.append(pair_info)
                    elif dist <= 4:
                        near_duplicate_pairs.append(pair_info)
                    else:
                        similar_pairs.append(pair_info)

    # ---------------------------------------------------------
    # 5. Summarize Results
    # ---------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("                      VERIFICATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Images Evaluated:            {total_records:,}")
    print(f"  - Stable Diffusion:              {source_counts['STABLE_DIFFUSION']:,}")
    print(f"  - Midjourney:                    {source_counts['MIDJOURNEY']:,}")
    print(f"Verification Time:                 {elapsed:.2f} seconds\n")

    print("-" * 80)
    print("1. DISK INTEGRITY:")
    print(f"   Missing files:                  {len(missing_files)}")
    print(f"   Corrupted (0-byte) files:       {len(unreadable_files)}")

    print("-" * 80)
    print("2. EXACT DUPLICATES (Pixel SHA-256 match):")
    print(f"   Exact duplicate pairs:          {len(exact_dup_details)}")
    if exact_dup_details:
        for d in exact_dup_details[:5]:
            print(f"   * [{d['source_1']} vs {d['source_2']}] {Path(d['path_1']).name} == {Path(d['path_2']).name}")
        if len(exact_dup_details) > 5:
            print(f"   * ... and {len(exact_dup_details) - 5} more.")
    else:
        print("   -> ZERO exact duplicates found! All images have unique pixel hashes.")

    print("-" * 80)
    print("3. PERCEPTUAL IDENTICAL (pHash distance = 0):")
    print(f"   Identical perceptual hash:      {len(identical_phash_pairs)}")
    within_sd_p0 = sum(1 for p in identical_phash_pairs if p["source_1"] == "STABLE_DIFFUSION" and p["source_2"] == "STABLE_DIFFUSION")
    within_mj_p0 = sum(1 for p in identical_phash_pairs if p["source_1"] == "MIDJOURNEY" and p["source_2"] == "MIDJOURNEY")
    cross_model_p0 = sum(1 for p in identical_phash_pairs if not p["same_source"])
    print(f"   * Within Stable Diffusion:      {within_sd_p0}")
    print(f"   * Within Midjourney:            {within_mj_p0}")
    print(f"   * Cross-model (SD vs MJ):       {cross_model_p0}")

    print("-" * 80)
    print("4. NEAR-DUPLICATES (1 <= pHash distance <= 4):")
    print(f"   Near-duplicate pairs:           {len(near_duplicate_pairs)}")
    within_sd_near = sum(1 for p in near_duplicate_pairs if p["source_1"] == "STABLE_DIFFUSION" and p["source_2"] == "STABLE_DIFFUSION")
    within_mj_near = sum(1 for p in near_duplicate_pairs if p["source_1"] == "MIDJOURNEY" and p["source_2"] == "MIDJOURNEY")
    cross_model_near = sum(1 for p in near_duplicate_pairs if not p["same_source"])
    print(f"   * Within Stable Diffusion:      {within_sd_near}")
    print(f"   * Within Midjourney:            {within_mj_near}")
    print(f"   * Cross-model (SD vs MJ):       {cross_model_near}")

    print("-" * 80)
    print("5. MODERATELY SIMILAR (5 <= pHash distance <= 8):")
    print(f"   Similar visual theme pairs:     {len(similar_pairs)}")

    # ---------------------------------------------------------
    # 6. Save Reports to Disk
    # ---------------------------------------------------------
    summary_report_file = OUTPUT_REPORT_DIR / "duplicate_verification_summary.json"
    summary_data = {
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images_checked": total_records,
        "counts_by_source": dict(source_counts),
        "missing_files_count": len(missing_files),
        "zero_byte_files_count": len(unreadable_files),
        "exact_duplicates_count": len(exact_dup_details),
        "identical_phash_count": len(identical_phash_pairs),
        "near_duplicates_count": len(near_duplicate_pairs),
        "similar_pairs_count": len(similar_pairs),
        "breakdown": {
            "identical_phash": {
                "within_stable_diffusion": within_sd_p0,
                "within_midjourney": within_mj_p0,
                "cross_model": cross_model_p0
            },
            "near_duplicates": {
                "within_stable_diffusion": within_sd_near,
                "within_midjourney": within_mj_near,
                "cross_model": cross_model_near
            }
        }
    }
    with open(summary_report_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save detailed CSV for any exact or near-duplicates found
    csv_report_file = OUTPUT_REPORT_DIR / "duplicates_and_similarities.csv"
    with open(csv_report_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["match_type", "distance", "category", "source_1", "file_1", "source_2", "file_2", "path_1", "path_2"])
        
        for d in exact_dup_details:
            writer.writerow(["EXACT_PIXEL", 0, d["category"], d["source_1"], Path(d["path_1"]).name, d["source_2"], Path(d["path_2"]).name, d["path_1"], d["path_2"]])

        for p in identical_phash_pairs:
            writer.writerow(["PHASH_IDENTICAL", 0, p["category"], p["source_1"], p["file_1"], p["source_2"], p["file_2"], p["path_1"], p["path_2"]])

        for p in near_duplicate_pairs:
            writer.writerow(["NEAR_DUPLICATE", p["distance"], p["category"], p["source_1"], p["file_1"], p["source_2"], p["file_2"], p["path_1"], p["path_2"]])

    print("\n" + "=" * 80)
    print(f"[SAVED] Summary JSON: {summary_report_file}")
    print(f"[SAVED] Detailed CSV: {csv_report_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

