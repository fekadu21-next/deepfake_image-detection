#!/usr/bin/env python3
"""
=====================================================================================
            GRID PORTRAIT EXTRACTION & SEPARATION TOOL
=====================================================================================
Extracts and separates individual portraits from composite grid images generated
by Stable Diffusion. Detects divider line coordinates, trims border line artifacts,
saves pristine individual portraits, checks for duplicates via SHA-256, and produces
an extraction manifest.
=====================================================================================
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Default configuration paths
DEFAULT_SRC_DIR = r"D:\image-forensics-project\raw_datasets\newgenimages\stabledefussion\peoples and portraits"
DEFAULT_OUT_DIR = r"D:\image-forensics-project\raw_datasets\newgenimages\stabledefussion\peoples_and_portraits_extracted"

# Verified layout configurations for the 6 grid composites
# (Enables precise boundary verification and fallback if dynamic line detection differs)
GRID_SPECS = {
    1: {
        "filename": "stabdefumepeopl1.png",
        "description": "Grid 1 (5x5, 25 portraits)",
        "rows": 5, "cols": 5,
        "y_cuts": [222, 433, 629, 824],
        "x_cuts": [206, 409, 613, 816],
    },
    2: {
        "filename": "stabdefumepoepl2.png",
        "description": "Grid 2 (4x3, 12 portraits)",
        "rows": 4, "cols": 3,
        "y_cuts": [256, 513, 766],
        "x_cuts": [343, 680],
    },
    3: {
        "filename": "stabdefumepoepl3.png",
        "description": "Grid 3 (5x6, 30 portraits)",
        "rows": 5, "cols": 6,
        "y_cuts": [206, 403, 601, 805],
        "x_cuts": [174, 344, 510, 681, 853],
    },
    4: {
        "filename": "stabdefumepoepl4.png",
        "description": "Grid 4 (Staggered 3-Row, 10 portraits)",
        "rows": 3,
        "y_cuts": [353, 692],
        "per_row_x_cuts": {
            0: [312, 573, 801],       # Row 0: 4 portraits
            1: [312, 655],            # Row 1: 3 portraits
            2: [359, 655],            # Row 2: 3 portraits
        }
    },
    5: {
        "filename": "stabdefumepoepl5.png",
        "description": "Grid 5 (4x5, 20 portraits)",
        "rows": 4, "cols": 5,
        "y_cuts": [253, 511, 765],
        "x_cuts": [207, 404, 603, 798],
    },
    6: {
        "filename": "stabdefumepoepl6.png",
        "description": "Grid 6 (5x4, 20 portraits)",
        "rows": 5, "cols": 4,
        "y_cuts": [224, 440, 644, 838],
        "x_cuts": [254, 510, 761],
    },
}


def calculate_sha256(filepath: str) -> str:
    """Calculates SHA-256 hash of a file for duplicate detection."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def detect_line_peaks(profile: np.ndarray, expected_count: int, min_dist: int = 40) -> list:
    """
    Finds sharp divider peaks in a 1D Sobel gradient projection.
    """
    threshold = np.mean(profile) + 1.2 * np.std(profile)
    candidates = []
    
    for i in range(15, len(profile) - 15):
        if profile[i] > threshold and profile[i] == max(profile[max(0, i-4):min(len(profile), i+5)]):
            candidates.append((i, profile[i]))
            
    # Sort candidates by gradient peak strength
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Filter with non-maximum suppression / minimum distance
    selected = []
    for idx, val in candidates:
        if all(abs(idx - s) >= min_dist for s in selected):
            selected.append(idx)
        if len(selected) == expected_count:
            break
            
    selected.sort()
    return selected


def get_cut_intervals(length: int, cuts: list, trim_margin: int = 3) -> list:
    """
    Converts cut coordinates into (start, end) intervals with border trimming.
    
    Args:
        length: Total dimension length (e.g. 1024)
        cuts: List of internal cut line center coordinates
        trim_margin: Number of pixels to trim inward to remove divider borders
    
    Returns:
        List of (start, end) pixel tuples for cropping
    """
    boundaries = [0] + sorted(cuts) + [length]
    intervals = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Apply inward margin trimming on internal boundaries
        c_start = start + (trim_margin if start > 0 else 0)
        c_end = end - (trim_margin if end < length else 0)
        
        if c_end > c_start:
            intervals.append((c_start, c_end))
    return intervals


def extract_portraits(src_dir: str, out_dir: str, trim_margin: int = 3) -> list:
    """
    Performs boundary detection and extracts portraits from all 6 grids.
    """
    os.makedirs(out_dir, exist_ok=True)
    extracted_records = []
    hash_set = set()
    duplicate_count = 0

    print("=" * 85)
    print("        STABLE DIFFUSION GRID PORTRAIT EXTRACTION TOOL")
    print("=" * 85)
    print(f"Source Directory:      {src_dir}")
    print(f"Destination Directory: {out_dir}")
    print(f"Border Trim Margin:    {trim_margin} px (inward from line centers)")
    print("-" * 85)

    summary_rows = []

    for grid_idx in sorted(GRID_SPECS.keys()):
        spec = GRID_SPECS[grid_idx]
        filename = spec["filename"]
        img_path = os.path.join(src_dir, filename)

        if not os.path.exists(img_path):
            print(f"[!] Warning: File not found: {filename}")
            continue

        # Load image via OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[!] Error: Could not read image {filename}")
            continue

        h, w, _ = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Dynamic verification of horizontal cuts
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
        expected_y_cuts = len(spec["y_cuts"])
        detected_y_cuts = detect_line_peaks(sobel_y, expected_y_cuts, min_dist=80)
        
        # If detected closely matches spec, use refined peaks, else fallback to spec
        if len(detected_y_cuts) == expected_y_cuts and all(abs(d - s) <= 8 for d, s in zip(detected_y_cuts, spec["y_cuts"])):
            y_cuts = detected_y_cuts
        else:
            y_cuts = spec["y_cuts"]

        y_intervals = get_cut_intervals(h, y_cuts, trim_margin=trim_margin)
        grid_extracted = 0
        cell_resolutions = []

        # Case A: Staggered rows (Grid 4)
        if "per_row_x_cuts" in spec:
            for r_idx, (y1, y2) in enumerate(y_intervals):
                row_cuts = spec["per_row_x_cuts"].get(r_idx, [])
                x_intervals = get_cut_intervals(w, row_cuts, trim_margin=trim_margin)
                
                for c_idx, (x1, x2) in enumerate(x_intervals):
                    crop = img_bgr[y1:y2, x1:x2]
                    out_filename = f"sd_portrait_grid{grid_idx}_r{r_idx+1:02d}_c{c_idx+1:02d}.png"
                    out_path = os.path.join(out_dir, out_filename)
                    cv2.imwrite(out_path, crop)

                    # Compute SHA-256
                    file_hash = calculate_sha256(out_path)
                    if file_hash in hash_set:
                        duplicate_count += 1
                        print(f"  [!] DUPLICATE DETECTED: {out_filename}")
                    hash_set.add(file_hash)

                    grid_extracted += 1
                    cell_resolutions.append(f"{x2-x1}x{y2-y1}")
                    extracted_records.append({
                        "filename": out_filename,
                        "source_grid": filename,
                        "grid_id": grid_idx,
                        "row": r_idx + 1,
                        "col": c_idx + 1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "sha256": file_hash,
                    })

        # Case B: Uniform 2D Grids (Grids 1, 2, 3, 5, 6)
        else:
            sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)).mean(axis=0)
            expected_x_cuts = len(spec["x_cuts"])
            detected_x_cuts = detect_line_peaks(sobel_x, expected_x_cuts, min_dist=80)

            if len(detected_x_cuts) == expected_x_cuts and all(abs(d - s) <= 8 for d, s in zip(detected_x_cuts, spec["x_cuts"])):
                x_cuts = detected_x_cuts
            else:
                x_cuts = spec["x_cuts"]

            x_intervals = get_cut_intervals(w, x_cuts, trim_margin=trim_margin)

            for r_idx, (y1, y2) in enumerate(y_intervals):
                for c_idx, (x1, x2) in enumerate(x_intervals):
                    crop = img_bgr[y1:y2, x1:x2]
                    out_filename = f"sd_portrait_grid{grid_idx}_r{r_idx+1:02d}_c{c_idx+1:02d}.png"
                    out_path = os.path.join(out_dir, out_filename)
                    cv2.imwrite(out_path, crop)

                    file_hash = calculate_sha256(out_path)
                    if file_hash in hash_set:
                        duplicate_count += 1
                        print(f"  [!] DUPLICATE DETECTED: {out_filename}")
                    hash_set.add(file_hash)

                    grid_extracted += 1
                    cell_resolutions.append(f"{x2-x1}x{y2-y1}")
                    extracted_records.append({
                        "filename": out_filename,
                        "source_grid": filename,
                        "grid_id": grid_idx,
                        "row": r_idx + 1,
                        "col": c_idx + 1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "sha256": file_hash,
                    })

        res_sample = cell_resolutions[0] if cell_resolutions else "N/A"
        summary_rows.append({
            "grid": f"Grid {grid_idx}",
            "file": filename,
            "layout": f"{len(y_intervals)} rows x {len(spec.get('x_cuts', []))+1 if 'x_cuts' in spec else 'var'} cols",
            "extracted": grid_extracted,
            "res_approx": f"~{res_sample} px"
        })
        print(f"  [+] {spec['description']:<38} -> Extracted {grid_extracted:>2} portraits")

    # Save manifest CSV
    manifest_path = os.path.join(out_dir, "extraction_manifest.csv")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("filename,source_grid,grid_id,row,col,width,height,sha256\n")
        for rec in extracted_records:
            f.write(f"{rec['filename']},{rec['source_grid']},{rec['grid_id']},{rec['row']},{rec['col']},{rec['width']},{rec['height']},{rec['sha256']}\n")

    print("-" * 85)
    print("EXTRACTION SUMMARY TABLE:")
    print("-" * 85)
    print(f"{'Source Image':<25} {'Layout Detected':<22} {'Extracted':<12} {'Resolution Approx':<15}")
    print("-" * 85)
    total_extracted = 0
    for r in summary_rows:
        print(f"{r['file']:<25} {r['layout']:<22} {r['extracted']:<12} {r['res_approx']:<15}")
        total_extracted += r["extracted"]
    print("-" * 85)
    print(f"TOTAL EXTRACTED PORTRAITS: {total_extracted} images")
    print(f"TOTAL EXACT DUPLICATES:    {duplicate_count}")
    print(f"MANIFEST SAVED TO:         {manifest_path}")
    print("=" * 85)

    return extracted_records


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual portraits from composite grid images."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SRC_DIR,
        help="Source directory containing the 6 grid PNG images."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT_DIR,
        help="Destination directory to save extracted individual portraits."
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=3,
        help="Pixel margin trimmed inward from divider line centers (default: 3px)."
    )

    args = parser.parse_args()
    extract_portraits(src_dir=args.source, out_dir=args.output, trim_margin=args.margin)


if __name__ == "__main__":
    main()
