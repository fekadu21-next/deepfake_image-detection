# Deepfake Image Detection & Forensics Dataset Pipeline

A production-grade, highly balanced computer vision dataset and preprocessing pipeline engineered for robust deepfake and synthetic image detection.

---

## 1. Project & Dataset Overview

This project provides an end-to-end forensic pipeline to curate, deduplicate, categorize, balance, and partition a 120,000-image forensic dataset:

- **Total Dataset Size:** 120,000 images
- **Class Distribution:** Perfectly balanced 3-class distribution (40,000 images per class):
  1. **REAL (40,000):** Pristine camera captures from SID_Set and diverse photography archives.
  2. **TAMPERED (40,000):** Spliced, copy-move, and digitally manipulated images from IMD2020.
  3. **SYNTHETIC (40,000):** Multi-generator AI images covering 3 state-of-the-art models:
     - **FLUX.1:** 16,099 images
     - **Midjourney v6:** 11,988 images
     - **Stable Diffusion (SD 1.5 / SDXL):** 11,913 images

---

## 2. 10 Balanced Content Categories

All 3 classes share identical content distributions across 10 semantic categories to eliminate semantic bias:

| Category | Real | Tampered | Synthetic (Flux + SD + MJ) | Total |
| :--- | :---: | :---: | :---: | :---: |
| people and portraits | 13,468 | 13,468 | 13,468 | 40,404 |
| animals and wildlife | 5,324 | 5,324 | 5,324 | 15,972 |
| everyday objects and technology | 5,178 | 5,178 | 5,178 | 15,534 |
| ground vehicles | 5,017 | 5,017 | 5,017 | 15,051 |
| crowds, meetings, and public gatherings | 2,927 | 2,927 | 2,927 | 8,781 |
| nature and landscapes | 2,767 | 2,767 | 2,767 | 8,301 |
| food and prepared meals | 2,486 | 2,486 | 2,486 | 7,458 |
| buildings and urban scenes | 1,694 | 1,694 | 1,694 | 5,082 |
| aircraft and helicopters | 910 | 910 | 910 | 2,730 |
| plants, flowers, and crops | 229 | 229 | 229 | 687 |
| **Total** | **40,000** | **40,000** | **40,000** | **120,000** |

---

## 3. Dataset Splits (80 / 10 / 10)

The dataset is partitioned with stratified category quotas and exact 0 duplicate leakage across splits:

- **Train (80%):** 96,000 images (32k Real, 32k Tampered, 32k Synthetic)
- **Validation (10%):** 12,000 images (4k Real, 4k Tampered, 4k Synthetic)
- **Test (10%):** 12,000 images (4k Real, 4k Tampered, 4k Synthetic)

---

## 4. Pipeline Execution Sequence

```
Raw Archives (Zip/Tar)
        │
        ▼
[1] extract_final_dataset.py
        │
        ▼
[2] create_deduplicated_csv.py (SHA-256 hash deduplication)
        │
        ▼
[3] analyze_content.py (CLIP zero-shot classification)
        │
        ▼
[4] reclassify_other_images_v3.py (Multi-prompt CLIP refinement)
        │
        ▼
[5] create_final_balanced_manifest.py (120k quota selection)
        │
        ▼
[6] download_sd_mj_quality.py (SD & MJ diversity ingestion)
        │
        ▼
[7] check_sd_mj_duplicates.py (Cross-dataset duplicate check)
        │
        ▼
[8] balance_and_merge_synthetic.py (Flux/SD/MJ synthetic integration)
        │
        ▼
[9] create_dataset_splits.py (80/10/10 stratified manifests)
        │
        ▼
[10] move_images_into_splits.py (Organize filesystem hierarchy)
        │
        ▼
[11-14] validate_final_dataset.py & verification scripts
```

---

## 5. Grid Portrait Extraction Utility

Extracts individual portraits from composite grid images (e.g. 5x5, 4x3, 5x6) using 1D Sobel gradient projection:

```powershell
python extract_grid_portraits.py --margin 3
```

- Automatic divider line detection
- Sub-pixel boundary alignment
- Inward border trimming (removes grid lines)
- SHA-256 uniqueness verification
