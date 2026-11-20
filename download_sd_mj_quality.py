from __future__ import annotations
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
import imagehash
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageFile

# Tolerate large image loads safely
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ============================================================
# 1. PROJECT PATHS & SETTINGS
# ============================================================

PROJECT_DIR = Path(r"D:\image-forensics-project")

# Newly collected images go to a dedicated folder under new_data/SYNTHETIC
# REAL, TAMPERED, and existing FLUX images in final_dataset/ are never touched!
OUTPUT_DIR = (
    PROJECT_DIR
    / "new_data"
    / "SYNTHETIC"
    / "collected_external_sources"
)

ANALYSIS_DIR = PROJECT_DIR / "analysis" / "multi_generator"
MANIFEST_PATH = ANALYSIS_DIR / "collected_images_manifest.csv"
SUMMARY_PATH = ANALYSIS_DIR / "collection_summary.json"
LOG_PATH = ANALYSIS_DIR / "collection.log"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("synthetic_collector")

# ============================================================
# 3. YOUR EXACT 10 CATEGORY QUOTAS (11,988 TOTAL PER SOURCE)
# ============================================================

CATEGORY_QUOTAS: Dict[str, int] = {
    "people and portraits": 4044,                     # 33.7%
    "animals and wildlife": 1596,                     # 13.3%
    "everyday objects and technology": 1548,         # 12.9%
    "ground vehicles": 1500,                          # 12.5%
    "crowds, meetings, and public gatherings": 876,   #  7.3%
    "nature and landscapes": 828,                      #  6.9%
    "food and prepared meals": 744,                    #  6.2%
    "buildings and urban scenes": 504,                  #  4.2%
    "aircraft and helicopters": 276,                    #  2.3%
    "plants, flowers and crops": 72,                    #  0.6%
}

TOTAL_CATEGORY_QUOTA = sum(CATEGORY_QUOTAS.values())  # 11,988

# ============================================================
# 4. QUALITY & REALISM FILTERING THRESHOLDS
# ============================================================

MIN_WIDTH = 512
MIN_HEIGHT = 512
MAX_ASPECT_RATIO = 2.0
MIN_LAPLACIAN_VARIANCE = 50.0       # Sharpness threshold
MIN_CATEGORY_MARGIN = 0.008         # CLIP category ranking confidence
PHASH_DUPLICATE_DISTANCE = 4        # Near-duplicate visual threshold
SAVE_PROGRESS_EVERY = 25            # Log and update summary every N saved images
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 5. CONTENT CATEGORIES AND CLIP PROMPTS
# ============================================================

CATEGORY_PROMPTS: Dict[str, List[str]] = {
    "people and portraits": [
        "a realistic photograph of a person's face or portrait",
        "a realistic photograph of one or more people",
        "a realistic photograph of a person in everyday life",
    ],
    "animals and wildlife": [
        "a realistic photograph of a wild animal",
        "a realistic photograph of a pet or domestic animal",
        "a realistic photograph of birds or wildlife",
    ],
    "nature and landscapes": [
        "a realistic photograph of a natural landscape",
        "a realistic photograph of mountains, rivers, or forests",
        "a realistic photograph of a beach or outdoor scenery",
    ],
    "ground vehicles": [
        "a realistic photograph of a car, bus, or truck",
        "a realistic photograph of a motorcycle or bicycle",
        "a realistic photograph of a train or other ground vehicle",
    ],
    "aircraft and helicopters": [
        "a realistic photograph of an airplane",
        "a realistic photograph of a helicopter",
        "a realistic photograph of an aircraft at an airport",
    ],
    "buildings and urban scenes": [
        "a realistic photograph of a building or house",
        "a realistic photograph of a city street or urban scene",
        "a realistic photograph of architecture",
    ],
    "everyday objects and technology": [
        "a realistic photograph of everyday household objects",
        "a realistic photograph of electronic devices or technology",
        "a realistic photograph of tools, furniture, or equipment",
    ],
    "food and prepared meals": [
        "a realistic photograph of prepared food or a meal",
        "a realistic photograph of fruit, vegetables, or ingredients",
        "a realistic photograph of food served on a plate",
    ],
    "crowds, meetings, and public gatherings": [
        "a realistic photograph of a crowd of people in a public place",
        "a realistic photograph of a meeting or group gathering",
        "a realistic photograph of an audience or public event",
    ],
    "plants, flowers and crops": [
        "a realistic photograph of flowers or ornamental plants",
        "a realistic photograph of crops growing in a field",
        "a realistic photograph of plants, leaves, or agricultural crops",
    ],
}

CATEGORY_NAMES = list(CATEGORY_QUOTAS.keys())

# Realism Prompts: Compares realistic photograph against non-photo art styles
PHOTO_PROMPTS = [
    "a realistic, crisp, authentic photograph taken with a DSLR camera, documentary photo, natural lighting",
    "a clear realistic real life photograph of real world subjects",
]

NON_PHOTO_PROMPTS = [
    "an anime drawing, manga illustration, cartoon, or comic art",
    "a 3D CGI digital render, fantasy concept art, video game graphics",
    "an oil painting, watercolor, artistic sketch, or abstract digital artwork",
    "a distorted, deformed, surreal, artificial illustration",
]

# ============================================================
# 6. LOAD CLIP MODEL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEVICE)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
logger.info("Loading CLIP model: %s", CLIP_MODEL_NAME)

from transformers import CLIPModel, CLIPProcessor

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_model.eval()

# Precompute prompt indexing for category classification
ALL_CATEGORY_PROMPTS = []
PROMPT_CATEGORY_INDEX = []

for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    for prompt in CATEGORY_PROMPTS[cat_name]:
        ALL_CATEGORY_PROMPTS.append(prompt)
        PROMPT_CATEGORY_INDEX.append(cat_idx)

PROMPT_CATEGORY_INDEX = np.asarray(PROMPT_CATEGORY_INDEX)

REALISM_PROMPTS = PHOTO_PROMPTS + NON_PHOTO_PROMPTS

# ============================================================
# 7. IMAGE QUALITY & PHOTOREALISM FILTERING
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_filename(text: str, max_length: int = 70) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return (text or "image")[:max_length]

def check_image_technical_quality(image: Image.Image) -> Tuple[bool, str, Optional[float]]:
    """Technical screening: resolution, aspect ratio, sharpness."""
    width, height = image.size

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, "resolution_too_small", None
    ratio = max(width / height, height / width)
    if ratio > MAX_ASPECT_RATIO:
        return False, "aspect_ratio_rejected", None
    # Check Laplacian sharpness variance
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    if sharpness < MIN_LAPLACIAN_VARIANCE:
        return False, "sharpness_below_threshold", sharpness

    return True, "passed", sharpness

@torch.inference_mode()
def verify_photorealism(image: Image.Image) -> Tuple[bool, float]:
    """
    Zero-shot CLIP verification to reject non-photorealistic images:
    Anime, cartoons, 3D CGI, paintings, sketches, and surreal art.
    Returns: (is_realistic, photo_confidence)
    """
    inputs = processor(
        text=REALISM_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]  # shape: [len(REALISM_PROMPTS)]
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    # Sum probabilities of photo prompts vs non-photo prompts
    num_photo = len(PHOTO_PROMPTS)
    photo_prob = float(np.sum(probs[:num_photo]))
    non_photo_prob = float(np.sum(probs[num_photo:]))

    is_realistic = photo_prob >= non_photo_prob
    return is_realistic, photo_prob

@torch.inference_mode()
def classify_category(image: Image.Image) -> Tuple[Optional[str], float, float, str]:
    """Assign image to one of the 10 target categories with confidence margin."""
    inputs = processor(
        text=ALL_CATEGORY_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = clip_model(**inputs)
    similarities = outputs.logits_per_image[0].float().cpu().numpy()

    category_scores = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        prompt_scores = similarities[PROMPT_CATEGORY_INDEX == cat_idx]
        category_scores[cat_name] = float(np.mean(prompt_scores))

    ranked = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = float(best_score - second_score)

    if margin < MIN_CATEGORY_MARGIN:
        return None, best_score, margin, "uncertain_category"

    return best_category, best_score, margin, "classified"

# ============================================================
# 8. DEDUPLICATION & RESUME TRACKING
# ============================================================

PIXEL_HASHES: Set[str] = set()
PHASHES: List[Tuple[imagehash.ImageHash, str]] = []
SAVED_COUNTS: Dict[str, Counter] = {
    "STABLE_DIFFUSION": Counter(),
    "MIDJOURNEY": Counter(),
}

MANIFEST_FIELDS = [
    "timestamp_utc",
    "source_name",
    "category",
    "pixel_sha256",
    "saved_file_sha256",
    "perceptual_hash",
    "width",
    "height",
    "sharpness",
    "photo_confidence",
    "clip_category_score",
    "clip_category_margin",
    "output_path",
]

def image_pixel_hash(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    payload = str(normalized.size).encode("utf-8") + normalized.tobytes()
    return hashlib.sha256(payload).hexdigest()

def image_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def is_perceptual_duplicate(phash: imagehash.ImageHash) -> bool:
    for existing_hash, _ in PHASHES:
        try:
            if phash - existing_hash <= PHASH_DUPLICATE_DISTANCE:
                return True
        except Exception:
            continue
    return False

def clean_leftover_tmp_files():
    """Remove any partial .tmp files left from an unexpected system power-down."""
    for tmp_file in OUTPUT_DIR.rglob("*.tmp"):
        try:
            tmp_file.unlink()
            logger.info("Cleaned interrupted temp file: %s", tmp_file.name)
        except Exception as e:
            logger.warning("Could not delete %s: %s", tmp_file, e)

def load_existing_outputs():
    """
    Scan OUTPUT_DIR to index all previously saved images.
    Guarantees that stopping and resuming never duplicates images
    and never loses track of current category counts.
    """
    clean_leftover_tmp_files()
    logger.info("Scanning existing downloaded images for resume...")

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        p for p in OUTPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    valid_images = 0
    for idx, path in enumerate(image_paths, start=1):
        try:
            with Image.open(path) as opened:
                opened.verify()  # Guarantee file is not corrupted
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                pixel_hash = image_pixel_hash(image)
                phash = imagehash.phash(image)

                PIXEL_HASHES.add(pixel_hash)
                PHASHES.append((phash, str(path)))

                # Determine source and category from path hierarchy
                category = path.parent.name
                source_name = path.parent.parent.name.upper()

                if source_name in SAVED_COUNTS and category in CATEGORY_QUOTAS:
                    SAVED_COUNTS[source_name][category] += 1
                    valid_images += 1
        except Exception as exc:
            logger.warning("Corrupted image found (%s), removing: %s", path.name, exc)
            try:
                path.unlink()
            except Exception:
                pass

        if idx % 1000 == 0:
            logger.info("Indexed %d existing images...", idx)

    logger.info("Existing valid images scanned: %d", valid_images)
    for source in ["STABLE_DIFFUSION", "MIDJOURNEY"]:
        total = sum(SAVED_COUNTS[source].values())
        logger.info("%s current progress: %d/%d images", source, total, TOTAL_CATEGORY_QUOTA)

def append_manifest(row: Dict[str, Any]):
    file_exists = MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if not file_exists or MANIFEST_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

def save_summary(status_by_source: Dict[str, str], scanned_by_source: Counter, rejected_by_source: Dict[str, Counter]):
    summary = {
        "last_updated_utc": utc_now(),
        "project_dir": str(PROJECT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": DEVICE,
        "clip_model": CLIP_MODEL_NAME,
        "total_quota_per_source": TOTAL_CATEGORY_QUOTA,
        "category_quotas": CATEGORY_QUOTAS,
        "sources": {},
    }

    for source_name in ["STABLE_DIFFUSION", "MIDJOURNEY"]:
        counts = {cat: int(SAVED_COUNTS[source_name][cat]) for cat in CATEGORY_NAMES}
        shortages = {cat: max(0, CATEGORY_QUOTAS[cat] - counts[cat]) for cat in CATEGORY_NAMES}
        summary["sources"][source_name] = {
            "status": status_by_source.get(source_name, "pending"),
            "scanned": int(scanned_by_source[source_name]),
            "accepted_total": int(sum(counts.values())),
            "shortage_total": int(sum(shortages.values())),
            "accepted_by_category": counts,
            "shortages_by_category": shortages,
            "rejected_reasons": dict(rejected_by_source[source_name]),
        }

    tmp_path = SUMMARY_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    tmp_path.replace(SUMMARY_PATH)

def is_source_complete(source_name: str) -> bool:
    return all(
        SAVED_COUNTS[source_name][cat] >= CATEGORY_QUOTAS[cat]
        for cat in CATEGORY_NAMES
    )

CHECKPOINT_PATH = ANALYSIS_DIR / "sd_checkpoint.json"
MJ_CHECKPOINT_PATH = ANALYSIS_DIR / "mj_checkpoint.json"

def get_last_completed_sd_part() -> int:
    """
    Detect the last completed part from checkpoint or existing manifest
    so restarting never re-downloads already completed parts.
    """
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("last_completed_part", 0))
        except Exception:
            pass

    max_part = 0
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out = row.get("output_path", "")
                    m = re.search(r"_sd_p(\d+)_", out)
                    if m:
                        max_part = max(max_part, int(m.group(1)))
        except Exception:
            pass

    return max_part

def save_sd_part_checkpoint(part_id: int):
    """Save the completed part to checkpoint file."""
    try:
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_completed_part": part_id, "updated_at_utc": utc_now()}, f, indent=2)
        tmp.replace(CHECKPOINT_PATH)
    except Exception as e:
        logger.warning("Could not save checkpoint: %s", e)

def get_last_completed_mj_shard() -> int:
    """
    Detect the last completed Midjourney shard from checkpoint or existing manifest
    so restarting never re-downloads already completed shards.
    """
    if MJ_CHECKPOINT_PATH.exists():
        try:
            with open(MJ_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("last_completed_shard", 0))
        except Exception:
            pass

    max_shard = 0
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out = row.get("output_path", "")
                    m = re.search(r"_mj_s(\d+)_", out)
                    if m:
                        max_shard = max(max_shard, int(m.group(1)))
        except Exception:
            pass

    return max_shard

def save_mj_shard_checkpoint(shard_id: int):
    """Save the completed Midjourney shard to checkpoint file."""
    try:
        tmp = MJ_CHECKPOINT_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_completed_shard": shard_id, "updated_at_utc": utc_now()}, f, indent=2)
        tmp.replace(MJ_CHECKPOINT_PATH)
    except Exception as e:
        logger.warning("Could not save Midjourney checkpoint: %s", e)

# ============================================================
# 9. RESILIENT DOWNLOAD HELPERS (WITH AUTO-RETRY)
# ============================================================

def download_with_retry(url: str, output_path: Path, max_retries: int = 5) -> bool:
    """Download a file with chunked streaming and automatic backoff on network drop."""
    delays = [5, 10, 20, 30, 60]
    temp_path = output_path.with_suffix(".downloading")

    for attempt in range(max_retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            # Resume partial download if supported
            existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            # 60-second read timeout to prevent disconnects on slow server periods
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                if response.status_code == 416: # Range not satisfiable (already complete)
                    temp_path.replace(output_path)
                    return True
                if response.status_code not in (200, 206):
                    logger.warning("HTTP %d for URL %s", response.status_code, url)
                    time.sleep(delays[attempt])
                    continue

                mode = "ab" if existing_bytes > 0 and response.status_code == 206 else "wb"
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)

            temp_path.replace(output_path)
            return True

        except (requests.RequestException, ConnectionError, TimeoutError, IOError) as exc:
            delay = delays[min(attempt, len(delays) - 1)]
            logger.warning("Network issue (%s). Retrying in %ds (Attempt %d/%d)...", exc, delay, attempt + 1, max_retries)
            time.sleep(delay)

    if temp_path.exists():
        temp_path.unlink()
    return False

# ============================================================
# 10. STREAMING SOURCES IMPLEMENTATION
# ============================================================

def stream_stable_diffusion_images(start_part: Optional[int] = None, end_part: int = 197) -> Generator[Tuple[Image.Image, str], None, None]:
    """
    Stream Stable Diffusion images part by part from DiffusionDB.
    Downloads one 500MB zip at a time, yields 1,000 images, and immediately cleans up.
    Automatically resumes from the last completed part (or Part 24/25).
    """
    temp_cache_dir = PROJECT_DIR / "new_data" / "SYNTHETIC" / "_temp_sd_zips"
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    # Automatically detect where to resume (Starts directly at Part 25 as requested)
    if start_part is None:
        last_part = get_last_completed_sd_part()
        start_part = max(25, last_part + 1)

    logger.info("Stable Diffusion resuming from Part %06d through Part %06d", start_part, end_part)

    for part_id in range(start_part, end_part + 1):
        zip_url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part_id:06d}.zip"
        zip_path = temp_cache_dir / f"part-{part_id:06d}.zip"

        logger.info("Fetching Stable Diffusion part %06d (zip)...", part_id)
        success = download_with_retry(zip_url, zip_path)
        if not success:
            logger.warning("Could not download part %06d; skipping to next.", part_id)
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for file_name in zf.namelist():
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        try:
                            image_data = zf.read(file_name)
                            with Image.open(io.BytesIO(image_data)) as img:
                                yield img.convert("RGB"), f"sd_p{part_id}_{file_name}"
                        except Exception:
                            continue
            # Successfully finished processing this part
            save_sd_part_checkpoint(part_id)
        except Exception as err:
            logger.warning("Error reading part %06d zip: %s", part_id, err)
        finally:
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass

def stream_midjourney_images(start_shard: Optional[int] = None) -> Generator[Tuple[Image.Image, str], None, None]:
    """
    Stream Midjourney images using public, verified Parquet shards:
    1. ehristoforu/midjourney-images  — downloaded via hf_hub_download (refs/convert/parquet branch)
    2. brivangl/midjourney-v6-llava   — direct resolve/main URLs (reliable)
    Fetches shard by shard, processes images, and cleans up.
    Automatically resumes from the last completed shard.
    """
    from huggingface_hub import hf_hub_download as _hf_hub_download

    temp_cache_dir = PROJECT_DIR / "new_data" / "SYNTHETIC" / "_temp_mj_shards"
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    # Source 1: ehristoforu — 3 shards via hf_hub_download (refs/convert/parquet branch)
    ehristoforu_filenames = ["default/train/0000.parquet", "default/train/0001.parquet", "default/train/0002.parquet"]

    # Source 2: brivangl/midjourney-v6-llava — 50 shards via direct URL
    v6_shards = [
        f"https://huggingface.co/datasets/brivangl/midjourney-v6-llava/resolve/main/data/train_{i:03d}.parquet"
        for i in range(50)
    ]

    # Build combined shard list as (source_tag, descriptor) tuples
    # For ehristoforu: descriptor is the filename string (use hf_hub_download)
    # For mj_v6: descriptor is the direct URL (use download_with_retry)
    all_shards = (
        [("ehristoforu", fn) for fn in ehristoforu_filenames]
        + [("mj_v6", u) for u in v6_shards]
    )
    total_shards = len(all_shards)

    if start_shard is None:
        last_shard = get_last_completed_mj_shard()
        start_shard = last_shard + 1

    logger.info("Midjourney resuming from Shard %d through Shard %d", start_shard, total_shards)

    for shard_idx, (source_tag, descriptor) in enumerate(all_shards, start=1):
        if shard_idx < start_shard:
            continue

        parquet_path = temp_cache_dir / f"mj_shard_{shard_idx:03d}.parquet"
        logger.info("Fetching Midjourney shard %d/%d (%s)...", shard_idx, total_shards, source_tag)

        success = False
        if source_tag == "ehristoforu":
            # Must use hf_hub_download; direct /resolve/refs/convert/parquet/ URLs return 404
            try:
                downloaded_path = _hf_hub_download(
                    repo_id="ehristoforu/midjourney-images",
                    filename=descriptor,
                    repo_type="dataset",
                    revision="refs/convert/parquet",
                    local_dir=str(temp_cache_dir),
                )
                # hf_hub_download saves under a nested path; copy/rename to our expected location
                import shutil as _shutil
                _shutil.copy2(downloaded_path, parquet_path)
                success = True
            except Exception as hf_err:
                logger.warning("Could not download ehristoforu shard %s via hf_hub_download: %s", descriptor, hf_err)
        else:
            success = download_with_retry(descriptor, parquet_path)

        if not success:
            logger.warning("Could not download Midjourney shard %d (%s); skipping.", shard_idx, source_tag)
            continue

        try:
            table = pq.read_table(parquet_path)
            num_rows = len(table)
            logger.info("Processing %d images from Midjourney shard %d...", num_rows, shard_idx)

            col_names = table.column_names
            image_col = next((c for c in ["image", "img", "picture"] if c in col_names), None)

            if not image_col:
                continue

            for row_idx in range(num_rows):
                val = table[image_col][row_idx].as_py()
                raw_bytes = None
                if isinstance(val, dict):
                    raw_bytes = val.get("bytes")
                elif isinstance(val, (bytes, bytearray)):
                    raw_bytes = val

                if raw_bytes:
                    try:
                        with Image.open(io.BytesIO(raw_bytes)) as img:
                            yield img.convert("RGB"), f"mj_s{shard_idx}_{row_idx}"
                    except Exception:
                        continue
            # Successfully finished processing this shard
            save_mj_shard_checkpoint(shard_idx)
        except Exception as err:
            logger.warning("Error reading parquet shard %d: %s", shard_idx, err)
        finally:
            if parquet_path.exists():
                try:
                    parquet_path.unlink()
                except Exception:
                    pass

# ============================================================
# 11. CORE COLLECTION RUNNER FOR ONE SOURCE
# ============================================================

def collect_source(
    source_name: str,
    status_by_source: Dict[str, str],
    scanned_by_source: Counter,
    rejected_by_source: Dict[str, Counter],
):
    logger.info("=" * 75)
    logger.info("STARTING SOURCE: %s", source_name)
    logger.info("Target: %d content-balanced images", TOTAL_CATEGORY_QUOTA)
    logger.info("=" * 75)

    if is_source_complete(source_name):
        logger.info("Source %s is already 100%% complete! Skipping.", source_name)
        status_by_source[source_name] = "already_complete"
        return

    status_by_source[source_name] = "running"
    source_start = time.time()
    saved_since_progress = 0

    # Select generator
    if source_name == "STABLE_DIFFUSION":
        stream = stream_stable_diffusion_images()
    else:
        stream = stream_midjourney_images()

    try:
        for record_index, (image, item_id) in enumerate(stream):
            scanned_by_source[source_name] += 1

            if is_source_complete(source_name):
                status_by_source[source_name] = "complete"
                logger.info("Source %s reached 100%% quota completion!", source_name)
                break

            # 1. Technical Quality Filter
            passed, reason, sharpness = check_image_technical_quality(image)
            if not passed:
                rejected_by_source[source_name][reason] += 1
                continue

            # 2. Photorealism Verification (Filter out anime, cartoons, CGI, 3D render, art)
            is_realistic, photo_confidence = verify_photorealism(image)
            if not is_realistic:
                rejected_by_source[source_name]["rejected_non_photorealistic"] += 1
                continue

            # 3. Content Category Assignment
            category, clip_score, clip_margin, cat_status = classify_category(image)
            if cat_status != "classified" or category is None:
                rejected_by_source[source_name]["uncertain_category"] += 1
                continue

            # 4. Check Category Quota
            if SAVED_COUNTS[source_name][category] >= CATEGORY_QUOTAS[category]:
                rejected_by_source[source_name]["category_quota_full"] += 1
                continue

            # 5. Exact Pixel Deduplication
            pixel_hash = image_pixel_hash(image)
            if pixel_hash in PIXEL_HASHES:
                rejected_by_source[source_name]["duplicate_pixel_hash"] += 1
                continue

            # 6. Perceptual Deduplication
            phash = imagehash.phash(image)
            if is_perceptual_duplicate(phash):
                rejected_by_source[source_name]["perceptual_duplicate"] += 1
                continue

            # 7. Atomic Save (save to .tmp, then rename to .jpg)
            category_dir = OUTPUT_DIR / source_name / category
            category_dir.mkdir(parents=True, exist_ok=True)

            img_num = SAVED_COUNTS[source_name][category] + 1
            safe_id = safe_filename(item_id)
            final_filename = f"{source_name.lower()}_{safe_id}_{img_num:06d}.jpg"
            output_path = category_dir / final_filename

            temp_output = output_path.with_suffix(".tmp")
            try:
                image.save(temp_output, format="JPEG", quality=95, optimize=True)
                temp_output.replace(output_path)
            except Exception as err:
                rejected_by_source[source_name]["save_error"] += 1
                if temp_output.exists():
                    temp_output.unlink()
                continue

            saved_file_hash = image_file_hash(output_path)
            PIXEL_HASHES.add(pixel_hash)
            PHASHES.append((phash, str(output_path)))
            SAVED_COUNTS[source_name][category] += 1
            saved_since_progress += 1

            # Append metadata to manifest CSV
            row = {
                "timestamp_utc": utc_now(),
                "source_name": source_name,
                "category": category,
                "pixel_sha256": pixel_hash,
                "saved_file_sha256": saved_file_hash,
                "perceptual_hash": str(phash),
                "width": image.width,
                "height": image.height,
                "sharpness": sharpness,
                "photo_confidence": round(photo_confidence, 4),
                "clip_category_score": round(clip_score, 4),
                "clip_category_margin": round(clip_margin, 4),
                "output_path": str(output_path),
            }
            append_manifest(row)

            # Periodic Progress Logging & Summary Save
            if saved_since_progress >= SAVE_PROGRESS_EVERY:
                elapsed = max(0.001, time.time() - source_start)
                accepted = sum(SAVED_COUNTS[source_name].values())
                logger.info(
                    "%s: accepted=%d/%d | scanned=%d | elapsed=%.1f min",
                    source_name,
                    accepted,
                    TOTAL_CATEGORY_QUOTA,
                    scanned_by_source[source_name],
                    elapsed / 60,
                )
                save_summary(status_by_source, scanned_by_source, rejected_by_source)
                saved_since_progress = 0

        if is_source_complete(source_name):
            status_by_source[source_name] = "complete"
        else:
            status_by_source[source_name] = "stream_exhausted"

    except KeyboardInterrupt:
        status_by_source[source_name] = "interrupted"
        logger.warning("\n[INTERRUPTED] Script paused by user (Ctrl+C). Progress is safely saved.")
    except Exception as exc:
        status_by_source[source_name] = "collection_error"
        logger.exception("Error during %s collection: %s", source_name, exc)
    finally:
        save_summary(status_by_source, scanned_by_source, rejected_by_source)

# ============================================================
# 12. MAIN EXECUTION PIPELINE
# ============================================================

def main():
    logger.info("=" * 75)
    logger.info("SYNTHETIC IMAGE COLLECTION: STABLE DIFFUSION & MIDJOURNEY")
    logger.info("Project directory: %s", PROJECT_DIR)
    logger.info("Output directory:  %s", OUTPUT_DIR)
    logger.info("Target per source: %d images across 10 balanced categories", TOTAL_CATEGORY_QUOTA)
    logger.info("Device:            %s", DEVICE)
    logger.info("=" * 75)

    # 1. Scan and index existing images (Enables stop-and-resume)
    load_existing_outputs()

    status_by_source = {"STABLE_DIFFUSION": "not_started", "MIDJOURNEY": "not_started"}
    scanned_by_source = Counter()
    rejected_by_source = {
        "STABLE_DIFFUSION": Counter(),
        "MIDJOURNEY": Counter(),
    }

    save_summary(status_by_source, scanned_by_source, rejected_by_source)

    # 2. Collect Stable Diffusion
    collect_source("STABLE_DIFFUSION", status_by_source, scanned_by_source, rejected_by_source)

    # 3. Collect Midjourney
    collect_source("MIDJOURNEY", status_by_source, scanned_by_source, rejected_by_source)

    # 4. Final Report
    logger.info("=" * 75)
    logger.info("FINAL COLLECTION REPORT")
    logger.info("=" * 75)

    for source_name in ["STABLE_DIFFUSION", "MIDJOURNEY"]:
        total = sum(SAVED_COUNTS[source_name].values())
        logger.info("%s: %d/%d images collected | status=%s", source_name, total, TOTAL_CATEGORY_QUOTA, status_by_source[source_name])
        for cat in CATEGORY_NAMES:
            cur = SAVED_COUNTS[source_name][cat]
            target = CATEGORY_QUOTAS[cat]
            logger.info("  %-42s %5d/%5d | shortage=%d", cat, cur, target, max(0, target - cur))

    logger.info("Manifest CSV: %s", MANIFEST_PATH)
    logger.info("Summary JSON: %s", SUMMARY_PATH)
    logger.info("Log File:     %s", LOG_PATH)
    logger.info("Collection finished cleanly.")

if __name__ == "__main__":
    main()
