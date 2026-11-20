import csv
import random
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\image-forensics-project")
ARCHIVE_DIR = PROJECT / "raw_datasets" / "SECOUND"
INPUT_CSV = PROJECT / "analysis" / "content" / "image_content_analysis.csv"

# New output folder: does not overwrite the previous content_v2 pilot.
OUTPUT_DIR = PROJECT / "analysis" / "content_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_CSV = OUTPUT_DIR / "other_reclassification_progress.csv"
FINAL_CSV = OUTPUT_DIR / "image_content_analysis_v3.csv"
BEFORE_COUNTS_CSV = OUTPUT_DIR / "category_counts_before.csv"
BEFORE_MAPPED_COUNTS_CSV = OUTPUT_DIR / "category_counts_before_mapped_to_ten_categories.csv"
AFTER_COUNTS_CSV = OUTPUT_DIR / "category_counts_after.csv"
MAPPING_SUMMARY_CSV = OUTPUT_DIR / "category_mapping_summary.csv"
REVIEW_CSV = OUTPUT_DIR / "reclassification_review.csv"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8
RANDOM_SEED = 42

# None means process ALL eligible original "other" rows, and ONLY those rows.
# For a small test, set this to 200, run once, inspect results, then set back to None.
MAX_IMAGES_PER_CLASS = None

CLASS_LABELS = ("real", "tampered", "synthetic")
OTHER_CATEGORY = "other subjects or unclear content"

# Ten mutually exclusive subject categories. The prompt wording describes visible
# image content, not whether an image is authentic or AI-generated.
PROMPTS_BY_CATEGORY = {
    "people and portraits": [
        "a photograph whose main subject is one person or a close-up human face",
        "a portrait photograph of a person, a family, or several identifiable people",
        "a photograph of a person standing, sitting, working, or posing",
        "a documentary photograph focused on people and their activities",
        "a photo of a human face or a person from the waist up",
    ],
    "animals and wildlife": [
        "a photograph whose main subject is an animal or wild mammal",
        "a photograph of one or more birds, such as pigeons, parrots, eagles, or chickens",
        "a wildlife photograph of animals in their natural habitat",
        "a photograph of pets or livestock, such as dogs, cats, cattle, sheep, or goats",
        "a close-up photograph of an animal, bird, insect, or reptile",
    ],
    "nature and landscapes": [
        "a landscape photograph of mountains, hills, valleys, or countryside scenery",
        "a photograph of a forest, woodland, or a natural outdoor environment",
        "a scenic photograph of a natural landscape with no single dominant animal or person",
        "a photograph of a waterfall, river, lake, ocean, or broad natural scenery",
        "a landscape showing sky, terrain, rocks, or natural formations",
    ],
    "ground vehicles": [
        "a photograph whose main subject is a car, taxi, bus, or truck",
        "a photograph of motorcycles, bicycles, tractors, or other land vehicles",
        "a road traffic scene where vehicles are the main subject",
        "a photograph of a parked automobile, bus, truck, or construction vehicle",
        "a photograph of land transportation on a road or in a parking area",
    ],
    "aircraft and helicopters": [
        "a photograph whose main subject is an airplane or passenger aircraft",
        "a photograph of a helicopter, military aircraft, or small airplane",
        "a photograph of an aircraft flying in the sky",
        "a photograph of airplanes parked at an airport or on a runway",
        "a close view of an aircraft, cockpit, wing, or helicopter",
    ],
    "buildings and urban scenes": [
        "a photograph whose main subject is buildings, houses, or architecture",
        "a cityscape photograph showing streets, buildings, or a skyline",
        "a photograph of an urban street, town center, or city infrastructure",
        "a photograph of a house, apartment building, office, school, or monument",
        "a wide photograph of a town or city where buildings and streets dominate",
    ],
    "everyday objects and technology": [
        "a photograph whose main subject is an everyday object or household item",
        "a photograph of electronic devices, computers, phones, cameras, or appliances",
        "a photograph of tools, machinery parts, furniture, or equipment",
        "a product photograph of an object isolated or arranged for display",
        "a close-up photograph of manufactured items rather than people, animals, or scenery",
    ],
    "food and prepared meals": [
        "a photograph whose main subject is cooked food or a prepared meal",
        "a photograph of food served on a plate, tray, table, or in a bowl",
        "a photograph of bread, injera, stew, rice, meat, or a prepared dish",
        "a photograph of fruit, vegetables, grains, or raw food ingredients",
        "a photograph of food preparation, cooking, or a meal being served",
    ],
    "crowds, meetings, and public gatherings": [
        "a wide photograph where a large crowd or many people together is the main subject",
        "a photograph of a meeting, conference, classroom, assembly, or group discussion",
        "a photograph of a public gathering, ceremony, rally, audience, or community event",
        "a photograph of a busy market or public place filled with many people",
        "a group photograph where the gathering as a whole matters more than an individual portrait",
    ],
    "plants, flowers, and crops": [
        "a photograph whose main subject is plants, leaves, flowers, or blossoms",
        "a photograph of crops growing in a cultivated field or agricultural farm",
        "a photograph of a garden, flowering plant, tree foliage, or plant close-up",
        "a photograph of agricultural crops, seedlings, vegetables growing in soil, or farmland",
        "a botanical photograph focused on vegetation rather than a wide landscape",
    ],
}

CATEGORIES = list(PROMPTS_BY_CATEGORY.keys())
ALL_PROMPTS = []
PROMPT_RANGES = {}
for category, prompts in PROMPTS_BY_CATEGORY.items():
    start = len(ALL_PROMPTS)
    ALL_PROMPTS.extend(prompts)
    PROMPT_RANGES[category] = (start, len(ALL_PROMPTS))

# Preserve original category assignment for these eight categories; only standardize
# the names so the final table uses the ten-category vocabulary.
OLD_CATEGORY_MAP = {
    "people and portraits": "people and portraits",
    "animals and wildlife": "animals and wildlife",
    "nature and landscapes": "nature and landscapes",
    "cars, trucks, motorcycles, and other ground vehicles": "ground vehicles",
    "aircraft and helicopters": "aircraft and helicopters",
    "buildings, houses, and architecture": "buildings and urban scenes",
    "everyday objects, tools, furniture, and electronics":
        "everyday objects and technology",
    "food, fruit, and vegetables": "food and prepared meals",
}

CSV_FIELDS = [
    "class_label", "archive", "member", "sha256", "category",
    "top_score", "width", "height", "status",
]
PROGRESS_FIELDS = CSV_FIELDS + ["category_margin", "review_flag"]
FINAL_FIELDS = CSV_FIELDS + [
    "original_category", "category_margin", "review_flag",
    "reclassification_method",
]


# ============================================================
# 2. HELPERS
# ============================================================

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def record_key(row):
    # Normalize class label to avoid REAL vs real key mismatches.
    return (
        (row.get("class_label") or "").strip().lower(),
        (row.get("archive") or "").strip(),
        (row.get("member") or "").strip(),
    )


def normalize_class_label(value):
    return (value or "").strip().lower()


def write_count_csv(path, counter, categories):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_label", "category", "count"]
        )
        writer.writeheader()
        for class_name in CLASS_LABELS:
            for category in categories:
                writer.writerow({
                    "class_label": class_name,
                    "category": category,
                    "count": counter.get((class_name, category), 0),
                })

def print_counts(title, counter, categories):
    print("\n" + title)
    print("-" * 112)
    print(f"{'CATEGORY':48} {'REAL':>12} {'TAMPERED':>12} {'SYNTHETIC':>12} {'TOTAL':>12}")
    print("-" * 112)
    for category in categories:
        values = [counter.get((c, category), 0) for c in CLASS_LABELS]
        print(
            f"{category:48} {values[0]:12,} {values[1]:12,} "
            f"{values[2]:12,} {sum(values):12,}"
        )
    print("-" * 112)


def build_category_prototypes(model, processor, device):
    """Build one normalized text prototype per category using explicit CLIP outputs."""
    text_inputs = processor(
        text=ALL_PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.inference_mode():
        # Explicitly call the text encoder because some Transformers versions
        # return a model-output object from get_text_features instead of a tensor.
        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
            return_dict=True,
        )
        text_features = model.text_projection(text_outputs.pooler_output)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-12)

        prototypes = []
        for category in CATEGORIES:
            start, end = PROMPT_RANGES[category]
            prototype = text_features[start:end].mean(dim=0)
            prototype = prototype / prototype.norm().clamp(min=1e-12)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes, dim=0)

    return prototypes


def classify_batch(model, processor, images, prototypes, device):
    """Classify already-decoded PIL images and return fields expected by the main loop."""
    if not images:
        return []

    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    prototypes = prototypes.to(device)

    with torch.inference_mode():
        # Explicitly extract the pooled vision representation and project it
        # into the same embedding space as the text prototypes.
        image_outputs = model.vision_model(
            pixel_values=image_inputs["pixel_values"],
            return_dict=True,
        )
        image_features = model.visual_projection(image_outputs.pooler_output)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-12)

        similarities = image_features @ prototypes.T
        # CLIP's learned scale is used for relative comparison, not calibrated confidence.
        scaled_scores = similarities * model.logit_scale.exp()
        relative_scores = scaled_scores.softmax(dim=1)

    results = []
    for i in range(len(images)):
        order = torch.argsort(relative_scores[i], descending=True)
        best_idx = int(order[0].item())
        second_idx = int(order[1].item())
        best_score = float(relative_scores[i, best_idx].item())
        second_score = float(relative_scores[i, second_idx].item())
        margin = best_score - second_score

        # Preliminary review heuristics; validate them against human-labelled examples.
        needs_review = best_score < 0.25 or margin < 0.04
        results.append({
            "category": CATEGORIES[best_idx],
            "top_score": f"{best_score:.6f}",
            "category_margin": f"{margin:.6f}",
            "review_flag": "review" if needs_review else "",
        })

    return results


# ============================================================
# 3. LOAD ORIGINAL ANALYSIS AND COUNT BEFORE MAPPING
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

for archive_name in (
    "train_real.zip",
    "train_tampered.zip",
    "train_full_synthetic_part1.zip",
    "train_full_synthetic_part2.zip",
):
    if not (ARCHIVE_DIR / archive_name).exists():
        print(f"WARNING: expected archive not found: {ARCHIVE_DIR / archive_name}")

original_rows = read_csv(INPUT_CSV)
print(f"Original CSV rows: {len(original_rows):,}")

before_counts = Counter()
for row in original_rows:
    if (row.get("status") or "").strip().lower() == "ok":
        before_counts[(normalize_class_label(row.get("class_label")),
                       row.get("category", ""))] += 1

original_categories = sorted({row.get("category", "") for row in original_rows})
write_count_csv(BEFORE_COUNTS_CSV, before_counts, original_categories)
print_counts("COUNTS BEFORE RECLASSIFICATION (ORIGINAL LABELS)",
             before_counts, original_categories)

# Map the original eight specific category names into the standardized names,
# but leave original "other" rows as "other" for a fair before/after comparison.
before_mapped_counts = Counter()
for row in original_rows:
    if (row.get("status") or "").strip().lower() != "ok":
        continue
    class_name = normalize_class_label(row.get("class_label"))
    old_category = row.get("category", "")
    mapped_category = OLD_CATEGORY_MAP.get(old_category, old_category)
    before_mapped_counts[(class_name, mapped_category)] += 1

before_mapped_categories = CATEGORIES + [OTHER_CATEGORY]
write_count_csv(
    BEFORE_MAPPED_COUNTS_CSV, before_mapped_counts, before_mapped_categories
)
print_counts(
    "COUNTS BEFORE RECLASSIFICATION (EIGHT ORIGINAL CATEGORIES NAME-MAPPED; OTHER UNCHANGED)",
    before_mapped_counts, before_mapped_categories
)

other_by_class = defaultdict(list)
for row in original_rows:
    if (
        row.get("category") == OTHER_CATEGORY
        and (row.get("status") or "").strip().lower() == "ok"
        and normalize_class_label(row.get("class_label")) in CLASS_LABELS
    ):
        other_by_class[normalize_class_label(row.get("class_label"))].append(row)

print("\nEligible original 'other' images to reclassify:")
for class_name in CLASS_LABELS:
    print(f"  {class_name.upper():10}: {len(other_by_class[class_name]):,}")

# Load resumable progress, but only accept rows matching the current category set
# and successful/error statuses. Existing v2 progress is deliberately not imported:
# this v3 run uses the revised prompt ensemble consistently.
completed = {}
if PROGRESS_CSV.exists():
    for row in read_csv(PROGRESS_CSV):
        # Ignore malformed/stale progress rows instead of silently treating them as done.
        category_is_valid = row.get("category", "") in CATEGORIES
        is_decode_error = row.get("status", "") == "reclass_decode_error"
        if category_is_valid or is_decode_error:
            completed[record_key(row)] = row
print(f"\nAlready processed in this v3 run: {len(completed):,}")

random.seed(RANDOM_SEED)
pending_by_archive = defaultdict(list)
for class_name in CLASS_LABELS:
    pending = [
        row for row in other_by_class[class_name]
        if record_key(row) not in completed
    ]
    if MAX_IMAGES_PER_CLASS is not None:
        pending = random.sample(pending, min(MAX_IMAGES_PER_CLASS, len(pending)))
    print(f"Scheduled {len(pending):,} remaining images from {class_name.upper()}")
    for row in pending:
        pending_by_archive[row["archive"]].append(row)

# ============================================================
# 4. LOAD CLIP ON CPU AND CLASSIFY ONLY ORIGINAL 'OTHER' ROWS
# ============================================================

print("\nLoading CLIP model on CPU...")
device = torch.device("cpu")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(device)
prototypes = build_category_prototypes(model, processor, device)
prototypes = prototypes.to(device)
print("Model and category prototypes loaded.")

need_header = not PROGRESS_CSV.exists() or PROGRESS_CSV.stat().st_size == 0
with open(PROGRESS_CSV, "a", newline="", encoding="utf-8-sig") as progress_file:
    writer = csv.DictWriter(progress_file, fieldnames=PROGRESS_FIELDS)
    if need_header:
        writer.writeheader()

    for archive_name, records in pending_by_archive.items():
        archive_path = ARCHIVE_DIR / archive_name
        print(f"\nProcessing {archive_name}: {len(records):,} selected rows")
        if not archive_path.exists():
            print(f"WARNING: archive missing, skipped: {archive_path}")
            continue

        with zipfile.ZipFile(archive_path, "r") as archive:
            for start in range(0, len(records), BATCH_SIZE):
                batch_records = records[start:start + BATCH_SIZE]
                images = []
                valid_records = []

                for row in batch_records:
                    try:
                        raw = archive.read(row["member"])
                        with Image.open(BytesIO(raw)) as source:
                            source.load()
                            images.append(source.convert("RGB"))
                        valid_records.append(row)
                    except Exception as exc:
                        failed = {
                            field: row.get(field, "") for field in CSV_FIELDS
                        }
                        failed.update({
                            "status": "reclass_decode_error",
                            "category_margin": "",
                            "review_flag": "review",
                        })
                        writer.writerow(failed)
                        progress_file.flush()
                        completed[record_key(row)] = failed
                        print(f"Decode error: {row.get('member', '')}: {exc}")

                if images:
                    predictions = classify_batch(
                        model, processor, images, prototypes, device
                    )
                    for row, prediction in zip(valid_records, predictions):
                        result = {field: row.get(field, "") for field in CSV_FIELDS}
                        result.update({
                            "category": prediction["category"],
                            "top_score": prediction["top_score"],
                            "status": "ok",
                            "category_margin": prediction["category_margin"],
                            "review_flag": prediction["review_flag"],
                        })
                        writer.writerow(result)
                        completed[record_key(row)] = result

                progress_file.flush()
                done_in_archive = min(start + len(batch_records), len(records))
                print(f"  Progress: {done_in_archive:,}/{len(records):,}")

# ============================================================
# 5. CREATE COMPLETE OUTPUT CSV; PRESERVE ORIGINAL SPECIFIC LABELS
# ============================================================

print("\nCreating complete v3 CSV...")
with open(FINAL_CSV, "w", newline="", encoding="utf-8-sig") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=FINAL_FIELDS)
    writer.writeheader()

    for original in original_rows:
        old_category = original.get("category", "")
        row = {field: original.get(field, "") for field in CSV_FIELDS}
        row.update({
            "original_category": old_category,
            "category_margin": "",
            "review_flag": "",
            "reclassification_method": "preserved_original_assignment",
        })

        if old_category == OTHER_CATEGORY:
            result = completed.get(record_key(original))
            if result:
                row["category"] = result.get("category", old_category)
                row["top_score"] = result.get("top_score", "")
                row["status"] = result.get("status", original.get("status", ""))
                row["category_margin"] = result.get("category_margin", "")
                row["review_flag"] = result.get("review_flag", "review")
                row["reclassification_method"] = "CLIP_multi_prompt_embedding"
            else:
                row["reclassification_method"] = "not_yet_reclassified"
                row["review_flag"] = "review"
        elif old_category in OLD_CATEGORY_MAP:
            row["category"] = OLD_CATEGORY_MAP[old_category]
            row["reclassification_method"] = "preserved_original_assignment_name_mapped"

        writer.writerow(row)

final_rows = read_csv(FINAL_CSV)
after_counts = Counter()
for row in final_rows:
    if (row.get("status") or "").strip().lower() == "ok":
        after_counts[(normalize_class_label(row.get("class_label")),
                      row.get("category", ""))] += 1

write_count_csv(AFTER_COUNTS_CSV, after_counts, CATEGORIES)
print_counts("COUNTS AFTER RECLASSIFICATION (STANDARDIZED TEN CATEGORIES)",
             after_counts, CATEGORIES)

# ============================================================
# 6. EXPLICIT OLD CATEGORY -> FINAL CATEGORY MAPPING SUMMARY
# ============================================================

mapping_counts = Counter()
for row in final_rows:
    if (row.get("status") or "").strip().lower() != "ok":
        continue
    key = (
        normalize_class_label(row.get("class_label")),
        row.get("original_category", ""),
        row.get("category", ""),
    )
    mapping_counts[key] += 1

with open(MAPPING_SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
    fields = [
        "class_label", "original_category", "final_category",
        "count_after_mapping",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for (class_name, old_cat, new_cat), count in sorted(mapping_counts.items()):
        writer.writerow({
            "class_label": class_name,
            "original_category": old_cat,
            "final_category": new_cat,
            "count_after_mapping": count,
        })

# Include all rows that need human review, plus rows that were not processed.
with open(REVIEW_CSV, "w", newline="", encoding="utf-8-sig") as f:
    fields = FINAL_FIELDS
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in final_rows:
        if row.get("review_flag") == "review" or row.get("reclassification_method") == "not_yet_reclassified":
            writer.writerow({field: row.get(field, "") for field in fields})

print("\nFINISHED")
print(f"Original CSV (unchanged): {INPUT_CSV}")
print(f"Final analysis CSV:       {FINAL_CSV}")
print(f"Progress CSV:             {PROGRESS_CSV}")
print(f"Before counts CSV:        {BEFORE_COUNTS_CSV}")
print(f"Before mapped counts CSV:  {BEFORE_MAPPED_COUNTS_CSV}")
print(f"After counts CSV:         {AFTER_COUNTS_CSV}")
print(f"Mapping summary CSV:      {MAPPING_SUMMARY_CSV}")
print(f"Review CSV:               {REVIEW_CSV}")
print(f"Rows in final CSV:        {len(final_rows):,}")

processed_other = sum(
    1 for row in final_rows
    if row.get("original_category") == OTHER_CATEGORY
    and row.get("reclassification_method") == "CLIP_multi_prompt_embedding"
)
unprocessed_other = sum(
    1 for row in final_rows
    if row.get("original_category") == OTHER_CATEGORY
    and row.get("reclassification_method") == "not_yet_reclassified"
)
review_count = sum(1 for row in final_rows if row.get("review_flag") == "review")
print(f"Original 'other' rows reclassified in v3: {processed_other:,}")
print(f"Original 'other' rows still unprocessed:  {unprocessed_other:,}")
print(f"Rows flagged for manual review:           {review_count:,}")

print("\nInterpretation:")
print("- The first eight specific original categories were not reclassified.")
print("- Their category names were mapped to the standardized ten-category vocabulary.")
print("- Only rows originally labelled 'other subjects or unclear content' were sent to CLIP.")
print("- 'top_score' is a relative CLIP-derived score, NOT a calibrated probability.")
print("- Content categories do not establish whether an image is real, tampered, or synthetic.")
