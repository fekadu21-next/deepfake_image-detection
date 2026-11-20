from pathlib import Path
import tensorflow as tf

# Project and dataset locations
PROJECT_DIR = Path(r"D:\image-forensics-project")
DATASET_DIR = PROJECT_DIR / "final_dataset"

# Keep this order consistent across training and evaluation.
CLASS_NAMES = ["REAL", "TAMPERED", "SYNTHETIC"]

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SPLITS = ["train", "validation", "test"]


def test_split(split_name):
    """Load a small number of image batches and check their shapes and labels."""
    split_dir = DATASET_DIR / split_name

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {split_dir}\n"
            "Check that your dataset split folders exist."
        )

    # Check that each expected class folder exists and contains images.
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_count = sum(
            1 for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".gif"
            }
        )

        if image_count == 0:
            raise RuntimeError(f"No image files found in {class_dir}")

        print(f"{class_name}: {image_count:,} image files found")

    print(f"\nLoading the {split_name} dataset...")

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split_name == "train"),
        seed=42,
    )

    # Check at most two batches to keep this test small.
    batch_count = 0

    for images, labels in dataset.take(2):
        batch_count += 1

        print(f"\nBatch {batch_count}")
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Image data type:", images.dtype.name)
        print("Label data type:", labels.dtype.name)
        print("First labels:", labels.numpy()[:10].tolist())

        # Images must have shape (batch, 160, 160, 3).
        assert len(images.shape) == 4
        assert tuple(images.shape[1:]) == (160, 160, 3)
        assert images.shape[0] == labels.shape[0]

        # Labels must correspond to the three expected classes.
        assert bool(
            tf.reduce_all(
                (labels >= 0) & (labels < len(CLASS_NAMES))
            ).numpy()
        ), "A label is outside the expected class range."

    if batch_count == 0:
        raise RuntimeError(f"No batches loaded from {split_name}")

    print(f"\nSUCCESS: {split_name} loaded correctly.")


def main():
    print("=" * 55)
    print("IMAGE DATASET LOADING TEST")
    print("=" * 55)
    print("TensorFlow version:", tf.__version__)
    print("Dataset directory:", DATASET_DIR)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Class order:", CLASS_NAMES)

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    for split_name in SPLITS:
        print("\n" + "=" * 55)
        print("Testing:", split_name.upper())
        print("=" * 55)
        test_split(split_name)

    print("\n" + "=" * 55)
    print("ALL LOADING TESTS PASSED!")
    print("TensorFlow successfully loaded sample batches from all splits.")
    print("Your dataset is ready for the next training setup.")
    print("=" * 55)


if __name__ == "__main__":
    main()
