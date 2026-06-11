import argparse
import pickle
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a MOTFM RGB pickle from a folder of 2D images."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Folder containing gastric cancer endoscopy images.",
    )
    parser.add_argument(
        "--output_path",
        default="data/gastric_cancer_rgb.pkl",
        type=str,
        help="Output pickle path.",
    )
    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        help="Square size used after resize/crop.",
    )
    parser.add_argument(
        "--valid_fraction",
        default=0.15,
        type=float,
        help="Fraction of images reserved for validation.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random split seed.")
    parser.add_argument(
        "--class_name",
        default="gastric_cancer",
        type=str,
        help="Class label assigned to every image unless --class_from_parent is set.",
    )
    parser.add_argument(
        "--class_from_parent",
        action="store_true",
        help="Use each image's immediate parent folder name as its class label.",
    )
    return parser.parse_args()


def iter_image_paths(input_dir: Path) -> Iterable[Path]:
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_rgb_chw(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.fit(
            img,
            size=(image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)


def split_paths(
    paths: List[Path], valid_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("--valid_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_fraction)))
    if valid_count >= len(shuffled):
        valid_count = len(shuffled) - 1
    return shuffled[valid_count:], shuffled[:valid_count]


def build_entries(
    paths: List[Path],
    root: Path,
    image_size: int,
    class_name: str,
    class_from_parent: bool,
) -> List[dict]:
    entries = []
    for path in paths:
        class_label = path.parent.name if class_from_parent else class_name
        entries.append(
            {
                "image": load_rgb_chw(path, image_size),
                "class": class_label,
                "name": path.stem,
                "metadata": {"source_path": str(path.relative_to(root))},
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir is not a directory: {input_dir}")

    paths = sorted(iter_image_paths(input_dir))
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images, found {len(paths)} in {input_dir}.")

    train_paths, valid_paths = split_paths(paths, args.valid_fraction, args.seed)
    dataset = {
        "train": build_entries(
            train_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
        ),
        "valid": build_entries(
            valid_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved: {output_path}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Valid samples: {len(dataset['valid'])}")
    print(f"Image shape: {dataset['train'][0]['image'].shape}")


if __name__ == "__main__":
    main()
