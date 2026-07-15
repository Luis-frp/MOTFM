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
        "--patient_level",
        default=1,
        type=int,
        help=(
            "Directory depth used to identify the patient from each image path. "
            "1 means the first folder under --input_dir, 2 means the second, and so on."
        ),
    )
    parser.add_argument(
        "--class_name",
        default="gastric_cancer",
        type=str,
        help="Class label assigned to every image unless --class_from_parent is set.",
    )
    parser.add_argument(
        "--class_from_parent",
        action="store_true",
        help="Use an ancestor folder name as the class label.",
    )
    parser.add_argument(
        "--class_parent_level",
        default=1,
        type=int,
        help=(
            "How many levels up from the image file to use as the class label. "
            "1 means the immediate parent folder, 2 means the grandparent folder, and so on."
        ),
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Store image paths instead of loading all images into memory; load them on demand during training.",
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


def get_patient_id(path: Path, root: Path, patient_level: int) -> str:
    rel_parts = path.relative_to(root).parts
    if patient_level < 1:
        raise ValueError("--patient_level must be >= 1.")
    if len(rel_parts) < patient_level:
        raise ValueError(
            f"Cannot infer patient id from '{path}': "
            f"relative path has only {len(rel_parts)} part(s), but --patient_level={patient_level}."
        )
    return rel_parts[patient_level - 1]


def split_paths_by_patient(
    paths: List[Path], root: Path, valid_fraction: float, seed: int, patient_level: int
) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("--valid_fraction must be between 0 and 1.")

    patient_to_paths = {}
    for path in paths:
        patient_id = get_patient_id(path, root, patient_level)
        patient_to_paths.setdefault(patient_id, []).append(path)

    patients = list(patient_to_paths.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    valid_target = max(1, int(round(len(paths) * valid_fraction)))
    valid_patients = []
    valid_count = 0
    for patient_id in patients:
        if valid_count >= valid_target and len(valid_patients) > 0:
            break
        valid_patients.append(patient_id)
        valid_count += len(patient_to_paths[patient_id])

    if len(valid_patients) == len(patients):
        if len(valid_patients) < 2:
            raise ValueError("Need at least 2 patients to create a train/valid split.")
        valid_patients = valid_patients[:1]

    valid_set = set(valid_patients)
    valid_paths = [p for pid in valid_patients for p in patient_to_paths[pid]]
    train_paths = [p for pid in patients if pid not in valid_set for p in patient_to_paths[pid]]

    if not train_paths or not valid_paths:
        raise ValueError(
            "Failed to create a non-empty train/valid split by patient. "
            "Check --patient_level and --valid_fraction."
        )

    return train_paths, valid_paths


def build_entries(
    paths: List[Path],
    root: Path,
    image_size: int,
    class_name: str,
    class_from_parent: bool,
    class_parent_level: int,
    lazy: bool,
) -> List[dict]:
    if class_from_parent and class_parent_level < 1:
        raise ValueError("--class_parent_level must be >= 1.")

    entries = []
    for path in paths:
        if class_from_parent:
            parent = path
            for _ in range(class_parent_level):
                parent = parent.parent
            class_label = parent.name
        else:
            class_label = class_name

        entry = {
            "class": class_label,
            "name": path.stem,
            "metadata": {"source_path": str(path.relative_to(root))},
        }
        if not lazy:
            entry["image"] = load_rgb_chw(path, image_size)
        else:
            entry["source_path"] = str(path.relative_to(root))
        entries.append(entry)
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

    train_paths, valid_paths = split_paths_by_patient(
        paths,
        input_dir,
        args.valid_fraction,
        args.seed,
        args.patient_level,
    )
    dataset = {
        "train": build_entries(
            train_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
            args.class_parent_level,
            args.lazy,
        ),
        "valid": build_entries(
            valid_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
            args.class_parent_level,
            args.lazy,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved: {output_path}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Valid samples: {len(dataset['valid'])}")
    if args.lazy:
        print("Storage mode: lazy (paths only)")
    else:
        print(f"Image shape: {dataset['train'][0]['image'].shape}")


if __name__ == "__main__":
    main()
