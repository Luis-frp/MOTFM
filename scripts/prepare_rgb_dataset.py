import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
        "--manifest_csv",
        default=None,
        type=str,
        help=(
            "Optional CSV manifest with columns 'path,class' (required), plus optional "
            "'patient' (rows sharing a patient id are kept in the same split) and 'split' "
            "(values 'train'/'valid'; when present for every row, splitting is taken verbatim "
            "from the CSV and --valid_fraction/--patient_level/--class_from_parent are ignored). "
            "'path' may be absolute or relative to --input_dir. When omitted, class/patient are "
            "inferred from the folder structure under --input_dir instead."
        ),
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
        default=0.1,
        type=float,
        help="Fraction of images (per class) reserved for validation.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random split seed.")
    parser.add_argument(
        "--patient_level",
        default=1,
        type=int,
        help=(
            "Directory depth used to identify the patient from each image path. "
            "1 means the first folder under --input_dir, 2 means the second, and so on. "
            "Use 0 when there is no patient/exam subfolder (e.g. <input_dir>/<class>/<image>) "
            "to split at the image level instead of grouping a whole class folder as one patient. "
            "Ignored when --manifest_csv is used."
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


def get_class_label(
    path: Path, class_name: str, class_from_parent: bool, class_parent_level: int
) -> str:
    if not class_from_parent:
        return class_name
    if class_parent_level < 1:
        raise ValueError("--class_parent_level must be >= 1.")
    parent = path
    for _ in range(class_parent_level):
        parent = parent.parent
    return parent.name


def get_patient_id(path: Path, root: Path, patient_level: int) -> str:
    """Identify a patient from the directories leading up to an image file.

    The id is the joined path of all directory components up to and including
    ``patient_level`` (e.g. for IGED's ``<class>/<patient>/<image>`` layout,
    ``patient_level=2`` yields ids like ``type_A/031``). Including the leading
    components keeps ids unique even if two classes happen to reuse the same
    patient folder name.

    ``patient_level=0`` means there is no patient subfolder to group by (e.g.
    ``<input_dir>/<class>/<image>``): each image becomes its own group, so
    splitting happens at the image level instead of grouping a whole class
    folder into a single "patient".
    """
    if patient_level == 0:
        return str(path)
    rel_parts = path.relative_to(root).parts
    if patient_level < 0:
        raise ValueError("--patient_level must be >= 0.")
    if len(rel_parts) <= patient_level:
        raise ValueError(
            f"Cannot infer patient id from '{path}': "
            f"relative path has only {len(rel_parts)} part(s), but --patient_level={patient_level}."
        )
    return "/".join(rel_parts[:patient_level])


def _split_patient_group(
    patient_to_items: Dict[str, List[Any]],
    valid_fraction: float,
    rng: random.Random,
) -> Tuple[List[Any], List[Any]]:
    """Splits one class's patients into train/valid by cumulative item count."""
    patients = list(patient_to_items.keys())
    rng.shuffle(patients)

    total = sum(len(v) for v in patient_to_items.values())
    valid_target = max(1, int(round(total * valid_fraction)))

    n = len(patients)
    i = 0
    count = 0
    valid_patients: List[str] = []
    while i < n - 1 and count < valid_target:
        valid_patients.append(patients[i])
        count += len(patient_to_items[patients[i]])
        i += 1

    train_patients = patients[i:]

    train = [item for pid in train_patients for item in patient_to_items[pid]]
    valid = [item for pid in valid_patients for item in patient_to_items[pid]]
    return train, valid


def _stratified_split(
    class_to_patients: Dict[str, Dict[str, List[Any]]],
    valid_fraction: float,
    seed: int,
) -> Tuple[List[Any], List[Any]]:
    """Splits items into train/valid independently per class, so every class is
    represented in every split, instead of whole classes landing in a single split."""
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("--valid_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    train_items: List[Any] = []
    valid_items: List[Any] = []

    for class_label in sorted(class_to_patients):
        train, valid = _split_patient_group(class_to_patients[class_label], valid_fraction, rng)
        if not valid:
            print(
                f"Warning: class '{class_label}' has no samples in the validation split "
                f"(only {len(class_to_patients[class_label])} patient(s)/group(s) available)."
            )
        train_items.extend(train)
        valid_items.extend(valid)

    if not train_items or not valid_items:
        raise ValueError(
            "Failed to create non-empty train/valid splits. Check --valid_fraction, "
            "--patient_level (or --manifest_csv 'patient' column), and your class assignment."
        )

    return train_items, valid_items


def split_paths_by_class_and_patient(
    paths: List[Path],
    root: Path,
    valid_fraction: float,
    seed: int,
    patient_level: int,
    class_name: str,
    class_from_parent: bool,
    class_parent_level: int,
) -> Tuple[List[Path], List[Path]]:
    """Splits images into train/valid, stratified by class. Within a class, images
    sharing a patient id (--patient_level) are kept together to avoid leakage."""
    class_to_patients: Dict[str, Dict[str, List[Path]]] = {}
    for path in paths:
        class_label = get_class_label(path, class_name, class_from_parent, class_parent_level)
        patient_id = get_patient_id(path, root, patient_level)
        class_to_patients.setdefault(class_label, {}).setdefault(patient_id, []).append(path)

    return _stratified_split(class_to_patients, valid_fraction, seed)


def _source_path_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_entries(
    paths: List[Path],
    root: Path,
    image_size: int,
    class_name: str,
    class_from_parent: bool,
    class_parent_level: int,
    lazy: bool,
) -> List[dict]:
    entries = []
    for path in paths:
        class_label = get_class_label(path, class_name, class_from_parent, class_parent_level)
        entry = {
            "class": class_label,
            "name": path.stem,
            "metadata": {"source_path": _source_path_str(path, root)},
        }
        if not lazy:
            entry["image"] = load_rgb_chw(path, image_size)
        else:
            # Store an absolute path so lazy loading works regardless of where the
            # pickle ends up (e.g. Kaggle: images stay under a read-only
            # /kaggle/input mount while the pickle is written to /kaggle/working).
            entry["source_path"] = str(path)
        entries.append(entry)
    return entries


def _read_manifest(manifest_csv: Path, root: Path) -> List[dict]:
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "path" not in fieldnames or "class" not in fieldnames:
            raise ValueError("--manifest_csv must have at least 'path' and 'class' columns.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"--manifest_csv is empty: {manifest_csv}")

    entries = []
    for row in rows:
        raw_path = (row.get("path") or "").strip()
        if not raw_path:
            raise ValueError("--manifest_csv has a row with an empty 'path'.")
        raw_class = (row.get("class") or "").strip()
        if not raw_class:
            raise ValueError(f"--manifest_csv row for '{raw_path}' has an empty 'class'.")

        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        entries.append(
            {
                "path": path,
                "class": raw_class,
                "patient": (row.get("patient") or "").strip() or None,
                "split": (row.get("split") or "").strip() or None,
            }
        )
    return entries


def build_entries_from_manifest_rows(
    rows: List[dict], root: Path, image_size: int, lazy: bool
) -> List[dict]:
    entries = []
    for row in rows:
        path: Path = row["path"]
        entry = {
            "class": row["class"],
            "name": path.stem,
            "metadata": {"source_path": _source_path_str(path, root)},
        }
        if not lazy:
            entry["image"] = load_rgb_chw(path, image_size)
        else:
            entry["source_path"] = str(path)
        entries.append(entry)
    return entries


def build_dataset_from_manifest(
    manifest_csv: Path,
    root: Path,
    image_size: int,
    valid_fraction: float,
    seed: int,
    lazy: bool,
) -> Tuple[List[dict], List[dict]]:
    rows = _read_manifest(manifest_csv, root)

    has_split_column = any(row["split"] for row in rows)
    if has_split_column:
        missing = [row for row in rows if not row["split"]]
        if missing:
            raise ValueError(
                f"--manifest_csv has a 'split' column but {len(missing)} row(s) leave it empty; "
                "fill in every row or remove the column to use automatic splitting."
            )
        allowed = {"train", "valid"}
        invalid = sorted({row["split"] for row in rows} - allowed)
        if invalid:
            raise ValueError(
                f"--manifest_csv 'split' column has unsupported value(s) {invalid}; "
                f"use one of {sorted(allowed)}."
            )
        train_rows = [row for row in rows if row["split"] == "train"]
        valid_rows = [row for row in rows if row["split"] == "valid"]
        if not train_rows or not valid_rows:
            raise ValueError("--manifest_csv must include at least one 'train' and one 'valid' row.")
    else:
        class_to_patients: Dict[str, Dict[str, List[dict]]] = {}
        for row in rows:
            patient_id = row["patient"] or str(row["path"])
            class_to_patients.setdefault(row["class"], {}).setdefault(patient_id, []).append(row)
        train_rows, valid_rows = _stratified_split(class_to_patients, valid_fraction, seed)

    return (
        build_entries_from_manifest_rows(train_rows, root, image_size, lazy),
        build_entries_from_manifest_rows(valid_rows, root, image_size, lazy),
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input_dir is not a directory: {input_dir}")

    if args.manifest_csv:
        manifest_csv = Path(args.manifest_csv).expanduser().resolve()
        if not manifest_csv.is_file():
            raise FileNotFoundError(f"--manifest_csv does not exist: {manifest_csv}")
        train_entries, valid_entries = build_dataset_from_manifest(
            manifest_csv=manifest_csv,
            root=input_dir,
            image_size=args.image_size,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
            lazy=args.lazy,
        )
    else:
        paths = sorted(iter_image_paths(input_dir))
        if len(paths) < 2:
            raise ValueError(f"Need at least 2 images, found {len(paths)} in {input_dir}.")

        train_paths, valid_paths = split_paths_by_class_and_patient(
            paths,
            input_dir,
            args.valid_fraction,
            args.seed,
            args.patient_level,
            args.class_name,
            args.class_from_parent,
            args.class_parent_level,
        )
        train_entries = build_entries(
            train_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
            args.class_parent_level,
            args.lazy,
        )
        valid_entries = build_entries(
            valid_paths,
            input_dir,
            args.image_size,
            args.class_name,
            args.class_from_parent,
            args.class_parent_level,
            args.lazy,
        )

    dataset = {"train": train_entries, "valid": valid_entries}

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
