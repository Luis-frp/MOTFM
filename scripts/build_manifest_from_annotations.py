import argparse
import csv
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prepare_rgb_dataset.py --manifest_csv (path,class,patient) from a "
            "patient-level annotations CSV, matching each patient id to its image folder "
            "under --input_dir."
        )
    )
    parser.add_argument(
        "--annotations_csv",
        required=True,
        type=str,
        help="CSV with one row per patient, containing at least the patient id and class columns.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Root folder containing the image files (patient folders may be nested under class folders).",
    )
    parser.add_argument(
        "--output_csv",
        default="data/manifest.csv",
        type=str,
        help="Where to write the resulting manifest CSV.",
    )
    parser.add_argument(
        "--patient_col",
        default="Patient_ID",
        type=str,
        help="Column name in --annotations_csv holding the patient id (default: Patient_ID).",
    )
    parser.add_argument(
        "--class_col",
        default="Gastritis_type",
        type=str,
        help="Column name in --annotations_csv holding the class label (default: Gastritis_type).",
    )
    return parser.parse_args()


def find_patient_dir(root: Path, patient_id: str) -> Path:
    candidates = [p for p in root.rglob(patient_id) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No folder named '{patient_id}' found under {root}. Check that --patient_col "
            "values match the actual patient folder names on disk."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found {len(candidates)} folders named '{patient_id}' under {root}: "
            f"{[str(c) for c in candidates]}. Patient folder names must be unique under --input_dir."
        )
    return candidates[0]


def list_images(patient_dir: Path) -> List[Path]:
    return sorted(
        p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--input_dir is not a directory: {root}")

    annotations_csv = Path(args.annotations_csv).expanduser().resolve()
    with annotations_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if args.patient_col not in fieldnames:
            raise ValueError(f"--patient_col '{args.patient_col}' not found. Available: {fieldnames}")
        if args.class_col not in fieldnames:
            raise ValueError(f"--class_col '{args.class_col}' not found. Available: {fieldnames}")
        rows = list(reader)

    manifest_rows = []
    for row in rows:
        patient_id = row[args.patient_col].strip()
        class_label = row[args.class_col].strip()
        if not patient_id or not class_label:
            print(f"Warning: skipping row with empty patient/class: {row}")
            continue

        patient_dir = find_patient_dir(root, patient_id)
        images = list_images(patient_dir)
        if not images:
            print(f"Warning: no images found for patient '{patient_id}' in {patient_dir}")
            continue

        for image_path in images:
            manifest_rows.append(
                {
                    "path": str(image_path.relative_to(root)),
                    "class": class_label,
                    "patient": patient_id,
                }
            )

    if not manifest_rows:
        raise ValueError("No image rows were produced; check --annotations_csv and --input_dir.")

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "patient"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    num_patients = len({row["patient"] for row in manifest_rows})
    print(f"Wrote {len(manifest_rows)} image rows for {num_patients} patients to {output_csv}")


if __name__ == "__main__":
    main()
