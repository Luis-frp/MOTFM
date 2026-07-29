import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a patient-level annotations CSV into a held-out test pool and a "
            "train/valid pool, stratified by class. The test pool is meant to be set "
            "aside entirely (never fed into build_manifest_from_annotations.py / "
            "prepare_rgb_dataset.py), so it stays free of leakage when later used to "
            "evaluate a classifier trained on real+generated images."
        )
    )
    parser.add_argument(
        "--annotations_csv",
        required=True,
        type=str,
        help="Original patient-level annotations CSV (one row per patient).",
    )
    parser.add_argument(
        "--class_col",
        default="Gastritis_type",
        type=str,
        help="Column used to stratify the split (default: Gastritis_type).",
    )
    parser.add_argument(
        "--test_fraction",
        default=0.2,
        type=float,
        help="Fraction of patients per class held out for the test pool (default: 0.2).",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random split seed.")
    parser.add_argument(
        "--output_train_csv",
        default="data/annotations_train_pool.csv",
        type=str,
        help="Where to write the remaining patients (feed this into build_manifest_from_annotations.py).",
    )
    parser.add_argument(
        "--output_test_csv",
        default="data/annotations_test.csv",
        type=str,
        help="Where to write the held-out test patients.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test_fraction must be between 0 and 1.")

    annotations_csv = Path(args.annotations_csv).expanduser().resolve()
    with annotations_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if args.class_col not in fieldnames:
            raise ValueError(f"--class_col '{args.class_col}' not found. Available: {fieldnames}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"--annotations_csv is empty: {annotations_csv}")

    class_to_rows: Dict[str, List[dict]] = {}
    for row in rows:
        class_to_rows.setdefault(row[args.class_col].strip(), []).append(row)

    rng = random.Random(args.seed)
    train_rows: List[dict] = []
    test_rows: List[dict] = []

    for class_label in sorted(class_to_rows):
        class_rows = list(class_to_rows[class_label])
        rng.shuffle(class_rows)
        n = len(class_rows)

        if n <= 1:
            test_target = 0
        else:
            test_target = max(1, int(round(n * args.test_fraction)))
            test_target = min(test_target, n - 1)

        class_test = class_rows[:test_target]
        class_train = class_rows[test_target:]

        if not class_test:
            print(f"Warning: class '{class_label}' has no patients in the test pool ({n} patient(s) total).")
        train_rows.extend(class_train)
        test_rows.extend(class_test)

    if not train_rows or not test_rows:
        raise ValueError(
            "Failed to create non-empty train-pool and test pools. Check --test_fraction "
            "and the number of patients per class."
        )

    fieldnames = list(rows[0].keys())
    output_train_csv = Path(args.output_train_csv).expanduser().resolve()
    output_test_csv = Path(args.output_test_csv).expanduser().resolve()
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    output_test_csv.parent.mkdir(parents=True, exist_ok=True)

    for output_path, out_rows in ((output_train_csv, train_rows), (output_test_csv, test_rows)):
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    print(f"Train/valid pool: {len(train_rows)} patients -> {output_train_csv}")
    print(f"Held-out test pool: {len(test_rows)} patients -> {output_test_csv}")


if __name__ == "__main__":
    main()
