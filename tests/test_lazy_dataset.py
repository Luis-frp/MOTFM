import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from utils.general_utils import load_and_prepare_data


def test_load_and_prepare_data_supports_lazy_manifest(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)

    manifest = {
        "train": [
            {
                "class": "Atrophy",
                "name": "sample",
                "source_path": str(image_path),
            }
        ]
    }
    pickle_path = tmp_path / "dataset.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(manifest, f)

    data = load_and_prepare_data(
        str(pickle_path),
        split="train",
        image_norm="none",
        lazy=True,
        image_size=8,
    )

    assert data["dataset"] is not None
    assert len(data["dataset"]) == 1
    sample = data["dataset"][0]
    assert sample["images"].shape == (3, 8, 8)
