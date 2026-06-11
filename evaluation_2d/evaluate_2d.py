import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D metrics between a generated dataset and a reference dataset."
    )
    parser.add_argument(
        "--generated_path",
        type=str,
        required=True,
        help="Path to generated dataset pickle.",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help=(
            "Path to reference dataset pickle. If omitted, --generated_path is used and reference "
            "images are read from `true_data` when available."
        ),
    )
    parser.add_argument(
        "--generated_split",
        type=str,
        default="train",
        help="Split key in generated dataset (default: train).",
    )
    parser.add_argument(
        "--reference_split",
        type=str,
        default="valid",
        help="Split key in reference dataset (default: valid).",
    )
    parser.add_argument(
        "--generated_image_key",
        type=str,
        default="image",
        help="Image key for generated samples (default: image). Use 'auto' to try common keys.",
    )
    parser.add_argument(
        "--reference_image_key",
        type=str,
        default="auto",
        help="Image key for reference samples (default: auto, tries true_data/reference_image/image).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: min(len(generated), len(reference))).",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["per_set_minmax", "shared_minmax", "none"],
        default="per_set_minmax",
        help="Normalization before metric computation (default: per_set_minmax).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for metric computation (default: auto).",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=16,
        help="Batch size for InceptionV3 feature extraction used by 2D FID.",
    )
    parser.add_argument(
        "--ms_ssim_weights",
        type=str,
        default="0.1,0.3,0.6",
        help="Comma-separated weights for MS-SSIM levels (default: 0.1,0.3,0.6).",
    )
    parser.add_argument(
        "--ms_ssim_kernel_size",
        type=int,
        default=5,
        help="Kernel size for MS-SSIM (default: 5).",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip 2D FID computation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save computed metrics as JSON.",
    )
    return parser.parse_args()


def _parse_weights(weights_csv: str) -> Tuple[float, ...]:
    try:
        weights = tuple(float(x.strip()) for x in weights_csv.split(",") if x.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --ms_ssim_weights='{weights_csv}'.") from exc
    if not weights:
        raise ValueError("MS-SSIM weights cannot be empty.")
    return weights


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def _load_pickle(path: Path) -> Dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(data).__name__}.")
    return data


def _resolve_split(data: Dict, split_name: str, label: str, path: Path) -> List[dict]:
    if split_name not in data:
        available = [k for k, v in data.items() if isinstance(v, list)]
        raise ValueError(f"{label} split '{split_name}' not found in {path}. Available: {available}")
    split = data[split_name]
    if not isinstance(split, list):
        raise ValueError(f"{label} split '{split_name}' must be a list, got {type(split).__name__}.")
    if len(split) == 0:
        raise ValueError(f"{label} split '{split_name}' is empty.")
    return split


def _ensure_channel_first_2d(tensor: torch.Tensor, *, source: str) -> torch.Tensor:
    x = torch.as_tensor(tensor, dtype=torch.float32).squeeze()
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"{source}: expected 2D/3D image, got shape {tuple(x.shape)}.")

    if x.shape[0] <= 4:
        return x
    if x.shape[-1] <= 4:
        return x.permute(2, 0, 1).contiguous()
    raise ValueError(
        f"{source}: cannot infer channel dimension from shape {tuple(x.shape)}. "
        "Expected [C,H,W], [H,W,C], or [H,W]."
    )


def _extract_image(entry: dict, *, source: str, image_key: str, reference: bool) -> torch.Tensor:
    if image_key != "auto":
        if image_key not in entry:
            raise KeyError(f"{source}: expected key '{image_key}'.")
        return _ensure_channel_first_2d(entry[image_key], source=f"{source}['{image_key}']")

    key_candidates = (
        ("true_data", "reference_image", "image")
        if reference
        else ("image", "generated_image", "true_data")
    )
    for key in key_candidates:
        if key in entry:
            return _ensure_channel_first_2d(entry[key], source=f"{source}['{key}']")
    raise KeyError(f"{source}: expected one of keys {list(key_candidates)}.")


def _pair_images(
    generated_entries: Sequence[dict],
    reference_entries: Sequence[dict],
    num_samples: Optional[int],
    generated_image_key: str,
    reference_image_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_pairs = min(len(generated_entries), len(reference_entries))
    if max_pairs <= 0:
        raise ValueError("No overlapping samples between generated and reference splits.")

    n = max_pairs if num_samples is None else min(int(num_samples), max_pairs)
    if n <= 0:
        raise ValueError(f"Invalid requested sample count: {num_samples}.")

    true_images: List[torch.Tensor] = []
    generated_images: List[torch.Tensor] = []

    for i in range(n):
        gen_image = _extract_image(
            generated_entries[i],
            source=f"generated[{i}]",
            image_key=generated_image_key,
            reference=False,
        )
        ref_image = _extract_image(
            reference_entries[i],
            source=f"reference[{i}]",
            image_key=reference_image_key,
            reference=True,
        )
        if gen_image.shape != ref_image.shape:
            raise ValueError(
                f"Shape mismatch at sample {i}: generated {tuple(gen_image.shape)} "
                f"vs reference {tuple(ref_image.shape)}."
            )
        generated_images.append(gen_image)
        true_images.append(ref_image)

    true_stack = torch.stack(true_images, dim=0)
    gen_stack = torch.stack(generated_images, dim=0)
    return true_stack, gen_stack


def _minmax(tensor: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_min = torch.amin(x)
    x_max = torch.amax(x)
    denom = (x_max - x_min).clamp_min(1e-8)
    return (x - x_min) / denom


def _normalize_pair(
    true_images: torch.Tensor,
    generated_images: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "none":
        return (
            torch.nan_to_num(true_images.float(), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(generated_images.float(), nan=0.0, posinf=0.0, neginf=0.0),
        )

    if mode == "per_set_minmax":
        return _minmax(true_images), _minmax(generated_images)

    if mode == "shared_minmax":
        true_images = torch.nan_to_num(true_images.float(), nan=0.0, posinf=0.0, neginf=0.0)
        generated_images = torch.nan_to_num(
            generated_images.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_min = torch.minimum(torch.amin(true_images), torch.amin(generated_images))
        x_max = torch.maximum(torch.amax(true_images), torch.amax(generated_images))
        denom = (x_max - x_min).clamp_min(1e-8)
        return (true_images - x_min) / denom, (generated_images - x_min) / denom

    raise ValueError(f"Unsupported normalization mode '{mode}'.")


def _to_three_channels(images: torch.Tensor) -> torch.Tensor:
    c = int(images.shape[1])
    if c == 3:
        return images
    if c < 3:
        repeats = (3 + c - 1) // c
        return images.repeat(1, repeats, 1, 1)[:, :3]
    return images[:, :3]


def _load_inception_backbone(device: torch.device) -> torch.nn.Module:
    try:
        import torchvision
    except Exception as exc:
        raise ImportError(
            "torchvision is required for 2D FID. Install torchvision or use --skip_fid."
        ) from exc

    try:
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained inception_v3 weights for 2D FID. Use --skip_fid."
        ) from exc

    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _extract_inception_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[i : i + batch_size].to(device)
        batch = _to_three_channels(batch).clamp(0.0, 1.0)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - imagenet_mean) / imagenet_std
        feats.append(model(batch).detach().cpu())
    return torch.cat(feats, dim=0)


def compute_metrics(
    true_images: torch.Tensor,
    generated_images: torch.Tensor,
    *,
    ms_ssim_weights: Tuple[float, ...],
    ms_ssim_kernel_size: int,
    device: torch.device,
    fid_batch_size: int,
    skip_fid: bool,
) -> Dict[str, Union[float, None]]:
    try:
        from monai.metrics import FIDMetric, compute_mmd, compute_ms_ssim, compute_ssim_and_cs
    except Exception as exc:
        raise ImportError(
            "MONAI metrics are required. Install monai (or ensure monai_generative installed MONAI)."
        ) from exc

    true_images = true_images.to(device)
    generated_images = generated_images.to(device)

    mmd_value = compute_mmd(true_images, generated_images, None)
    try:
        ms_ssim_value = compute_ms_ssim(
            true_images,
            generated_images,
            2,
            weights=ms_ssim_weights,
            kernel_size=int(ms_ssim_kernel_size),
        )
        ssim_value, _ = compute_ssim_and_cs(
            true_images,
            generated_images,
            2,
            kernel_size=(int(ms_ssim_kernel_size), int(ms_ssim_kernel_size)),
            kernel_sigma=(1.5, 1.5),
        )
    except ValueError as exc:
        raise ValueError(
            "SSIM/MS-SSIM failed. If images are small, use fewer --ms_ssim_weights levels "
            "or smaller --ms_ssim_kernel_size."
        ) from exc

    mse_value = torch.mean((true_images - generated_images) ** 2)
    psnr_value = 10.0 * torch.log10(1.0 / mse_value.clamp_min(1e-12))

    metrics: Dict[str, Union[float, None]] = {
        "mmd": float(mmd_value.item()),
        "ms_ssim": float(ms_ssim_value.mean().item()),
        "ssim": float(ssim_value.mean().item()),
        "mse": float(mse_value.item()),
        "psnr": float(psnr_value.item()),
        "fid": None,
    }

    if skip_fid:
        return metrics

    fid_metric = FIDMetric()
    inception_model = _load_inception_backbone(device)
    real_features = _extract_inception_features(
        inception_model, true_images, device, fid_batch_size
    )
    gen_features = _extract_inception_features(
        inception_model, generated_images, device, fid_batch_size
    )
    fid_score = fid_metric(real_features, gen_features)
    metrics["fid"] = float(fid_score.item())
    return metrics


def main() -> None:
    args = parse_args()
    weights = _parse_weights(args.ms_ssim_weights)
    device = _resolve_device(args.device)

    generated_path = Path(args.generated_path).expanduser().resolve()
    reference_path = (
        generated_path
        if args.reference_path is None
        else Path(args.reference_path).expanduser().resolve()
    )
    if not generated_path.exists():
        raise FileNotFoundError(f"--generated_path does not exist: {generated_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"--reference_path does not exist: {reference_path}")

    generated_data = _load_pickle(generated_path)
    reference_data = _load_pickle(reference_path)
    generated_entries = _resolve_split(
        generated_data, args.generated_split, label="Generated", path=generated_path
    )
    reference_entries = _resolve_split(
        reference_data, args.reference_split, label="Reference", path=reference_path
    )

    true_images, generated_images = _pair_images(
        generated_entries,
        reference_entries,
        args.num_samples,
        args.generated_image_key,
        args.reference_image_key,
    )
    true_images, generated_images = _normalize_pair(true_images, generated_images, args.normalization)

    metrics = compute_metrics(
        true_images,
        generated_images,
        ms_ssim_weights=weights,
        ms_ssim_kernel_size=args.ms_ssim_kernel_size,
        device=device,
        fid_batch_size=args.fid_batch_size,
        skip_fid=args.skip_fid,
    )

    print(f"Generated: {generated_path} [{args.generated_split}]")
    print(f"Reference: {reference_path} [{args.reference_split}]")
    print(f"Samples used: {int(generated_images.shape[0])}")
    print(f"MMD: {metrics['mmd']:.6f}")
    print(f"MS-SSIM: {metrics['ms_ssim']:.6f}")
    print(f"SSIM: {metrics['ssim']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.6f}")
    if metrics["fid"] is None:
        print("FID: skipped")
    else:
        print(f"FID: {metrics['fid']:.6f}")

    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_path": str(generated_path),
            "reference_path": str(reference_path),
            "generated_split": args.generated_split,
            "reference_split": args.reference_split,
            "samples_used": int(generated_images.shape[0]),
            "metrics": metrics,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
