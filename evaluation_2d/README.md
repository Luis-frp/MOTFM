# 2D Evaluation

This folder contains a script to compute 2D metrics between a generated dataset and a reference dataset.

## Metrics

The script computes:

- `MMD` (MONAI `compute_mmd`)
- `MS-SSIM` (MONAI `compute_ms_ssim`)
- `SSIM` (MONAI `compute_ssim_and_cs`)
- `MSE`
- `PSNR`
- `FID` for 2D images using `torchvision.models.inception_v3` features + MONAI `FIDMetric`

## Usage

```bash
python evaluation_2d/evaluate_2d.py \
  --generated_path /path/to/generated.pkl \
  --reference_path /path/to/reference.pkl \
  --generated_split train \
  --reference_split valid \
  --num_samples 200
```

If inference was run with `--include_reference`, the generated pickle already contains both images:

```bash
python evaluation_2d/evaluate_2d.py \
  --generated_path /path/to/generated_with_reference.pkl \
  --generated_split valid \
  --reference_split valid \
  --num_samples 200
```

Input expectations:

- Both inputs are `.pkl` files with list-like splits (`train`, `valid`, etc.).
- Generated samples should contain `image`.
- Reference samples should contain `image` or `true_data`.
- Images are expected as `[C, H, W]`, `[H, W, C]`, or `[H, W]`.

Notes:

- Use `--skip_fid` to skip 2D FID if `torchvision` or pretrained InceptionV3 weights are unavailable.
- Use `--output_json results.json` to save the computed metrics.
- Normalization options before metric computation:
  - `per_set_minmax` (default)
  - `shared_minmax`
  - `none`
