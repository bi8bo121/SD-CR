# SD-CR

**Structure–Degradation Co-Reasoning for Infrared–Visible Image Fusion under Asymmetric Degradation**

Official implementation of **SD-CR** for infrared–visible image fusion under asymmetric degradation.

This repository provides a three-stage training pipeline and inference and evaluation code:

- **Stage 1 — Visible image restoration:** train the visible image restoration network.
- **Stage 2 — Image fusion:** initialize from the Stage 1 checkpoint and train the fusion branch.
- **Stage 3 — Fine-tuning:** continue from the Stage 2 checkpoint using a staged freezing and fine-tuning schedule.

The `test.py` script generates fused images and computes evaluation metrics using a Stage 3 checkpoint.

[Installation](#installation) · [Data Preparation](#data-preparation) · [Checkpoints](#checkpoints) · [Training](#training) · [Inference and Evaluation](#inference-and-evaluation)

## Repository Structure

| File or directory | Description |
| --- | --- |
| `stage1.py` | Stage 1: visible image restoration training |
| `stage2.py` | Stage 2: fusion training |
| `stage3.py` | Stage 3: staged fine-tuning |
| `test.py` | Inference and fusion metric evaluation using a Stage 3 checkpoint |
| `models/` | Model architectures and supporting modules |
| `data/` | Dataset loading and training augmentations |
| `losses/` | Stage 1 loss functions |
| `trainers/` | Stage 2 and Stage 3 fusion loss functions |
| `utils/` | Training and validation metrics for Stages 2 and 3 |
| `configs/` | Example dataset configuration files |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |

Run all commands below from the project root directory.

## Installation

Python **3.9 or later** is required. Create a dedicated environment:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

Install PyTorch and torchvision for your CPU or CUDA platform, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency list is based on the packages imported by the code. Exact dependency versions have not been pinned to a verified environment.

| Package | Purpose |
| --- | --- |
| `piq` | Required by Stage 1 for `piq.brisque` |
| `cpbd` | Optional CPBD evaluation in `test.py` |
| `pyiqa` | Optional MUSIQ and BRISQUE evaluation when `--use_pyiqa` is enabled |
| `openpyxl` | Optional Excel export |

Install the optional evaluation and export packages as needed:

```bash
python -m pip install cpbd pyiqa openpyxl
```

If an optional metric dependency is unavailable, the corresponding metric may be recorded as `NaN`. Without Excel support, evaluation results may be exported only as CSV files.

The multiline commands below use Bash line continuation (`\`). On Windows, run them in a Bash-compatible terminal or combine each command into a single line.

## Data Preparation

The loader in `data/dataset_v2.py` reads **pre-generated degraded images**. A complete degradation generation pipeline is not included in this repository.

Training uses the following visible-image degradation types:

| Directory name | Degradation |
| --- | --- |
| `VI_Blur` | Blur |
| `VI_Haze` | Haze |
| `VI_Low_light` | Low light |
| `VI_Noise` | Noise |
| `VI_Over_exposure` | Overexposure |

Each degradation type uses four intensity directory names: `slight`, `moderate`, `average`, and `extreme`. Keep these names exactly as shown for compatibility with the loader.

For example, organize the slight blur subset under `datasets/DDL/VI_Blur/VI_Blur_slight/`:

| Subdirectory | Contents |
| --- | --- |
| `train/Infrared/` | Grayscale infrared images |
| `train/Visible/` | Degraded RGB visible images |
| `train/Visible_gt/` | Clean RGB visible reference images |
| `test/Infrared/` | Infrared images used for validation during training |
| `test/Visible/` | Degraded visible images used for validation during training |

Use the same structure for the remaining degradation types and intensities. The alternative layout `VI_Blur/slight/{train,test}/...` is also supported.

- Paired infrared, visible, and clean visible images must have matching filename stems.
- During training, images are resized to **320 × 320** and randomly cropped to **256 × 256**.
- Validation during training reads from `test/`; this split does not contain `Visible_gt/`.
- Datasets and paired example images are not bundled with the repository.

## Checkpoints

The shared **`ckpt`** folder can be accessed through Baidu Netdisk:

**[Download from Baidu Netdisk](https://pan.baidu.com/s/1peWQgwwDZ91zKZOTU_ZCaw?pwd=1jpv)**  
**Extraction code:** `1jpv`

Checkpoints are distributed separately from the source repository. The commands in this README use the following checkpoint filenames and locations. Place the required files at these paths, or pass their actual locations using the corresponding command-line arguments.

| Checkpoint | Path used in the examples | Purpose |
| --- | --- | --- |
| Stage 1, epoch 150 | `checkpoints_l1_balanced_5types/restoration_epoch_150.pt` | Restoration initialization for Stages 2 and 3 |
| Stage 2, epoch 120, full checkpoint | `stage2_results_fresh/stage2_ep120_full.pth` | Initialization for Stage 3 |
| Stage 3, epoch 27 | `stage3_results/stage3_ep027.pth` | Inference and evaluation with `test.py` |

**For inference only, a complete Stage 3 checkpoint is sufficient.** You do not need to provide a separate Stage 1 checkpoint to `test.py`.

## Training

### Stage 1: Visible Image Restoration

```bash
python stage1.py --data-root datasets/DDL
```

By default, Stage 1 trains for **150 epochs** and saves model weights every **5 epochs**.

- Checkpoint directory: `checkpoints_l1_balanced_5types/`
- Validation image directory: `val_results_l1_balanced_5types/`

### Stage 2: Image Fusion

Initialize from the Stage 1 restoration checkpoint:

```bash
python stage2.py --data-root datasets/DDL \
  --stage1-ckpt checkpoints_l1_balanced_5types/restoration_epoch_150.pt
```

Stage 2 trains for **150 epochs** by default. Model weights and full training checkpoints are saved to `stage2_results_fresh/`.

### Stage 3: Fine-tuning

Provide the Stage 1 checkpoint and the Stage 2 full checkpoint:

```bash
python stage3.py --data-root datasets/DDL \
  --stage1-ckpt checkpoints_l1_balanced_5types/restoration_epoch_150.pt \
  --stage2-ckpt stage2_results_fresh/stage2_ep120_full.pth
```

Stage 3 trains for **30 epochs** by default. The initial freezing schedule is applied for the first **8 epochs**, followed by further fine-tuning. Outputs are saved to `stage3_results/`.

The Stage 3 command above initializes from **Stage 2 epoch 120**, while Stage 2's default training duration is 150 epochs. These are separate settings; retain the specified initialization checkpoint when following this configuration.

### Device Selection and Resuming Training

All three training scripts accept `--device cpu` or `--device cuda:0`.

To resume **Stage 1 or Stage 2** from a full training checkpoint, explicitly add `--resume <checkpoint_path>` to the relevant command.

## Inference and Evaluation

### Single Dataset

Prepare your own paired infrared and visible images under a dataset root, for example:

| Directory | Contents |
| --- | --- |
| `datasets/example/ir/` | Infrared input images |
| `datasets/example/vis/` | Visible input images with matching filenames |

The `datasets/example/` directory is an illustration and is not included in this repository.

Run inference and evaluation:

```bash
python test.py \
  --stage3_ckpt stage3_results/stage3_ep027.pth \
  --dataset_root datasets/example \
  --ir_dir ir --vis_dir vis \
  --dataset_name example \
  --results_root outputs/evaluation
```

The script pairs infrared and visible images by filename and saves fused images to:

```text
outputs/evaluation/fused_stage3_ep027/example/
```

It also writes the following evaluation files:

| File | Contents |
| --- | --- |
| `all_details.csv` | Per-image metric results |
| `all_summary.csv` | Summary metric results |
| `run_config.json` | Configuration used for the evaluation run |

Without clean visible reference images, the script computes **EN, SD, SF, and AG**, along with optional quality metrics:

- **CPBD**, when its dependency is available.
- **MUSIQ and BRISQUE**, when `pyiqa` is installed and `--use_pyiqa` is supplied.

### Evaluation with Clean Visible References

If the dataset also contains clean visible images in `Visible_gt/`, add the following argument to the single-dataset command:

```bash
--gt_dir Visible_gt
```

The script then matches infrared, visible, and clean visible images by filename and computes additional metrics.

### Multiple Datasets

Edit `configs/inference_datasets.example.json` to point to your actual dataset locations, then run:

```bash
python test.py \
  --stage3_ckpt stage3_results/stage3_ep027.pth \
  --dataset_config configs/inference_datasets.example.json \
  --results_root outputs/evaluation
```

Use **exactly one** of `--dataset_root` and `--dataset_config`.

The evaluation script saves fused images only. The model's forward output also includes `restored_image`; saving restoration outputs requires adapting the inference code.

## Implementation Notes

- **Restoration output:** Stage 1 adds the decoder output to the degraded visible input. In Stages 2 and 3, `restored_image` is taken directly from the decoder output. This difference is part of the current implementation.
- **Dependencies:** Exact dependency versions are not pinned to a verified environment; use a compatible PyTorch and torchvision installation for your platform.
- **Release validation:** The validation reported during code preparation covered imports, checkpoint loading, and inference on one real image. Long-duration training and full-dataset evaluation were not rerun as part of that preparation.
- **License:** No project-level license file is currently included in this release.
