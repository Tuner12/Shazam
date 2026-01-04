# SHAZAM Gene Expression Prediction Pipeline

This repository contains scripts for multi-teacher knowledge distillation-based gene expression prediction from histopathology images. The pipeline consists of two main components: feature extraction and cross-validation training.

## Overview

The SHAZAM pipeline uses multiple pre-trained vision transformer models (teachers) to extract features from histopathology images, then trains a student model using knowledge distillation to predict gene expression levels.

### Pipeline Flow

1. **Feature Extraction** (`feature_extraction_st_prediction_shazam.py`): Extracts multi-level features from histopathology images using multiple teacher models
2. **Cross-Validation Training** (`shazam_st_prediction.py`): Trains a student model using patient-level K-fold cross-validation with knowledge distillation


## Script 1: Feature Extraction

### `feature_extraction_st_prediction_shazam.py`

Extracts multi-level features (early, middle, high) from histopathology images using multiple pre-trained vision transformer models.

#### Supported Models

- `hoptimus1`: H-Optimus-1 model
- `gigapath`: GigaPath model
- `virchow2`: Virchow2 model
- `phikon_v2`: Phikon v2 model
- `uni_v2`: UNI2-h model
- `musk`: Musk model

#### Input Structure

```
root-dir/
├── patient_1/
│   ├── metadata.jsonl
│   └── image_files/
├── patient_2/
│   ├── metadata.jsonl
│   └── image_files/
└── ...
```

#### Output Structure

```
output-base/
├── patient_1/
│   ├── hoptimus1_features.pt
│   ├── gigapath_features.pt
│   ├── virchow2_features.pt
│   ├── phikon_v2_features.pt
│   └── uni_v2_features.pt
├── patient_2/
│   └── ...
└── ...
```

Each `*_features.pt` file contains a tuple: `(low_features, mid_features, high_features, labels)`

#### Usage

```bash
python feature_extraction_st_prediction_shazam.py \
    --root-dir /path/to/patient/data/ \
    --metadata-name metadata.jsonl \
    --models hoptimus1 gigapath virchow2 phikon_v2 uni_v2 \
    --output-base /path/to/output/features/ \
    --device cuda:0 \
    --batch-size 16 \
    --num-workers 4
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--root-dir` | Yes | - | Root directory containing patient subdirectories |
| `--metadata-name` | No | `metadata.jsonl` | Name of metadata file in each subdirectory |
| `--models` | Yes | - | Space-separated list of model names to use |
| `--output-base` | Yes | - | Root directory for saving extracted features |
| `--device` | No | `cpu` | Device to use (e.g., `cuda:0`, `cuda:1`) |
| `--batch-size` | No | `16` | Batch size for feature extraction |
| `--num-workers` | No | `4` | Number of data loading workers |


#### Authentication

Some model weights are retrieved from the Hugging Face Hub and may require authentication. Do NOT hardcode tokens in the source. Instead, set an environment variable before running the script, for example:

```bash
export HUGGINGFACE_TOKEN=hf_xxxYOUR_TOKEN_xxx
python feature_extraction_st_prediction_shazam.py \
    --root-dir /path/to/patient/data/ \
    --models hoptimus1 gigapath virchow2 phikon_v2 uni_v2 \
    --output-base /path/to/output/features/ \
    --device cuda:0
```

The code reads the token from `HUGGINGFACE_TOKEN` (fallbacks: `HF_TOKEN` or `HF_HUB_TOKEN`) and uses it for authentication with the Hugging Face Hub. This prevents accidental leakage of secrets in commits.

## Script 2: Cross-Validation Training

### `shazam_st_prediction.py`

Performs patient-level K-fold cross-validation training of a student model using knowledge distillation from multiple teacher models.

#### Features

- **Patient-level K-fold CV**: Automatically determines fold count (5 folds if >5 patients, otherwise patient count)
- **Multi-teacher Knowledge Distillation**: Combines features from multiple teacher models
- **Multi-GPU Support**: Automatic DataParallel for multi-GPU training
- **Best Model Selection**: Saves model with best Pearson correlation for each fold

#### Model Architecture

The student model (`StudentModel15Layers`) consists of:
- **MoE (Mixture of Experts) layers**: For each feature level (low, mid, high)
- **Multi-layer Cross-Attention**: For feature fusion
- **15-layer deep network**: With residual connections
- **Gene expression prediction head**: Outputs predictions for all genes

#### Input Structure

```
feature-root/
├── patient_1/
│   ├── hoptimus1_features.pt
│   ├── gigapath_features.pt
│   ├── virchow2_features.pt
│   ├── phikon_v2_features.pt
│   └── uni_v2_features.pt
├── patient_2/
│   └── ...
└── ...
```

#### Output Structure

```
output-dir/
├── fold1/
│   └── fold1_best.pth
├── fold2/
│   └── fold2_best.pth
├── ...
├── fold1_results.json
├── fold2_results.json
├── ...
└── cv_summary.json
```

Each `fold*_results.json` contains:
```json
{
  "fold": 1,
  "pearson": 0.85,
  "mse": 0.023,
  "preds": [...],
  "labels": [...]
}
```

#### Usage

**Single GPU:**
```bash
python shazam_st_prediction.py \
    --feature-root /path/to/features/ \
    --models hoptimus1 gigapath virchow2 phikon_v2 uni_v2 \
    --output-dir /path/to/output/ \
    --device cuda:0 \
    --batch-size 64 \
    --num-epochs 50 \
    --seed 42
```

**Multi-GPU:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python shazam_st_prediction.py \
    --feature-root /path/to/features/ \
    --models hoptimus1 gigapath virchow2 phikon_v2 uni_v2 \
    --output-dir /path/to/output/ \
    --device cuda:0 \
    --batch-size 64 \
    --num-epochs 50 \
    --seed 42
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--feature-root` | Yes | - | Root directory containing feature files |
| `--models` | Yes | - | Space-separated list of model names (must match feature extraction) |
| `--output-dir` | Yes | - | Output directory for trained models and results |
| `--device` | No | `cpu` | Main device (e.g., `cuda:0`). For multi-GPU, set `CUDA_VISIBLE_DEVICES` |
| `--batch-size` | No | `64` | Batch size for training |
| `--num-epochs` | No | `50` | Number of training epochs per fold |
| `--seed` | No | `42` | Random seed for reproducibility |

#### Multi-GPU Training

1. Set `CUDA_VISIBLE_DEVICES` environment variable:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ```

2. Specify main GPU with `--device cuda:0`

3. The script automatically enables `nn.DataParallel` when multiple GPUs are detected



