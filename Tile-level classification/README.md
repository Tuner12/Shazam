
````markdown
# Tile-level Classification Pipeline

This document describes the complete **Tile-level Classification** pipeline, including **feature extraction**, **single-model training**, and **multi-model training**, with **environment-variable–based authentication** for Hugging Face models.

---

## Directory Structure

```text
Tile-level classification/
├── feature_extract4sample.py   # Feature extraction script
├── single_model.py             # Single-model training script
├── multi_model.py              # Multi-model training script
├── README.md                 # This document
````

---

## 0. Environment Setup 

Some models (e.g. `UNI-v2`, `GigaPath`, `Virchow`, `MUSK`) are loaded from **Hugging Face Hub** and require authentication.


#### Step 1: Export Hugging Face token

```bash
export HF_TOKEN=hf_your_huggingface_token_here
```



## 1. Feature Extraction

### File Path

```text
feature_extract4sample.py
```

### Purpose

This script extracts **multi-level features** (early / middle / high) from tile images using different foundation models, including:

* `uni_v2`
* `virchow2`
* `phikon_v2`
* `gigapath`
* `hoptimus1`

Extracted features are saved as `.pt` files for downstream training.

---

### Usage

```bash
python feature_extract4sample.py
```

Before running, make sure:

* `HF_TOKEN` is exported (see Section 0)
* Dataset paths are correctly set in the script

---

### Output

For each model `<model_name>`:

```text
<model_name>_train_features.pt
<model_name>_val_features.pt
<model_name>_test_features.pt
```

Each `.pt` file contains:

```python
(low_level_features, mid_level_features, high_level_features, labels)
```

---

## 2. Single-Model Training

### File Path

```text
single_model.py
```

### Purpose

Train a **single classifier** using features extracted from one backbone model.

---

### Usage

```bash
python single_model.py
```

---

### Output

```text
single_model.pth            # Trained model weights
single_model_results.json   # Training / validation metrics
```

---

## 3. Multi-Model Training

### File Path

```text
multi_model.py
```

### Purpose

Train a **fusion model** that combines features from multiple backbone models (e.g. UNI + Virchow + Phikon).

---

### Usage

```bash
python multi_model.py
```

---

### Output

```text
multi_model.pth
multi_model_results.json
```

---

## Notes

### 1. Data Paths

Modify dataset paths in scripts according to your local or cluster environment.

---

### 2. GPU Acceleration

* GPU is used by default
* Ensure CUDA is available
* Optionally control GPUs via:

```bash
export CUDA_VISIBLE_DEVICES=0
```



## Summary of the Pipeline

1. Export environment variables

```bash
export HF_TOKEN=...
```

2. Extract features

```bash
python feature_extract4sample.py
```

3. Train single model

```bash
python single_model.py
```

4. Train multi-model fusion

```bash
python multi_model.py
```

---


