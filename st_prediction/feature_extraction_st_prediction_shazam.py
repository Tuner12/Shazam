#!/usr/bin/env python3
"""
Batch feature extraction for each subdirectory under the specified root directory.

Each subdirectory should contain:
- metadata.jsonl (filename can be specified)
- Corresponding image files

Output results are saved under the specified output root directory with the following structure:
    <output-base>/
        <subdirectory-name>/
            <model-name>_features.pt

Usage example:
    python feature_extract4shazamv2_patient.py \
      --root-dir /data/anqili/processed_data_bypatient/breast_cancer/ \
      --metadata-name metadata.jsonl \
      --models hoptimus1 gigapath virchow2 phikon_v2 uni_v2 \
      --output-base /data/anqili/geneprediction/breast_cancer/shazam_feature/ \
      --device cuda:2 \
      --batch-size 16 \
      --num-workers 4
"""
import os
import argparse
from pathlib import Path
import json
import h5py
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from huggingface_hub import login
from transformers import AutoModel
from tqdm import tqdm
from genedataset import GeneExpressionJSONLDataset


# Use token from environment to avoid embedding secrets in source
def login_from_env():
    token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN') or os.environ.get('HF_HUB_TOKEN')
    if token:
        login(token)

# --- Model Wrapper Classes ---
class Virchow2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        output = self.model(x)
        return output[:, 0]

class PhikonWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)
        return out.last_hidden_state[:, 0]

class Muskwrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out[0]

# --- Get encoder and forward function ---
def has_UNI():
    try:
        path = os.environ['UNI_CKPT_PATH']
        return True, path
    except KeyError:
        return False, ''


def get_encoder(model_name, extract_layers=['early', 'middle']):
    extracted_features = []
    def hook_fn(module, input, output):
        feat = output[0] if isinstance(output, tuple) else output
        extracted_features.append(feat[:, 0])

    if model_name == 'uni_v2':
        login_from_env()
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        layers = {'early': model.blocks[7], 'middle': model.blocks[15]}
    elif model_name == 'hoptimus1':
        login_from_env()
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True)
        layers = {'early': model.blocks[13], 'middle': model.blocks[26]}
    elif model_name == 'gigapath':
        login_from_env()
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        layers = {'early': model.blocks[13], 'middle': model.blocks[26]}
    elif model_name == 'musk':
        login_from_env()
        base = timm.create_model("musk_large_patch16_384", pretrained=True)
        model = Muskwrapper(base)
        layers = {'early': base.beit3.encoder.layers[7], 'middle': base.beit3.encoder.layers[15]}
    elif model_name == 'virchow2':
        login_from_env()
        from timm.layers import SwiGLUPacked
        base_model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        model = Virchow2Wrapper(base_model)
        # Assign layer_map to layers so that unified hook registration won't cause errors
        layers = {
            'early': base_model.blocks[10],
            'middle': base_model.blocks[20],
        }
        model.eval()
    elif model_name == 'phikon_v2':
        base = AutoModel.from_pretrained("owkin/phikon-v2")
        model = PhikonWrapper(base)
        layers = {'early': base.encoder.layer[7], 'middle': base.encoder.layer[15]}
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Register hooks
    for name, layer in layers.items():
        if name in extract_layers:
            layer.register_forward_hook(hook_fn)
    model.eval()

    def forward_fn(x):
        nonlocal extracted_features
        extracted_features = []
        _ = model(x)
        feats = {}
        for i, layer_name in enumerate(extract_layers):
            feats[layer_name] = extracted_features[i]
        return feats

    return model, forward_fn

# --- Feature extraction function ---
def extract_features(model_name, data_loader, save_path, device='cpu'):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, forward_fn = get_encoder(model_name)
    model.to(dev)

    low_feats, mid_feats, high_feats, labels = [], [], [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(data_loader, desc=model_name):
            imgs = imgs.to(dev)
            feats = forward_fn(imgs)
            low_feats.append(feats['early'].cpu())
            mid_feats.append(feats['middle'].cpu())
            high_feats.append(model(imgs).cpu())
            labels.append(lbls)
    low = torch.cat(low_feats)
    mid = torch.cat(mid_feats)
    high = torch.cat(high_feats)
    lbls = torch.cat(labels)
    torch.save((low, mid, high, lbls), save_path)
    print(f"Saved {model_name} features to {save_path}")

# --- Main entry point ---
def main():
    parser = argparse.ArgumentParser(description="Batch extract model features from subdirectories")
    parser.add_argument("--root-dir", required=True, help="Root directory containing multiple subdirectories")
    parser.add_argument("--metadata-name", default="metadata.jsonl", help="Metadata filename")
    parser.add_argument("--models", nargs='+', required=True, help="List of models")
    parser.add_argument("--output-base", required=True, help="Root directory for saving features")
    parser.add_argument("--device", default="cpu", help="Device, e.g., cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    root = Path(args.root_dir)
    base_out = Path(args.output_base)
    base_out.mkdir(parents=True, exist_ok=True)

    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        meta = sub / args.metadata_name
        if not meta.exists():
            print(f"Skipping {sub.name}, {args.metadata_name} not found")
            continue
        ds = GeneExpressionJSONLDataset(str(meta), str(sub), transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        out_sub = base_out / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)
        for m in args.models:
            save_file = out_sub / f"{m}_features.pt"
            extract_features(m, loader, str(save_file), device=args.device)

if __name__ == '__main__':
    main()
