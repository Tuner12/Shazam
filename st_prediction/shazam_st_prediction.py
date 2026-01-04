#!/usr/bin/env python3
"""
Patient-level K-Fold cross-validation training script (supports multi-GPU training), 
saves the best Pearson correlation metrics and corresponding model for each fold:
- Input: A root directory containing multiple patient subdirectories, each with several `<model>_features.pt` files
- Automatically determines number of folds: 5 folds if patient count > 5; otherwise folds = patient count
- Each fold splits by patient: training set vs validation set
- For each fold, trains StudentModel15Layers
- After each validation round, records best Pearson and corresponding MSE, predictions, labels, model state
- Saves best model for each fold, results JSON for each fold, and summary JSON

Multi-GPU training:
- First set environment variable `CUDA_VISIBLE_DEVICES=0,1,2,3`,
- Then specify main GPU via `--device cuda:0`; script will automatically enable `nn.DataParallel`.

Usage example:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python multi_moe_distill_gene_cv.py \
      --feature-root /data/.../shazam_feature \
      --models gigapath hoptimus1 phikon_v2 uni_v2 virchow2 \
      --output-dir /data/.../cv_results \
      --device cuda:0 \
      --batch-size 64 \
      --num-epochs 30 \
      --seed 42
"""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

# 1. Pearson Correlation
def calculate_pearson(preds, labels):
    """
    Calculate PCC for each gene separately, then average
    preds: [n_patches, n_genes]
    labels: [n_patches, n_genes]
    """
    # detach to avoid gradient tracking
    arr_p = preds.detach().cpu().numpy()  # [n_patches, n_genes]
    arr_l = labels.detach().cpu().numpy()  # [n_patches, n_genes]
    
    n_genes = arr_p.shape[1]
    gene_pccs = []
    
    for gene_idx in range(n_genes):
        gene_preds = arr_p[:, gene_idx]  # [n_patches]
        gene_labels = arr_l[:, gene_idx]  # [n_patches]
        
        # Check if there's sufficient variance
        if np.std(gene_preds) == 0 or np.std(gene_labels) == 0:
            gene_pccs.append(0.0)  # If variance is 0, set to 0
            continue
            
        try:
            pcc = np.corrcoef(gene_preds, gene_labels)[0, 1]
            if np.isnan(pcc):
                gene_pccs.append(0.0)  # If result is nan, set to 0
            else:
                gene_pccs.append(pcc)
        except Exception as e:
            print(f"Warning: Error calculating PCC for gene {gene_idx}: {e}")
            gene_pccs.append(0.0)
    
    mean_pcc = float(np.mean(gene_pccs))
    if np.isnan(mean_pcc):
        return 0.0  # If mean is nan, return 0
    return mean_pcc

# 2. Cross-Attention Block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, features):
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(K.size(-1))
        attn = self.softmax(scores)
        out = torch.matmul(attn, V)
        out = self.output_layer(out)
        return self.layernorm(features + out), attn

# 3. Multi-layer Cross-Attention
class MultiCrossAttentionLayers(nn.Module):
    def __init__(self, d_model, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model) for _ in range(num_layers)])
    def forward(self, features):
        attn_weights = []
        for layer in self.layers:
            features, w = layer(features)
            attn_weights.append(w)
        pooled = features.mean(dim=1)
        return pooled, attn_weights

# 4. MoE Single Layer
class MoEOnePerLevel(nn.Module):
    def __init__(self, in_dims, d_model=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(in_d, d_model) for in_d in in_dims])
        self.gate = nn.Sequential(
            nn.Linear(len(in_dims)*d_model,128), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(128), nn.Linear(128,len(in_dims))
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, *feats):
        ps = [self.proj[i](f) for i,f in enumerate(feats)]
        cat = torch.cat(ps, dim=1)
        w = torch.softmax(self.gate(cat), dim=1)
        outs = [self.ln(ps[i] * w[:,i:i+1]) for i in range(len(ps))]
        return torch.stack(outs, dim=1)

# 5. Feature Mapping
class FeatureMapper(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

# 6. StudentModel15Layers
class StudentModel15Layers(nn.Module):
    def __init__(self, dim_low, dim_mid, dim_high, d_model, n_genes):
        super().__init__()
        self.moe1 = MoEOnePerLevel(dim_low,  d_model)
        self.moe2 = MoEOnePerLevel(dim_mid,  d_model)
        self.moe3 = MoEOnePerLevel(dim_high, d_model)
        self.seg1 = MultiCrossAttentionLayers(d_model,4)
        self.seg2 = MultiCrossAttentionLayers(d_model,4)
        self.seg3 = MultiCrossAttentionLayers(d_model,4)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.reg = nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(d_model), nn.Linear(d_model,n_genes))
        # Add activation function to ensure gene prediction output is non-negative
        self.activation = nn.Softplus()  # Smoother than ReLU, output is always positive
        self.map_low  = nn.ModuleList([FeatureMapper(in_d,d_model) for in_d in dim_low])
        self.map_mid  = nn.ModuleList([FeatureMapper(in_d,d_model) for in_d in dim_mid])
        self.map_high = nn.ModuleList([FeatureMapper(in_d,d_model) for in_d in dim_high])
    def forward(self, features):
        low_feats  = features[:len(self.map_low)]
        mid_feats  = features[len(self.map_low):len(self.map_low)+len(self.map_mid)]
        high_feats = features[-len(self.map_high):]
        out1,_ = self.seg1(self.moe1(*low_feats))
        s2,_   = self.seg2(self.moe2(*mid_feats)); out2 = self.ln2(s2 + self.alpha*out1)
        s3,_   = self.seg3(self.moe3(*high_feats)); out3 = self.ln3(s3 + self.beta*out2 + self.alpha*out1)
        # Apply activation function to ensure gene prediction output is non-negative
        gene_pred = self.reg(out3)
        gene_pred = self.activation(gene_pred)
        return out1, out2, out3, gene_pred

# 7. Loss Functions
def distill_pair(s, t): return (1 - nn.functional.cosine_similarity(s, t, dim=-1).mean()) + nn.SmoothL1Loss()(s, t)

def multi_level_loss(out1, out2, out3, features, model):
    m = model.module if hasattr(model, 'module') else model
    low_feats  = features[:len(m.map_low)]
    mid_feats  = features[len(m.map_low):len(m.map_low)+len(m.map_mid)]
    high_feats = features[-len(m.map_high):]
    loss = 0
    for i, f in enumerate(low_feats): loss += distill_pair(out1, m.map_low[i](f))
    for i, f in enumerate(mid_feats): loss += distill_pair(out2, m.map_mid[i](f))
    for i, f in enumerate(high_feats): loss += distill_pair(out3, m.map_high[i](f))
    return loss

def ridge_loss(preds, labels, model, l2_lambda=1e-3):
    mse = nn.MSELoss()(preds, labels)
    l2  = sum((p**2).sum() for p in model.parameters())
    return mse + l2_lambda * l2

# 8. Load Features
def load_features(root, pats, models):
    low, mid, high, lbls = [], [], [], []
    for p in pats:
        pdir = os.path.join(root, p)
        for i, m in enumerate(models):
            data = torch.load(os.path.join(pdir, f"{m}_features.pt"), map_location='cpu')
            if i == 0: lbls.append(data[3])
            low.append(data[0]); mid.append(data[1]); high.append(data[2])
    num_models = len(models)
    low_feats  = [torch.cat(low[i::num_models], 0) for i in range(num_models)]
    mid_feats  = [torch.cat(mid[i::num_models], 0) for i in range(num_models)]
    high_feats = [torch.cat(high[i::num_models], 0) for i in range(num_models)]
    for i, lbl in enumerate(lbls):
        print(f"Patient {pats[i]}: label shape = {tuple(lbl.shape)}")
    labels     = torch.cat(lbls,     0)
    return low_feats, mid_feats, high_feats, labels

# Main Process
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-root', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    patients = sorted([d.name for d in os.scandir(args.feature_root) if d.is_dir()])
    k = min(5, len(patients))
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    summary = []

    for fold, (tr, vl) in enumerate(kf.split(patients), 1):
        print(f"Starting fold {fold}/{k}")
        train_p = [patients[i] for i in tr]
        val_p   = [patients[i] for i in vl]
        tlow, tmid, thigh, tlbl = load_features(args.feature_root, train_p, args.models)
        vlow, vmid, vhigh, vlbl = load_features(args.feature_root, val_p,   args.models)
        n_genes = tlbl.size(1)

        train_ds = TensorDataset(*tlow, *tmid, *thigh, tlbl)
        val_ds   = TensorDataset(*vlow, *vmid, *vhigh, vlbl)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

        dims_low  = [x.size(1) for x in tlow]
        dims_mid  = [x.size(1) for x in tmid]
        dims_high = [x.size(1) for x in thigh]
        model = StudentModel15Layers(dims_low, dims_mid, dims_high, d_model=128, n_genes=n_genes)
        model.to(device)

        if device.type.startswith('cuda') and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs for training")

        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        best_pear, best_mse, best_preds, best_labels, best_state = -float('inf'), None, None, None, None
        for epoch in tqdm(range(1, args.num_epochs+1), desc=f"Fold {fold} Epochs"):  # tqdm for epochs
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold {fold} Training", leave=False):  # tqdm for train batches
                feats  = [b.to(device) for b in batch[:-1]]
                labels = batch[-1].to(device)
                out1, out2, out3, preds = model(feats)
                loss = ridge_loss(preds, labels, model) + 0.01 * multi_level_loss(out1, out2, out3, feats, model)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            allp, alll = [], []
            for batch in tqdm(val_loader, desc=f"Fold {fold} Validation", leave=False):  # tqdm for val batches
                feats  = [b.to(device) for b in batch[:-1]]
                labels = batch[-1].to(device)
                _, _, _, preds = model(feats)
                allp.append(preds); alll.append(labels)
            preds_concat  = torch.cat(allp, dim=0)
            labels_concat = torch.cat(alll, dim=0)
            pear = calculate_pearson(preds_concat, labels_concat)
            mse  = nn.MSELoss()(preds_concat, labels_concat).item()
            print(f"Fold {fold} Epoch {epoch}: Val Pearson={pear:.4f}, Val MSE={mse:.4f}")
            if pear > best_pear:
                best_pear = pear
                best_mse = mse
                best_preds = preds_concat.cpu().tolist()
                best_labels = labels_concat.cpu().tolist()
                best_state = model.module.state_dict() if hasattr(model,'module') else model.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"fold{fold}_best.pth")
        torch.save(best_state, ckpt_path)

        result = {
            'fold': fold,
            'pearson': best_pear,
            'mse': best_mse,
            'preds': best_preds,
            'labels': best_labels
        }
        json_path = os.path.join(args.output_dir, f"fold{fold}_results.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(result, jf, ensure_ascii=False, indent=2)
        summary.append({'fold': fold, 'pearson': best_pear, 'mse': best_mse, 'checkpoint': ckpt_path, 'result_json': json_path})

    summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    print('Cross-Validation completed, results saved to:', args.output_dir)
