#!/usr/bin/env python3
"""
WSI 单模型生存分析 · 仅 ABMIL · 无蒸馏 · 无 cross-attention · 5-fold
"""
import os
import argparse, random, math
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored
from scipy import stats
import signal
import sys

# def handle_sigterm(signum, frame):
#     print("\n[Signal] Received SIGTERM (kill). Exiting gracefully...", flush=True)
#     sys.exit(0)

# signal.signal(signal.SIGTERM, handle_sigterm)
# signal.signal(signal.SIGINT, handle_sigterm)
# ------------------ CLI ------------------ #
ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True)
ap.add_argument('--splits_dir', required=True)  # split0.csv - split4.csv
ap.add_argument('--root', required=True)        # root/<teacher>_features/pt_files
ap.add_argument('--teacher', required=True)     # one teacher name
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--fold_idx', type=int, default=None, help='Specify one fold to run (0~4); default runs all 5 folds')

args = ap.parse_args()

TEACHER_DIR = Path(args.root) / f"{args.teacher}/merged_pt_files"
BATCH_SIZE = 64

# ------------------ Utils ------------------ #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# def c_index(t, e, r):
#     n, c, ties = 0, 0, 0
#     for i in range(len(t)):
#         for j in range(i+1, len(t)):
#             if e[i] == e[j] == 0:
#                 continue
#             if (e[i] == 1 and t[i] < t[j]) or (e[j] == 1 and t[j] < t[i]):
#                 hi, lo = (j, i) if t[i] < t[j] else (i, j)
#             else:
#                 continue
#             n += 1
#             if r[hi] > r[lo]:
#                 c += 1
#             elif math.isclose(r[hi], r[lo]):
#                 ties += 1
#     return (c + 0.5 * ties) / max(1, n)
def c_index(times, events, risks):
    """
    times: numpy array, survival times
    events: numpy array, 1=event occurred, 0=censored
    risks: numpy array, model predicted risks (higher = worse)
    """
    # 注意 sksurv 需要 event indicator 是 True/False
    events_bool = (events == 1)
    score = concordance_index_censored(events_bool, times, risks, tied_tol=1e-08)[0]
    return score

# class CoxPHLoss(nn.Module):
#     def forward(self, risk, time, event):
#         # Step 1: manually build the "at risk" matrix
#         n = len(time)
#         R_mat = torch.zeros((n, n), dtype=torch.float32, device=risk.device)
#         for i in range(n):
#             for j in range(n):
#                 if time[j] >= time[i]:
#                     R_mat[i, j] = 1.0

#         # Step 2: compute loss based on R_mat
#         theta = risk.reshape(-1)
#         exp_theta = torch.exp(theta)

#         # numerator: theta_i
#         # denominator: log(sum_j exp(theta_j) where j >= i)
#         loss = -((theta - torch.log(torch.matmul(R_mat, exp_theta))) * event).sum() / event.sum()

#         return loss
class CoxPHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
        """
        计算稳定的 Cox 部分似然损失。
        - risk: 模型预测的风险分数（越大风险越高），shape = [B]
        - time: 生存时间，shape = [B]
        - event: 事件指示（1=发生，0=删失），shape = [B]
        """
        # 按时间降序排序，使早死亡的人排前面
        t, order = time.sort(descending=True)
        risk = risk[order].float()
        event = event[order]

        # 数值稳定：截断 risk 防止 exp 爆炸
        # theta = torch.clamp(risk, min=-20.0, max=20.0)
        log_den = torch.logcumsumexp(risk, dim=0)
        # log_den = torch.nan_to_num(log_den, nan=0.0, posinf=0.0, neginf=-20.0)

        # 计算部分似然的负 log（注意 event.sum 可能为 0）
        loss = -((risk - log_den) * event).sum() / (event.sum() + 1e-8)
        return loss
# ------------------ Dataset ------------------ #
class WSIDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wsi = row['slide_id']
        wsi = os.path.splitext(os.path.basename(wsi))[0]
        # print(wsi)
        low, mid, high = torch.load(TEACHER_DIR / f"{wsi}.pt", map_location='cpu')  # Tuple format
        # print(high.shape)
        label = {
            'time': torch.tensor(row['survival_months'], dtype=torch.float32),
            'event': torch.tensor(row['censorship'], dtype=torch.float32)
        }
        # print(label)
        return high, label, wsi  # ← Only use high-level feature

# def collate_fn(batch):
#     return batch[0]  # batch_size = 1
def collate_fn(batch):
    feats = [item[0] for item in batch]  # list of patch tensors
    times = torch.tensor([item[1]['time'] for item in batch], dtype=torch.float32)
    events = torch.tensor([item[1]['event'] for item in batch], dtype=torch.float32)
    wsi_names = [item[2] for item in batch]
    return feats, times, events, wsi_names

class ABMIL(nn.Module):
    def __init__(self, C, hidden=128, embed_dim=128, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(C, embed_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, hidden)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, 1, bias=False)
        self.classifier = nn.Linear(embed_dim, 1)
    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        a = self.dropout2(self.tanh(self.fc2(x)))
        a = self.fc3(a)
        w = torch.softmax(a, 0)
        z = (w * x).sum(0)  # [C]

        risk = self.classifier(z).squeeze()
        return risk

# ------------------ Training ------------------ #
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loader_bar = tqdm(loader, desc="Training", dynamic_ncols=True, leave=False)
    for patches, times, events,_ in loader_bar:
        risk_list = []
        for bag in patches:
            bag = bag.to(device)
            risk = model(bag)
            risk_list.append(risk)
        # risk_vec = torch.stack(risk_list).squeeze()
        risk_vec = torch.stack(risk_list).view(-1)

        times, events = times.to(device), events.to(device)
        loss = loss_fn(risk_vec, times, events)
        # print("loss",loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    total_wsi = sum(len(feats) for feats, _, _,_ in loader)  
    with tqdm(total=total_wsi, desc="Evaluating", ncols=100, dynamic_ncols=True, leave=False) as pbar:
        for feats, t_batch, e_batch,_ in loader:
            for bag, t, e in zip(feats, t_batch, e_batch):
                bag = bag.to(device)
                risk = model(bag).item()
                risks.append(risk)
                times.append(t.item())
                events.append(e.item())
    return c_index(np.array(times), np.array(events), np.array(risks))

@torch.no_grad()
def inference_and_save(model, dataset, device, save_path):
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for feats, times, events, wsi_names in tqdm(loader, desc="Test inference"):
        bag = feats[0].to(device)
        pred_risk = model(bag).item()
        time = times[0].item()
        event = events[0].item()
        wsi_name = wsi_names[0]
        results.append({
            "slide_id": wsi_name,
            "pred_risk": pred_risk,
            "time": time,
            "event": event
        })
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

def extract_case_id(slide_path):
    filename = os.path.basename(slide_path)
    return '-'.join(filename.split('-')[:3])

# def load_split(idx, all_df):
#     split_df = pd.read_csv(Path(args.splits_dir) / f"splits_{idx}.csv", dtype=str, keep_default_na=False, na_values=[''])
#     tr_ids = split_df['train'].dropna().unique().tolist()
#     va_ids = split_df['val'].dropna().unique().tolist()
#     te_ids = split_df['test'].dropna().unique().tolist()

#     return (
#         all_df[all_df['slide_id'].apply(extract_case_id).isin(tr_ids)],
#         all_df[all_df['slide_id'].apply(extract_case_id).isin(va_ids)],
#         all_df[all_df['slide_id'].apply(extract_case_id).isin(te_ids)]
#     )
def load_split(idx, all_df):
    split_df = pd.read_csv(Path(args.splits_dir) / f"splits_{idx}.csv", dtype=str, keep_default_na=False, na_values=[''])
    tr_ids = split_df['train'].dropna().unique().tolist()
    va_ids = split_df['val'].dropna().unique().tolist()
    te_ids = split_df['test'].dropna().unique().tolist()

    def is_valid_slide(slide_path):
        wsi = os.path.splitext(os.path.basename(slide_path))[0]
        pt_path = TEACHER_DIR / f"{wsi}.pt"
        return pt_path.exists()

    train_df = all_df[all_df['slide_id'].apply(extract_case_id).isin(tr_ids)]
    val_df   = all_df[all_df['slide_id'].apply(extract_case_id).isin(va_ids)]
    test_df  = all_df[all_df['slide_id'].apply(extract_case_id).isin(te_ids)]

    train_df = train_df[train_df['slide_id'].apply(is_valid_slide)]
    val_df   = val_df[val_df['slide_id'].apply(is_valid_slide)]
    test_df  = test_df[test_df['slide_id'].apply(is_valid_slide)]

    return train_df, val_df, test_df

# ------------------ Main ------------------ #
def main():
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_df = pd.read_csv(args.csv)
    # slide_id_name = 
    first_slide_path = all_df.loc[0, 'slide_id']
    slide_id = os.path.splitext(os.path.basename(first_slide_path))[0]
    # infer in_dim
    sample_pt = torch.load(TEACHER_DIR / f"{slide_id}.pt")[2]  # high
    in_dim = sample_pt.shape[1]
    
    csv_name = Path(args.csv).stem  # e.g. "TCGA_BLCA_Splits"
    dataset_name = csv_name.split('_')[1]  # "BLCA"
    model_name = args.teacher

    checkpoint_dir = Path("checkpoints") / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True) 

    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    for fold in folds:
        print(f"\n===== Fold {fold} =====")
        train_df, val_df, test_df = load_split(fold, all_df)

        train_loader = DataLoader(WSIDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        print("train_len",len(train_loader))
        val_loader   = DataLoader(WSIDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        print("val_len",len(val_loader))
        test_loader  = DataLoader(WSIDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        print("test_len",len(test_loader))
        model = ABMIL(C = in_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn = CoxPHLoss()

        best_val, best_test = 0, 0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for epoch in epoch_bar:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            print(train_loss)
            val_c = evaluate(model, val_loader, device)
            test_c = evaluate(model, test_loader, device)
            if val_c > best_val:
                best_val, best_test = val_c, test_c
                checkpoint_path = checkpoint_dir / f"{model_name}_fold{fold}_bestval.pt"
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()
            tqdm.write(f"Epoch {epoch:02d}: loss={train_loss:.4f} val={val_c:.4f} test={test_c:.4f}")

        print(f"Fold {fold}: best val={best_val:.4f}, best test={best_test:.4f}")
        val_scores.append(best_val)
        test_scores.append(best_test)
        best_model_path = checkpoint_dir / f"{model_name}_fold{fold}_bestval.pt"
        model.load_state_dict(torch.load(best_model_path))
        test_dataset = WSIDataset(test_df)
        save_json_path = checkpoint_dir / f"{model_name}_fold{fold}_test_preds.json"
        inference_and_save(model, test_dataset, device, save_json_path)
    print("\n===== 5-fold Summary =====")
    # print("val  mean: {:.4f} ± {:.4f}".format(np.mean(val_scores), np.std(val_scores)))
    # print("test mean: {:.4f} ± {:.4f}".format(np.mean(test_scores), np.std(test_scores)))
    # val_mean = np.mean(val_scores)
    # val_std = np.std(val_scores)
    # val_ci95 = 1.96 * val_std / np.sqrt(5)

    # test_mean = np.mean(test_scores)
    # test_std = np.std(test_scores)
    # test_ci95 = 1.96 * test_std / np.sqrt(5)
    val_mean = np.mean(val_scores)
    val_std = np.std(val_scores, ddof=1)
    val_ci95 = stats.t.ppf(0.975, df=4) * val_std / np.sqrt(5)


    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores, ddof=1)
    test_ci95 = stats.t.ppf(0.975, df=4) * test_std / np.sqrt(5)
    print('\n===== 5-fold Summary =====')
    print(f'val mean {val_mean:.4f} ± {val_std:.4f} (95% CI: [{val_mean-val_ci95:.4f}, {val_mean+val_ci95:.4f}])')
    print(f'test mean {test_mean:.4f} ± {test_std:.4f} (95% CI: [{test_mean-test_ci95:.4f}, {test_mean+test_ci95:.4f}])')

if __name__ == "__main__":
    main()

# python wsi_abmil_survival.py \
#   --csv all_cases.csv \
#   --splits_dir splits82/TCGA_BLCA_survival_100 \
#   --root TCGA_BLCA_multi_features \
#   --teacher virchow2_features \
#   --epochs 30 --lr 3e-4
# python single_model.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teacher phikon_v2_features --epochs 20 --lr 2e-4 | tee single_log/blcaphikon.txt
# CUDA_VISIBLE_DEVICES=4 python single_model.py --csv dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir splits82/TCGA_KIRC_survival_100 --root TCGA_KIRC_multi_features --teacher phikon_v2_features --epochs 20 --lr 2e-4 | tee single_log/kircphikon.txt
# CUDA_VISIBLE_DEVICES=5 python single_model.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teacher virchow2_features --epochs 20 --lr 2e-4 | tee single_log/blcavirchow.txt
