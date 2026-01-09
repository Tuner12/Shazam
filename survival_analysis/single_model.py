#!/usr/bin/env python3
"""
WSI 单模型生存分析 · 仅 ABMIL · 使用NLL Loss · 5-fold
✅ 使用离散时间 NLL 损失函数
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
from single_dataset import SingleWSIDataset, single_collate

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
ap.add_argument('--n_bins', type=int, default=4, help='Number of time bins for discretization (default: 4)')

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
    events: numpy array, 1=event occurred (death), 0=censored (alive)
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

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Discrete-time negative log-likelihood (NLL).
    hazards: (B, T) tensor (probabilities in (0,1))
    S: (B, T) tensor or None
    Y: (B,1) or (B,) tensor of integer bin indices (1-based: 1..T)
    c: (B,1) or (B,) censorship (1=censored, 0=event)
    """
    device = hazards.device
    batch_size = hazards.size(0)
    T = hazards.size(1)

    Y = Y.view(batch_size, 1).long().to(device)
    c = c.view(batch_size, 1).float().to(device)

    if S is None:
        hazards_clamped = hazards.clamp(min=eps, max=1.0 - eps)
        S = torch.cumprod(1.0 - hazards_clamped, dim=1)

    # defensive: if Y looks 0-based, convert to 1-based
    if torch.min(Y) == 0:
        Y = (Y + 1).clamp(min=1)

    # clamp
    Y = Y.clamp(min=1, max=T)

    ones_col = torch.ones((batch_size, 1), dtype=S.dtype, device=device)
    S_padded = torch.cat([ones_col, S], dim=1)  # (B, T+1)

    hazards_idx = (Y - 1).clamp(min=0, max=T-1)
    hazards_at_Y = torch.gather(hazards, 1, hazards_idx)  # (B,1)

    S_before = torch.gather(S_padded, 1, (Y - 1).clamp(min=0))  # S_{Y-1}
    S_at_Y = torch.gather(S_padded, 1, Y.clamp(max=T))          # S_{Y}

    uncensored_loss = - (1.0 - c) * (torch.log(S_before.clamp(min=eps)) + torch.log(hazards_at_Y.clamp(min=eps)))
    censored_loss = - c * torch.log(S_at_Y.clamp(min=eps))

    neg_l = censored_loss + uncensored_loss
    loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss
# ------------------ Dataset ------------------ #
# WSIDataset和collate_fn已从single_dataset.py导入

class ABMIL(nn.Module):
    def __init__(self, C, n_bins=4, hidden=128, embed_dim=128, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(C, embed_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, hidden)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, 1, bias=False)
        
        # 输出离散时间的 hazards
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, n_bins),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        a = self.dropout2(self.tanh(self.fc2(x)))
        a = self.fc3(a)
        w = torch.softmax(a, 0)
        z = (w * x).sum(0)  # [embed_dim]

        hazards = self.classifier(z)  # (n_bins,)
        return hazards

# ------------------ Training ------------------ #
def train_one_epoch(model, loader, optimizer, device, n_bins, alpha=0.4):
    model.train()
    total_loss = 0
    cindex_sum = 0.0
    cindex_count = 0
    
    loader_bar = tqdm(loader, desc="Training", dynamic_ncols=True, leave=False)
    for patches, times, censorships, Y_bins, _ in loader_bar:
        # 跳过空批次（collate 在整批均无有效样本时返回 None）
        if patches is None or times is None or censorships is None or Y_bins is None:
            continue
        hazards_list = []
        for bag in patches:
            bag = bag.to(device)
            hazards = model(bag)  # (n_bins,)
            hazards_list.append(hazards)
        
        hazards_batch = torch.stack(hazards_list)  # (B, n_bins)
        S_batch = torch.cumprod(1.0 - hazards_batch.clamp(min=1e-7, max=1.0-1e-7), dim=1)  # (B, n_bins)
        
        times = times.to(device)
        censorships = censorships.to(device)
        Y_bins = Y_bins.to(device)
        
        loss = nll_loss(hazards_batch, S_batch, Y_bins, censorships, alpha=alpha)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 计算 C-index
        risk = (-torch.sum(S_batch, dim=1)).detach()
        try:
            times_np = times.detach().cpu().numpy()
            events_np = (1.0 - censorships).detach().cpu().numpy()  # 1=事件, 0=删失
            risks_np = risk.detach().cpu().numpy()
            cidx = c_index(times_np, events_np, risks_np)
            cindex_sum += cidx
            cindex_count += 1
        except Exception:
            pass
            
    avg_loss = total_loss / len(loader)
    avg_cindex = cindex_sum / max(1, cindex_count) if cindex_count > 0 else 0.0
    return avg_loss, avg_cindex


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    # 统计总进度时兼容空批次
    total_wsi = sum(0 if feats is None else len(feats) for feats, _, _, _, _ in loader)  
    with tqdm(total=total_wsi, desc="Evaluating", ncols=100, dynamic_ncols=True, leave=False) as pbar:
        for feats, t_batch, c_batch, Y_batch, _ in loader:
            # 跳过空批次
            if feats is None or t_batch is None or c_batch is None or Y_batch is None:
                continue
            for bag, t, c in zip(feats, t_batch, c_batch):
                bag = bag.to(device)
                hazards = model(bag)  # (n_bins,)
                S = torch.cumprod(1.0 - hazards.clamp(min=1e-7, max=1.0-1e-7), dim=0)
                risk_score = (-torch.sum(S)).item()
                
                risks.append(risk_score)
                times.append(t.item())
                events.append((1.0 - c).item())  # 转换为 event
                pbar.update(1)
    return c_index(np.array(times), np.array(events), np.array(risks))

@torch.no_grad()
def inference_and_save(model, dataset, device, save_path):
    """
    推理函数，保存预测结果到JSON文件
    """
    model.eval()
    results = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_collate)
    for feats, times, censorships, Y_bins, wsi_names in tqdm(loader, desc="Test inference"):
        # 跳过空批次
        if feats is None or times is None or censorships is None or Y_bins is None or wsi_names is None:
            continue
        bag = feats[0].to(device)
        hazards = model(bag)  # (n_bins,)
        S = torch.cumprod(1.0 - hazards.clamp(min=1e-7, max=1.0-1e-7), dim=0)
        risk_score = (-torch.sum(S)).item()
        
        results.append({
            "slide_id": wsi_names[0],
            "pred_hazards": hazards.cpu().tolist(),
            "pred_survival": S.cpu().tolist(),
            "pred_risk": risk_score,
            "survival_months": times[0].item(),
            "Y": Y_bins[0].item(),
            "censorship": censorships[0].item()
        })
    
    output_data = {
        "description": {
            "pred_hazards": "predicted hazard probabilities h(t) for each time bin",
            "pred_survival": "predicted survival probabilities S(t) = prod(1-h)",
            "pred_risk": "risk score = -sum(S), higher=worse prognosis",
            "censorship": "1=censored (alive), 0=death event",
            "Y": "discretized bin index (1-based)",
            "survival_months": "survival time in months"
        },
        "results": results
    }
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def extract_case_id(slide_path):
    filename = os.path.basename(slide_path)
    return '-'.join(filename.split('-')[:3])

# 旧的load_split函数已被移除，现在使用SingleWSIDataset.create_fold_datasets方法

# ------------------ Main ------------------ #
def main():
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_df = pd.read_csv(args.csv)
    
    # 离散化处理现在由SingleWSIDataset自动完成
    n_bins = args.n_bins
    print(f"使用 {n_bins} 个bins进行生存时间离散化")
    
    # 定义splits_dir路径
    splits_dir = Path(args.splits_dir)
    
    # 推断特征维度
    first_slide_path = all_df.loc[0, 'slide_id']
    slide_id = os.path.splitext(os.path.basename(first_slide_path))[0]
    sample_pt = torch.load(TEACHER_DIR / f"{slide_id}.pt", weights_only=True)[2]  # high
    in_dim = sample_pt.shape[1]
    
    csv_name = Path(args.csv).stem
    dataset_name = csv_name.split('_')[1]
    model_name = args.teacher

    checkpoint_dir = Path("checkpoints_nll") / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True) 

    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    for fold in folds:
        print(f"\n===== Fold {fold} =====")
        
        # 使用新的SingleWSIDataset.create_fold_datasets方法创建数据集
        # 自动展开多个slide为独立样本，使用bool类型的fold文件
        train_dataset, val_dataset, test_dataset = SingleWSIDataset.create_fold_datasets(
            all_df, TEACHER_DIR, splits_dir, fold, n_bins=n_bins, 
            create_Y=True, expand_multi_slides=True
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=single_collate)
        print("train_len",len(train_loader))
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=single_collate)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=single_collate)
        
        model = ABMIL(C=in_dim, n_bins=n_bins).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val = 0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for epoch in epoch_bar:
            train_loss, train_c = train_one_epoch(model, train_loader, optimizer, device, n_bins)
            val_c = evaluate(model, val_loader, device)
            
            if val_c > best_val:
                best_val = val_c
                checkpoint_path = checkpoint_dir / f"{model_name}_fold{fold}_nll_bestval.pt"
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()
            tqdm.write(f"Epoch {epoch:02d}: loss={train_loss:.4f} train_c={train_c:.4f} val={val_c:.4f}")

        print(f"Fold {fold}: best val={best_val:.4f}")
        val_scores.append(best_val)
        
        # 测试集评估
        best_model_path = checkpoint_dir / f"{model_name}_fold{fold}_nll_bestval.pt"
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_c = evaluate(model, test_loader, device)
        test_scores.append(test_c)
        
        # test_dataset = WSIDataset(test_df)
        save_json_path = checkpoint_dir / f"{model_name}_fold{fold}_nll_test_preds.json"
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

# =============================================================================
# 单教师模型训练指令 - 使用SingleWSIDataset类，只使用high层特征
# =============================================================================
# 
# 基本命令格式:
# CUDA_VISIBLE_DEVICES=<GPU_ID> python single_model_nll_v2.py \
#   --csv <CSV文件路径> \
#   --splits_dir <fold分割目录> \
#   --root <特征根目录> \
#   --teacher <教师模型名称> \
#   --epochs <训练轮数> \
#   --lr <学习率> \
#   --fold_idx <指定fold> \
#   --n_bins <时间分箱数>
#
# 参数说明:
# --csv: 包含slide_id, survival_months, censorship列的CSV文件
# --splits_dir: 包含splits_0_bool.csv到splits_4_bool.csv的目录
# --root: 包含各教师模型特征目录的根目录
# --teacher: 单个教师模型目录名称 (gigapath_features, hoptimus1_features, 
#           phikon_v2_features, uni_v2_features, virchow2_features)
# --epochs: 训练轮数，默认20
# --lr: 学习率，默认2e-4
# --fold_idx: 指定单个fold训练，不指定则训练所有fold
# --n_bins: 生存时间分箱数，默认4
#
# =============================================================================

# 示例训练命令 (按教师模型分组):

# === Gigapath模型训练 ===
# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher gigapath_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/KIRC_gigapath.txt

# CUDA_VISIBLE_DEVICES=1 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features \
#   --teacher gigapath_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/BRCA_gigapath.txt

# === Hoptimus1模型训练 ===
# CUDA_VISIBLE_DEVICES=2 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher hoptimus1_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/KIRC_hoptimus1.txt

# CUDA_VISIBLE_DEVICES=3 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features \
#   --teacher hoptimus1_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/BRCA_hoptimus1.txt

# === Phikon_v2模型训练 ===
# CUDA_VISIBLE_DEVICES=4 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher phikon_v2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/KIRC_phikon_v2.txt

# CUDA_VISIBLE_DEVICES=5 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features \
#   --teacher phikon_v2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/BLCA_phikon_v2.txt

# === Uni_v2模型训练 ===
# CUDA_VISIBLE_DEVICES=6 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher uni_v2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/KIRC_uni_v2.txt

# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features \
#   --teacher uni_v2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/BRCA_uni_v2.txt

# === Virchow2模型训练 ===
# CUDA_VISIBLE_DEVICES=1 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher virchow2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/KIRC_virchow2.txt

# CUDA_VISIBLE_DEVICES=2 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features \
#   --teacher virchow2_features \
#   --epochs 20 --lr 2e-4 \
#   | tee single_log/BRCA_virchow2.txt

# =============================================================================
# 按数据集分组的完整训练命令:
# =============================================================================

# === KIRC数据集 - 所有教师模型 ===
# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher gigapath_features --epochs 20 --lr 2e-4 | tee single_log/KIRC_gigapath.txt
# CUDA_VISIBLE_DEVICES=1 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher hoptimus1_features --epochs 20 --lr 2e-4 | tee single_log/KIRC_hoptimus1.txt
# CUDA_VISIBLE_DEVICES=2 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher phikon_v2_features --epochs 20 --lr 2e-4 | tee single_log/KIRC_phikon_v2.txt
# CUDA_VISIBLE_DEVICES=3 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher uni_v2_features --epochs 20 --lr 2e-4 | tee single_log/KIRC_uni_v2.txt
# CUDA_VISIBLE_DEVICES=4 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher virchow2_features --epochs 20 --lr 2e-4 | tee single_log/KIRC_virchow2.txt

# === BRCA数据集 - 所有教师模型 ===
# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher gigapath_features --epochs 20 --lr 2e-4 | tee single_log/BRCA_gigapath.txt
# CUDA_VISIBLE_DEVICES=1 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher hoptimus1_features --epochs 20 --lr 2e-4 | tee single_log/BRCA_hoptimus1.txt
# CUDA_VISIBLE_DEVICES=2 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher phikon_v2_features --epochs 20 --lr 2e-4 | tee single_log/BRCA_phikon_v2.txt
# CUDA_VISIBLE_DEVICES=3 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher uni_v2_features --epochs 20 --lr 2e-4 | tee single_log/BRCA_uni_v2.txt
# CUDA_VISIBLE_DEVICES=4 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BRCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher virchow2_features --epochs 20 --lr 2e-4 | tee single_log/BRCA_virchow2.txt

# === BLCA数据集 - 所有教师模型 ===
# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features --teacher gigapath_features --epochs 20 --lr 2e-4 | tee single_log/BLCA_gigapath.txt
# CUDA_VISIBLE_DEVICES=1 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features --teacher hoptimus1_features --epochs 20 --lr 2e-4 | tee single_log/BLCA_hoptimus1.txt
# CUDA_VISIBLE_DEVICES=2 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features --teacher phikon_v2_features --epochs 20 --lr 2e-4 | tee single_log/BLCA_phikon_v2.txt
# CUDA_VISIBLE_DEVICES=3 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features --teacher uni_v2_features --epochs 20 --lr 2e-4 | tee single_log/BLCA_uni_v2.txt
# CUDA_VISIBLE_DEVICES=4 python single_model_nll_v2.py --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features --teacher virchow2_features --epochs 20 --lr 2e-4 | tee single_log/BLCA_virchow2.txt

# =============================================================================
# 单fold训练示例 (用于调试或快速测试):
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 python single_model_nll_v2.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teacher phikon_v2_features \
#   --epochs 5 --lr 2e-4 --fold_idx 0 \
#   | tee single_log/KIRC_phikon_v2_fold0_test.txt

# =============================================================================
# 注意事项:
# =============================================================================
# 1. 确保所有路径存在且可访问
# 2. 使用不同的CUDA_VISIBLE_DEVICES避免GPU冲突
# 3. 训练日志会自动保存到single_log/目录
# 4. 模型检查点会保存到checkpoints_single/目录
# 5. 新版本使用SingleWSIDataset类，只加载high层特征
# 6. 使用bool类型的fold文件进行数据分割
# 7. 自动进行生存时间离散化处理
# 8. 支持自动展开多个slide为独立样本
# 9. 单教师模型训练，内存占用更少，训练速度更快

