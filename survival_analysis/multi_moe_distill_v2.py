#!/usr/bin/env python3
"""
WSI‑level survival · 5‑fold CV using split<i>.csv  (ID‑enumeration format)
--------------------------------------------------------------------------

`splits_dir` 必须含有  ↓↓↓
    split0.csv
    split1.csv
    ...
    split4.csv

每个文件格式示例
----------------
train,val,test
TCGA-DK-A31K,,
,TCGA-CU-A0YO,
,,TCGA-CU-A0YO
...

⚠️ 只读取这 5 个文件；`split_bool<i>.csv` 完全忽略。
"""

# -------------------- Imports -------------------- #
import argparse, random, math
from pathlib import Path
from typing import List, Dict
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored
# from torch import amp
from torch import autocast
from torch.amp import GradScaler

# scaler = amp.GradScaler(device='cuda')

# -------------------- CLI ------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument('--csv',       required=True, help='full meta CSV: wsi_id,survival_months,censorship')
ap.add_argument('--splits_dir',required=True, help='folder containing split0.csv … split4.csv')
ap.add_argument('--root',      required=True, help='root with <teacher>_features/pt_files')
ap.add_argument('--teachers',  nargs='+',    required=True, help='teacher folder names')
ap.add_argument('--epochs',    type=int,   default=20)
ap.add_argument('--lr',        type=float, default=2e-4)
ap.add_argument('--lambda_dist', type=float, default=0.01)
ap.add_argument('--seed',      type=int,   default=42)
ap.add_argument('--fold_idx', type=int, default=None, help='Specify a single fold to run (0-4); if not set, all folds will be run')

ap.add_argument('--batch_size', type=int, default=1)
args = ap.parse_args()

TEACHER_DIRS = [Path(args.root)/f'{t}/merged_pt_files' for t in args.teachers]
N_TEACHERS   = len(TEACHER_DIRS)
D_MODEL      = 128
# BATCH_SIZE   = 1

# ---------------- reproducibility ---------------- #
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# --------------- metrics & loss ------------------ #
# def c_index(t,e,r):
#     n,c,tie = 0,0,0
#     for i in range(len(t)):
#         for j in range(i+1,len(t)):
#             if e[i]==e[j]==0: continue
#             if (e[i]==1 and t[i]<t[j]) or (e[j]==1 and t[j]<t[i]):
#                 hi,lo = (j,i) if t[i]<t[j] else (i,j)
#             else: continue
#             n+=1
#             if r[hi]>r[lo]: c+=1
#             elif math.isclose(r[hi],r[lo]): tie+=1
#     return (c+0.5*tie)/max(1,n)
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
#     def forward(self,risk,time,event):
#         idx=torch.argsort(time,descending=True)
#         risk,event=risk[idx],event[idx]
#         return -((risk-torch.log(torch.cumsum(torch.exp(risk),0)))*event).sum()/event.sum()
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
# class CoxPHLoss(nn.Module):
#     def forward(self, risk, time, event):
#         print("=== CoxPHLoss Debugging Start ===")
#         n = len(time)
#         R_mat = torch.zeros((n, n), dtype=risk.dtype, device=risk.device)
#         for i in range(n):
#             for j in range(n):
#                 if time[j] >= time[i]:
#                     R_mat[i, j] = 1.0

#         theta = risk.reshape(-1)
#         print(f"[theta] min: {theta.min().item():.6f}, max: {theta.max().item():.6f}, mean: {theta.mean().item():.6f}")

#         exp_theta = torch.exp(theta)
#         print(f"[exp_theta] min: {exp_theta.min().item():.6f}, max: {exp_theta.max().item():.6f}, mean: {exp_theta.mean().item():.6f}")

#         denom = torch.matmul(R_mat, exp_theta)+ 1e-8
#         print(f"[denom (R_mat @ exp_theta)] min: {denom.min().item():.6f}, max: {denom.max().item():.6f}, mean: {denom.mean().item():.6f}")

#         log_denom = torch.log(denom)
#         print(f"[log_denom] min: {log_denom.min().item():.6f}, max: {log_denom.max().item():.6f}, mean: {log_denom.mean().item():.6f}")

#         loss = -((theta - log_denom) * event).sum() / event.sum()
        
#         print(f"[loss] value: {loss.item():.6f}")

#         if not torch.isfinite(loss):
#             print("⚠️ [Warning] CoxPHLoss computed NaN or Inf!")

#         print("=== CoxPHLoss Debugging End ===\n")

#         return loss
class CoxPHLoss(nn.Module):
    def forward(self, risk, time, event):
        n = len(time)
        # R_mat = torch.zeros((n, n), dtype=torch.float32, device=risk.device)
        R_mat = torch.zeros((n, n), dtype=risk.dtype, device=risk.device)

        for i in range(n):
            for j in range(n):
                if time[j] >= time[i]:
                    R_mat[i, j] = 1.0

        theta = risk.reshape(-1)
        theta = torch.clamp(theta, min=-20, max=20)    # 防止 exp 爆炸
        exp_theta = torch.exp(theta)

        denom = torch.matmul(R_mat, exp_theta) + 1e-8  # 防止0
        denom = torch.clamp(denom, min=1e-8)            # 双保险

        log_denom = torch.log(denom)
        log_denom = torch.nan_to_num(log_denom, nan=0.0, posinf=0.0, neginf=-20.0)

        loss = -((theta - log_denom) * event).sum() / (event.sum() + 1e-8)

        # if not torch.isfinite(loss):
        #     raise ValueError("CoxPHLoss resulted in NaN or Inf!")

        return loss

# ---------------- split parser ------------------- #
def load_split_df(split_idx:int)->pd.DataFrame:
    path = Path(args.splits_dir)/f'splits_{split_idx}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    # keep_blank_values to preserve empty cells ⇒ NaN
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    return df

def get_ids(df_split:pd.DataFrame, col:str)->List[str]:
    return df_split[col].dropna().unique().tolist()
def extract_case_id(slide_path):
    filename = os.path.basename(slide_path)
    return '-'.join(filename.split('-')[:3])
def prepare_fold(full_df:pd.DataFrame, split_idx:int):
    df_split=load_split_df(split_idx)
    tr_ids = get_ids(df_split,'train')
    val_ids= get_ids(df_split,'val')
    tst_ids= get_ids(df_split,'test')
    return (
        full_df[full_df['slide_id'].apply(extract_case_id).isin(tr_ids)],
        full_df[full_df['slide_id'].apply(extract_case_id).isin(val_ids)],
        full_df[full_df['slide_id'].apply(extract_case_id).isin(tst_ids)]
    )

# ---------------- Dataset ------------------------ #
class WSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row['slide_id']
        wsi = os.path.splitext(os.path.basename(slide_path))[0]


        # ------------- 读取一个教师的 (low, mid, high) ------------- #
        bag = {'low': [], 'mid': [], 'high': []}
        for d in TEACHER_DIRS:
            low_t, mid_t, high_t = torch.load(d / f'{wsi}.pt', map_location='cpu', weights_only=True)  # tuple
            bag['low' ].append(low_t)
            # print("low",low_t.shape)
            bag['mid' ].append(mid_t)
            bag['high'].append(high_t)

        label = {
            'time':  torch.tensor(row['survival_months'], dtype=torch.float32),
            'event': torch.tensor(row['censorship'],      dtype=torch.float32)
        }
        return bag, label


def collate(batch):
    feats = {'low': [], 'mid': [], 'high': []}
    times, events = [], []

    for bag, label in batch:
        for k in feats:
            for i, t in enumerate(bag[k]):
                if len(feats[k]) <= i:
                    feats[k].append([])
                feats[k][i].append(t)
        times.append(label['time'])
        events.append(label['event'])

    # 不要stack！！！直接保持list of tensors
    return feats, torch.stack(times), torch.stack(events)


# -------------- Model components ---------------- #
# class ABMIL(nn.Module):
#     def __init__(self,C,H=256):
#         super().__init__(); self.fc1=nn.Linear(C,H); self.tanh=nn.Tanh(); self.fc2=nn.Linear(H,1,bias=False)
#     def forward(self,x): a=self.fc2(self.tanh(self.fc1(x))); w=torch.softmax(a,0); return (w*x).sum(0)
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

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        a = self.dropout2(self.tanh(self.fc2(x)))
        a = self.fc3(a)
        w = torch.softmax(a, 0)
        return (w * x).sum(0)


class MoE(nn.Module):
    def __init__(self, input_dims, output_dim=128):
        """
        input_dims: List of input dimensions from different teachers, all unified to d_model after mapping
        output_dim: final fusion feature dim (default 512)
        """
        super().__init__()

        self.input_dims = input_dims
        # self.output_dim = output_dim
        self.num_experts = len(input_dims)

        self.gate = nn.Sequential(
            nn.Linear(self.num_experts * output_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.num_experts)
        )

        self.out_ln = nn.LayerNorm(output_dim)

    def forward(self, features):
        """
        features: List of tensor [d_feat] for each teacher
        """
        # 1. 映射到统一空间
        # proj_feats = [proj(f) for proj, f in zip(self.proj_layers, features)]  # List of [output_dim]

        # 2. 计算 gating 权重
        # features = torch.stack(features, dim=0) # 5 dmodel
        # concat_feats = features.flatten(0)
        concat_feats = torch.cat(features, dim=0)  # [num_experts * output_dim]
        # print('concat shape',concat_feats.shape)
        gate_scores = torch.softmax(self.gate(concat_feats), dim=-1)  # [num_experts]
        weighted_features = []
        for w, f in zip(gate_scores, features):
            weighted_feat = self.out_ln(w.unsqueeze(-1) * f)  # [n_patch, d_model]
            weighted_features.append(weighted_feat)
        # 3. 加权融合
        # out = [self.out_ln(w * f) for w, f in zip(gate_scores, features)]
        # print('stach shape',torch.stack(weighted_features, dim=0).shape)
        # 4. 最后归一化
        return torch.stack(weighted_features, dim=0)



class CrossBlk(nn.Module):
    def __init__(self,d):
        super().__init__(); self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d); self.ln=nn.LayerNorm(d)
    def forward(self,x):
        q,k,v=self.q(x),self.k(x),self.v(x)
        att=(q@k.transpose(-2,-1))/math.sqrt(k.size(-1))
        return self.ln(x + self.o(torch.softmax(att,-1)@v))

class MultiCross(nn.Module):
    def __init__(self,d,l=4): super().__init__(); self.blocks=nn.ModuleList([CrossBlk(d) for _ in range(l)])
    def forward(self,x):
        for blk in self.blocks: x=blk(x)
        return x.mean(1)                          # ← [B,d]




class Student(nn.Module):
    def __init__(self, d_teachers, d_model=128, n_layers=2):
        super().__init__()
        self.d_model = d_model

        # 每个 teacher 一个 ABMIL 聚合器
        self.pool_low  = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])
        self.pool_mid  = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])
        self.pool_high = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])

        # 特征降维，把原来不同维度特征统一到 d_model
        self.map_low  = nn.ModuleList([nn.Linear(d, d_model) for d in d_teachers])
        self.map_mid  = nn.ModuleList([nn.Linear(d, d_model) for d in d_teachers])
        self.map_high = nn.ModuleList([nn.Linear(d, d_model) for d in d_teachers])
        
        # MoE融合（wsi级特征）
        self.moe_low  = MoE([d_model] * len(d_teachers), d_model)
        self.moe_mid  = MoE([d_model] * len(d_teachers), d_model)
        self.moe_high = MoE([d_model] * len(d_teachers), d_model)

        # attention
        self.cross_low = MultiCross(d_model, n_layers)
        self.cross_mid = MultiCross(d_model, n_layers)
        self.cross_high = MultiCross(d_model, n_layers)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        # residual
        self.res_ln2 = nn.LayerNorm(d_model)
        self.res_ln3 = nn.LayerNorm(d_model)

        # risk预测
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

        # self.abmil = ABMIL(d_model)


    def forward(self, bag):
        """
        bag: {'low': List of 5 tensors [n_patch, d_feat], 'mid': ..., 'high': ...}
        """

        # for level in ['low', 'mid', 'high']:
        #     for i, feat in enumerate(bag[level]):
        #         print(f"=== Feature {level} {i} ===")
        #         print(f"min: {feat.min().item():.4f}, max: {feat.max().item():.4f}, mean: {feat.mean().item():.4f}, std: {feat.std().item():.4f}")

        # for level in ['low', 'mid', 'high']:
        #     for i in range(len(bag[level])):
        #         bag[level][i] = safe_feature(bag[level][i])


        low_proj  = [mapper(feat) for mapper, feat in zip(self.map_low,  bag['low'])]
        mid_proj  = [mapper(feat) for mapper, feat in zip(self.map_mid,  bag['mid'])]
        high_proj = [mapper(feat) for mapper, feat in zip(self.map_high, bag['high'])]
        # for idx, tensor in enumerate(low_proj):
        #     print('low_proj',tensor.shape)
        low_vec = [pool(feat) for pool, feat in zip(self.pool_low,  low_proj)]
        mid_vec = [pool(feat) for pool, feat in zip(self.pool_mid,  mid_proj)]
        high_vec = [pool(feat) for pool, feat in zip(self.pool_high,  high_proj)]
        # for idx, tensor in enumerate(low_vec):
        #     print('low_vec',tensor.shape)

        
        low_feat = self.moe_low(low_vec).unsqueeze(0)
        mid_feat = self.moe_mid(mid_vec).unsqueeze(0)
        high_feat = self.moe_high(high_vec).unsqueeze(0)
        # print('moe shape',low_feat.shape)
  
        low_out = self.cross_low(low_feat)  # [1, d_model]
        mid_out = self.cross_mid(mid_feat)
        mid_out = self.res_ln2(mid_out + self.alpha * low_out)
        high_out = self.cross_high(high_feat)
        high_out = self.res_ln3(high_out + self.beta * mid_out)
   
        risk = self.head(high_out).squeeze()
        return low_out, mid_out, high_out, risk


# cos,huber=nn.CosineSimilarity(-1),nn.HuberLoss()
# def dist_pair(s,t): return ((1 - cos(s, t)) + huber(s, t)).mean()
# def dist_pair(student_feat, teacher_feat):
#     """
#     一个简单的示例蒸馏损失: (1 - cos_sim) + SmoothL1
#     student_feat: [B, d_model]
#     teacher_feat: [B, d_model]
#     """
#     cos_term = 1.0 - nn.functional.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
#     smooth_l1= nn.HuberLoss()(student_feat, teacher_feat)
#     return cos_term + smooth_l1
def dist_pair(student_feat, teacher_feat, eps=1e-8):
    """
    (1 - cos_sim) + SmoothL1 loss，防止除以0导致NaN
    """
    # 防止 norm 很小导致的 NaN
    student_norm = student_feat.norm(dim=-1, keepdim=True).clamp(min=eps)
    teacher_norm = teacher_feat.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos_sim = (student_feat * teacher_feat).sum(dim=-1) / (student_norm * teacher_norm).squeeze(-1)
    cos_term = 1.0 - cos_sim.mean()

    smooth_l1 = nn.HuberLoss()(student_feat, teacher_feat)
    return cos_term + smooth_l1

# def distill(l,m,h,bag,model):
#     loss=0
#     for i,t in enumerate(bag['low' ]): loss+=dist_pair(l,model.map_low [i](t.mean(0)))
#     for i,t in enumerate(bag['mid' ]): loss+=dist_pair(m,model.map_mid [i](t.mean(0)))
#     for i,t in enumerate(bag['high']): loss+=dist_pair(h,model.map_high[i](t.mean(0)))
#     return loss/(3*N_TEACHERS)
def distill(out1, out2, out3, bag, model):
    """
    out1, out2, out3: model输出的low/mid/high (每个都是 [d_model])
    bag: {'low': [teacher1_patches, teacher2_patches, ...], ...}
    model: Student模型，用来取map_low/map_mid/map_high
    """
    loss = 0
    # with torch.no_grad(): 
    for i in range(len(bag['low'])):
        # 1. teacher i 的 low/mid/high patch 特征
        low_feat = bag['low'][i]   # shape: [npatch, d_teacher]
        mid_feat = bag['mid'][i]
        high_feat = bag['high'][i]
        # if not torch.isfinite(low_feat).all():
        #     print(f"[NaN detected in original low_feat {i}]")
        #     raise ValueError("NaN in teacher low_feat!")
        # if not torch.isfinite(mid_feat).all():
        #     print(f"[NaN detected in original mid_feat {i}]")
        #     raise ValueError("NaN in teacher mid_feat!")
        # if not torch.isfinite(high_feat).all():
        #     print(f"[NaN detected in original high_feat {i}]")
        #     raise ValueError("NaN in teacher high_feat!")
        # 2. 通过 Student里的map_low等，统一到 d_model
        low_proj = model.map_low[i](low_feat)    # [npatch, d_model]
        mid_proj = model.map_mid[i](mid_feat)
        high_proj = model.map_high[i](high_feat)
        low_vec = model.pool_low[i](low_proj).unsqueeze(0)
        mid_vec = model.pool_mid[i](mid_proj).unsqueeze(0)
        high_vec = model.pool_high[i](high_proj).unsqueeze(0)
        # n_patch = high_vec.size(0)
        # print("low_proj",low_proj.shape)
        # print("ou1",out1.shape)
        # # 3. 每个 teacher 自己做 ABMIL池化成 wsi级特征
        # low_wsi_feat = model.pool_low[i](low_proj)    # [d_model]
        # mid_wsi_feat = model.pool_mid[i](mid_proj)
        # high_wsi_feat = model.pool_high[i](high_proj)
        # print("low_wsi_feat",low_wsi_feat.shape)
        # print(out1.shape)
        # 4. 蒸馏 loss
        # loss += dist_pair(out1, low_proj)
        # loss += dist_pair(out2, mid_proj)
        # loss += dist_pair(out3, high_proj)
        # if not torch.isfinite(loss).all():
        #     print(f"[NaN detected in final distill loss]")
        #     raise ValueError("Final distill loss NaN!")
        # print('low_vec',low_vec.shape)
        # print('out1',out1.shape)
        
        loss += dist_pair(out1, low_vec)
   

   
        loss += dist_pair(out2, mid_vec) 
   

        
        loss += dist_pair(out3, high_vec) 
   
        del low_feat, mid_feat, high_feat, low_proj, mid_proj, high_proj, low_vec, mid_vec, high_vec
        torch.cuda.empty_cache()
        # print(loss)
        # print(bag['low'][0].shape[0])
        # print(3 * len(bag['low']))
        # print(loss / (3 * len(bag['low'])))
    return loss / (3 * len(bag['low']))

# from torch.cuda.amp import autocast, GradScaler

# def train_epoch(model, loader, opt, cox, lmbd, dev,  # ← 接口保持不变
#                 acc_size: int = 128, clip_grad: float = 1.0):
#     """
#     • 每看到一个样本：
#         1) 计算蒸馏损失并立即 backward，累积梯度；
#         2) 仅把 risk/time/event 的标量值放到 CPU 缓冲。
#     • 当缓冲达到 acc_size (=128)：
#         1) 重新构造 risks 张量（需要梯度）计算 Cox；
#         2) backward Cox；clip + step；zero_grad；
#         3) 清空缓冲，继续下一轮。
#     • 末尾不足 acc_size 的样本同理处理。
#     """
#     model.train()
#     scaler = GradScaler(init_scale=512, growth_interval=100)

#     # -----------  用 Python list 缓冲数值，绝不留计算图 -----------
#     buf_risk, buf_time, buf_event = [], [], []
#     seen = 0

#     opt.zero_grad(set_to_none=True)      # 提前清 grad，只在 step 时再清

#     for feats, times, events in tqdm(loader, leave=False, dynamic_ncols=True):
#         B = times.size(0)                # loader 批尺寸
#         for i in range(B):
#             # ---------- 1. 单样本前向 ----------
#             bag_i = {lvl: [t[i].to(dev) for t in lst] for lvl, lst in feats.items()}
#             with autocast(device_type='cuda'):
#                 l_s, m_s, h_s, risk = model(bag_i)
#                 distill_loss = distill(l_s, m_s, h_s, bag_i, model)

#             # ---------- 2. 累积蒸馏梯度 ----------
#             scaler.scale(distill_loss / acc_size).backward()   # /acc_size ≈ 平均化

#             # ---------- 3. 把 Cox 需要的标量挪到 CPU 缓冲 ----------
#             buf_risk.append(risk.detach().cpu())   # 只有数值，无图
#             buf_time.append(times[i].item())
#             buf_event.append(events[i].item())
#             seen += 1

#             # ---------- 4. 满 acc_size ⇒ 计算 Cox + 更新 ----------
#             if seen == acc_size:
#                 _cox_backward_and_step(buf_risk, buf_time, buf_event,
#                                        model, opt, cox, scaler,
#                                        acc_size, clip_grad, dev)
#                 buf_risk.clear(); buf_time.clear(); buf_event.clear()
#                 seen = 0
#                 torch.cuda.empty_cache()

#     # ----------- 处理 epoch 末尾残余样本 -----------
#     if seen:
#         _cox_backward_and_step(buf_risk, buf_time, buf_event,
#                                model, opt, cox, scaler,
#                                acc_size, clip_grad, dev)

#     return 0.0   # 若要返回 loss，可自行记录

# # ---------------------------------------------------------------
# def _cox_backward_and_step(buf_risk, buf_time, buf_event,
#                            model, opt, cox, scaler,
#                            denom, clip_grad, dev):
#     """
#     ① 把 CPU list 转成 GPU tensor（128×1），需要梯度；
#     ② 计算 Cox → backward；
#     ③ clip & step & scaler.update().
#     """
#     # 所有 censored ⇒ 跳过
#     if sum(buf_event) == 0:
#         opt.zero_grad(set_to_none=True)
#         return

#     risks  = torch.tensor(buf_risk,  device=dev, dtype=torch.float32,
#                           requires_grad=True).view(-1)
#     times  = torch.tensor(buf_time,  device=dev, dtype=torch.float32)
#     events = torch.tensor(buf_event, device=dev, dtype=torch.float32)

#     with autocast(device_type='cuda'):
#         cox_loss = cox(risks, times, events)

#     scaler.scale(cox_loss / denom).backward()       # /denom 与蒸馏一致

#     # -------- clip / step --------
#     scaler.unscale_(opt)
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
#     scaler.step(opt)
#     scaler.update()
#     opt.zero_grad(set_to_none=True)

# -------------- train & eval helpers -------------- #
def train_epoch(model, ldr, opt, cox, lmbd, dev):
    model.train()
    scaler = GradScaler(init_scale=512, growth_interval=100)
    loader_bar = tqdm(ldr, desc="Training", dynamic_ncols=True, leave=False)

    all_risks, all_distills, all_times, all_events = [], [], [], []
    total_loss = 0
    acc_steps = 0
    accumulate_steps = 6
    for feats, times, events in loader_bar:
        for i in range(times.shape[0]):  # 遍历 batch 内每个 WSI
            bag_i = {level: [tensor[i].to(dev) for tensor in ts_list] for level, ts_list in feats.items()}
            with autocast(device_type='cuda'):
                l_s, m_s, h_s, risk = model(bag_i)
                distill_loss = distill(l_s, m_s, h_s, bag_i, model)
            scaler.scale(lmbd * distill_loss / accumulate_steps).backward(retain_graph=True)
            all_risks.append(risk)
            all_times.append(times[i].to(dev))
            all_events.append(events[i].to(dev))
            all_distills.append(distill_loss)
            acc_steps += 1
            
            if acc_steps == accumulate_steps:
                opt.zero_grad()
                risks_batch = torch.stack(all_risks)
                times_batch = torch.stack(all_times)
                events_batch = torch.stack(all_events)
                
                if events_batch.sum() == 0:
                    print("⚠️ Skip batch: All censored (no observed event)")
                    # all_risks.clear(); all_times.clear(); all_events.clear(); all_distills.clear()
                    # acc_steps = 0
                    # continue
                # distill_loss = torch.stack(all_distills).mean()
                else:
                    with autocast(device_type='cuda'):
                        loss = cox(risks_batch, times_batch, events_batch) 
                    # try:
                    #     ci = c_index(
                    #         times_batch.detach().cpu().numpy(),
                    #         events_batch.detach().cpu().numpy(),
                    #         risks_batch.detach().cpu().numpy()
                    #     )
                    #     print('c_index', ci)
                    # except Exception as e:
                    #     print(f"⚠️ Warning: Cannot compute c_index this batch ({str(e)}), skip.")

                    # + (lmbd * distill_loss)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                    distill_mean = sum([d.item() for d in all_distills]) / accumulate_steps
                    total_loss += loss.item()+ lmbd * distill_mean

                all_risks.clear(); all_times.clear(); all_events.clear()
                acc_steps = 0

    # 若剩余不足128个样本未处理
    if acc_steps > 0 and len(all_events) > 0 and torch.stack(all_events).sum() > 0:
        opt.zero_grad()
        risks_batch = torch.stack(all_risks)
        times_batch = torch.stack(all_times)
        events_batch = torch.stack(all_events)
        # distill_loss = torch.stack(all_distills).mean()
        with autocast(device_type='cuda'):
            loss = cox(risks_batch, times_batch, events_batch) 
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        distill_mean = sum([d.item() for d in all_distills]) / accumulate_steps
        total_loss += loss.item()+ lmbd * distill_mean

    return total_loss / max(1, len(ldr))

# def train_epoch(model, ldr, opt, cox, lmbd, dev):
#     model.train()
#     total_loss = 0
#     scaler = GradScaler(init_scale=512, growth_interval=100)
#     loader_bar = tqdm(ldr, desc="Training", dynamic_ncols=True, leave=False)
#     for feats, times, events in loader_bar:
#         batch_risks = []
#         # distill_loss = 0
#         distill_losses = []
#         for i in range(times.shape[0]):  # 遍历batch内每一个WSI
#             # bag_i = {l: [ts[i]] for l, ts in feats.items()}  # 拿到第i个WSI的low/mid/high
#             # bag_i = {l: [t.to(dev) for t in ts] for l, ts in bag_i.items()}  # 移到GPU
#             bag_i = {level: [tensor[i].to(dev) for tensor in ts_list] for level, ts_list in feats.items()}
#             with autocast(device_type='cuda'):
#                 l_s, m_s, h_s, risk = model(bag_i)
#                 dist_loss = distill(l_s, m_s, h_s, bag_i, model)
#             batch_risks.append(risk)
#             distill_losses.append(dist_loss)
#         batch_risks = torch.stack(batch_risks)  # [B]
#         distill_loss = torch.stack(distill_losses).mean()
#         times, events = times.to(dev), events.to(dev)
#         # print('cox',cox(batch_risks, times, events))
#         # 总loss = Cox loss + distillation loss
#         if events.sum() == 0:
#             print("⚠️ Skip batch: All censored (no observed event)")
#             continue
#         opt.zero_grad()
#         with autocast(device_type='cuda'):
#             loss = cox(batch_risks, times, events) + (lmbd * distill_loss )
#         # print('c_index',c_index(np.array(times), np.array(events), np.array(batch_risks)))
#         # print('c_index', c_index(
#         #     times.detach().cpu().numpy(),
#         #     events.detach().cpu().numpy(),
#         #     batch_risks.detach().cpu().numpy()
#         # ))
#         try:
#             ci = c_index(
#                 times.detach().cpu().numpy(),
#                 events.detach().cpu().numpy(),
#                 batch_risks.detach().cpu().numpy()
#             )
#             print('c_index', ci)
#         except Exception as e:
#             print(f"⚠️ Warning: Cannot compute c_index this batch ({str(e)}), skip.")

#         print(f"[Error] Loss value abnormal: {loss.item()}")
#         print(f"[Debug] Cox loss: {cox(batch_risks, times, events).item():.6f}")
#         print(f"[Debug] Distill loss: {distill_loss.item():.6f}")
#         # if not torch.isfinite(loss):
#         #     print(f"[Error] Loss value abnormal: {loss.item()}")
#         #     print(f"[Debug] Cox loss: {cox(batch_risks, times, events).item():.6f}")
#         #     print(f"[Debug] Distill loss: {distill_loss.item():.6f}")
#         #     raise ValueError("发现 loss NaN 或 Inf")
#         # opt.zero_grad()
#         # loss.backward()
#         scaler.scale(loss).backward()  # 用 scaler.scale
#         scaler.unscale_(opt)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可以设 1.0-5.0之间

#         scaler.step(opt)               #  scaler.step
#         scaler.update()                # scaler.update
#         for idx, mapper in enumerate(model.map_low):
#             if mapper.weight.grad is not None:
#                 g = mapper.weight.grad
#                 if not torch.isfinite(g).all():
#                     print(f"[Gradient Error] NaN/Inf detected in map_low[{idx}].weight.grad!")
#                     print(f"Grad min: {g.min().item():.6f}, max: {g.max().item():.6f}, mean: {g.mean().item():.6f}, std: {g.std().item():.6f}")
#                     print(f"NaN locations: {torch.isnan(g).nonzero()}")
#                     raise ValueError("Backward中，Linear权重的grad出现了NaN或者Inf！！")

#         # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) 
#         # opt.step()
#         total_loss += loss.item()

#     return total_loss / len(ldr)

# def _stats(name, t):
#     t32 = t.float()
#     print(f"[{name}] shape={tuple(t.shape)} "
#           f"min={t32.min().item():.4g}  max={t32.max().item():.4g} "
#           f"mean={t32.mean().item():.4g}  std={t32.std().item():.4g}")

# def train_epoch_dbg(model, loader, optimizer, cox_loss_fn, lambda_dist, device, max_steps=3):
#     scaler = GradScaler()
#     model.train()
#     torch.autograd.set_detect_anomaly(True)   # 打开 autograd 异常检测

#     for step, (feats, times, events) in enumerate(loader):
#         if step >= max_steps:
#             break
        
#         optimizer.zero_grad(set_to_none=True)

#         # 取 batch[0]，逐样本调试
#         bag0 = {lvl: [t[0].to(device) for t in feat_list] for lvl, feat_list in feats.items()}

#         # forward
#         with autocast(device_type='cuda', dtype=torch.float16):
#             l_s, m_s, h_s, risk = model(bag0)

#         _stats("low_out", l_s)
#         _stats("mid_out", m_s)
#         _stats("high_out", h_s)
#         _stats("risk", risk)

#         # distill
#         distill_loss = distill(l_s, m_s, h_s, bag0, model)
#         _stats("distill_loss", distill_loss)

#         # Cox loss
#         t0 = times[0:1].to(device)
#         e0 = events[0:1].to(device)
#         with autocast(device_type='cuda', dtype=torch.float16):
#             cox_loss = cox_loss_fn(risk.unsqueeze(0), t0, e0)
#         _stats("cox_loss", cox_loss)

#         # Total loss
#         total_loss = cox_loss + lambda_dist * distill_loss
#         _stats("total_loss", total_loss)

#         # backward
#         scaler.scale(total_loss).backward()

#         # Unscale and check gradients
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

#         for name, p in model.named_parameters():
#             if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
#                 _stats(f"❌ NaN or Inf grad in {name}", p.grad)
#                 raise RuntimeError(f"Gradient explosion detected in {name}!")

#         scaler.step(optimizer)
#         scaler.update()

@torch.no_grad()
def evaluate(model, ldr, dev):
    model.eval()
    risks, times, events = [], [], []
    loader_bar = tqdm(ldr, desc="Evaluating", dynamic_ncols=True, leave=False)
    for feats, t_batch, e_batch in loader_bar:
        for i in range(t_batch.shape[0]):

            bag_i = {level: [ts_list[i].to(dev) for ts_list in feats[level]] for level in ['low', 'mid', 'high']}
 
            _, _, _, risk = model(bag_i)

            risks.append(risk.item())
            times.append(t_batch[i].item())
            events.append(e_batch[i].item())

    return c_index(np.array(times), np.array(events), np.array(risks))

# ------------------------- main -------------------------- #
def main():
    set_seed(args.seed)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_df=pd.read_csv(args.csv)
    first_slide_path = full_df.loc[0, 'slide_id']
    slide_id = os.path.splitext(os.path.basename(first_slide_path))[0]
    # infer feature dims
    # 正确提取每个teacher的特征维度
    d_teachers = []
    for teacher_dir in TEACHER_DIRS:
        sample = torch.load(teacher_dir / f"{slide_id}.pt")
        low, _, _ = sample
        d_teachers.append(low.shape[1])

    print(d_teachers)  # 比如 [1536, 1536, 1536, 1024, 1024]


    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    print('hh')
    print(args.batch_size)
    for fold in folds:
        print(f'===== Fold {fold} =====')
        tr_df,val_df,tst_df = prepare_fold(full_df, fold)

        tr_loader = DataLoader(WSIDataset(tr_df ), batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
        val_loader= DataLoader(WSIDataset(val_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        tst_loader= DataLoader(WSIDataset(tst_df), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        # model=Student(d_teachers).to(dev)
        model = Student(d_teachers)
        # model = nn.DataParallel(model)
        model = model.to(dev)
        opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        cox=CoxPHLoss()

        best_val,best_test=0,0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for ep in epoch_bar:
            # train_epoch_dbg(model, tr_loader, opt, cox, args.lambda_dist, dev)
            tr_loss=train_epoch(model,tr_loader,opt,cox,args.lambda_dist,dev)
            val_c  =evaluate(model,val_loader,dev)
            test_c =evaluate(model,tst_loader,dev)
            if val_c>best_val: best_val,best_test=val_c,test_c
            sch.step()
            tqdm.write(f'ep{ep:02d} loss{tr_loss:.4f} val{val_c:.4f} test{test_c:.4f}')

        print(f'Fold {fold} | best val C {best_val:.4f} | best test C {best_test:.4f}')
        val_scores.append(best_val); test_scores.append(best_test)

    print('\\n===== 5‑fold summary =====')
    # print('val  mean {:.4f} ± {:.4f}'.format(np.mean(val_scores), np.std(val_scores)))
    # print('test mean {:.4f} ± {:.4f}'.format(np.mean(test_scores),np.std(test_scores)))
    val_mean = np.mean(val_scores)
    val_std = np.std(val_scores)
    val_ci95 = 1.96 * val_std / np.sqrt(5)

    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)
    test_ci95 = 1.96 * test_std / np.sqrt(5)

    print('\n===== 5-fold Summary =====')
    print(f'val mean {val_mean:.4f} ± {val_std:.4f} (95% CI: [{val_mean-val_ci95:.4f}, {val_mean+val_ci95:.4f}])')
    print(f'test mean {test_mean:.4f} ± {test_std:.4f} (95% CI: [{test_mean-test_ci95:.4f}, {test_mean+test_ci95:.4f}])')

if __name__=='__main__':
    main()


# python wsi_survival_splitcv.py \
#   --csv       all_cases.csv \
#   --splits_dir splits82/TCGA_BLCA_survival_100 \
#   --root      TCGA_BLCA_multi_features \
#   --teachers  gigapath_features hoptimus0_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 30 --lr 3e-4 --lambda_dist 0.1
# CUDA_VISIBLE_DEVICES=1 python multi_moe_distill_v2.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 --fold_idx 0 | tee multilog/BLCA0.txt
# CUDA_VISIBLE_DEVICES=0 python multi_moe_distill_v2.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 > runBLCA.txt 2>&1
# python multi_moe_distill_v2.py --csv dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir splits82/TCGA_KIRC_survival_100 --root TCGA_KIRC_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 100 --lr 2e-4 > runKIRC.txt 2>&1