# #!/usr/bin/env python3
# """
# WSI‑level survival · 5‑fold CV using split<i>.csv  (ID‑enumeration format)
# --------------------------------------------------------------------------

# `splits_dir` 必须含有  ↓↓↓
#     split0.csv
#     split1.csv
#     ...
#     split4.csv

# 每个文件格式示例
# ----------------
# train,val,test
# TCGA-DK-A31K,,
# ,TCGA-CU-A0YO,
# ,,TCGA-CU-A0YO
# ...

# ⚠️ 只读取这 5 个文件；`split_bool<i>.csv` 完全忽略。
# """

# -------------------- Imports -------------------- #
import argparse, random, math
from pathlib import Path
from typing import List, Dict
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored
from scipy import stats
from wsi_dataset import WSIDataset, collate, extract_case_id, is_valid_slide
# from torch import amp
# from torch import autocast
# from torch.amp import GradScaler
from torch.amp import GradScaler, autocast
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
ap.add_argument('--batch_size', type=int, default=8)
ap.add_argument('--n_bins', type=int, default=4, help='Number of time bins for discretization (default: 4)')
args = ap.parse_args()

TEACHER_DIRS = [Path(args.root)/f'{t}/merged_pt_files' for t in args.teachers]
N_TEACHERS   = len(TEACHER_DIRS)
D_MODEL      = 128
# BATCH_SIZE   = 1

# ---------------- reproducibility ---------------- #
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
#     def __init__(self):
#         super().__init__()

#     def forward(self, risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor):
#         """
#         计算稳定的 Cox 部分似然损失。
#         - risk: 模型预测的风险分数（越大风险越高），shape = [B]
#         - time: 生存时间，shape = [B]
#         - event: 事件指示（1=死亡，0=删失/存活），shape = [B]
#         """
#         # 按时间降序排序，使早死亡的人排前面
#         t, order = time.sort(descending=True)
#         risk = risk[order].float()
#         event = event[order]

#         # 数值稳定：截断 risk 防止 exp 爆炸
#         # theta = torch.clamp(risk, min=-20.0, max=20.0)
#         log_den = torch.logcumsumexp(risk, dim=0)
#         # log_den = torch.nan_to_num(log_den, nan=0.0, posinf=0.0, neginf=-20.0)

#         # 计算部分似然的负 log（注意 event.sum 可能为 0）
#         loss = -((risk - log_den) * event).sum() / (event.sum() + 1e-8)
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

# ---------------- split parser ------------------- #
# 旧的prepare_fold函数已被移除，现在使用WSIDataset.create_fold_datasets方法


# ---------------- Dataset ------------------------ #
# WSIDataset, collate, extract_case_id, is_valid_slide 已从 wsi_dataset.py 导入


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
    def __init__(self, d_teachers, d_model=128, n_layers=4, n_bins=4):
        super().__init__()
        self.d_model = d_model
        self.n_bins = n_bins  # 离散时间区间数量

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

        # 修改：输出离散时间的 hazards（每个 bin 的风险概率）
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_bins),  # 输出 n_bins 个 logits
            nn.Sigmoid()  # 转换为概率 [0, 1]
        )

        # self.abmil = ABMIL(d_model)


    def forward(self, bag):
        """
        bag: {'low': List of 5 tensors [n_patch, d_feat], 'mid': ..., 'high': ...}
        返回：
            - low_out, mid_out, high_out: 中间层特征
            - low_vec, mid_vec, high_vec: teacher 特征向量列表
            - hazards: (n_bins,) 每个时间区间的风险概率
        """

        low_proj  = [mapper(feat) for mapper, feat in zip(self.map_low,  bag['low'])]
        mid_proj  = [mapper(feat) for mapper, feat in zip(self.map_mid,  bag['mid'])]
        high_proj = [mapper(feat) for mapper, feat in zip(self.map_high, bag['high'])]

        low_vec = [pool(feat) for pool, feat in zip(self.pool_low,  low_proj)]
        mid_vec = [pool(feat) for pool, feat in zip(self.pool_mid,  mid_proj)]
        high_vec = [pool(feat) for pool, feat in zip(self.pool_high,  high_proj)]

        low_feat = self.moe_low(low_vec).unsqueeze(0)
        mid_feat = self.moe_mid(mid_vec).unsqueeze(0)
        high_feat = self.moe_high(high_vec).unsqueeze(0)
  
        low_out = self.cross_low(low_feat)  # [1, d_model]
        mid_out = self.cross_mid(mid_feat)
        mid_out = self.res_ln2(mid_out + self.alpha * low_out)
        high_out = self.cross_high(high_feat)
        high_out = self.res_ln3(high_out + self.beta * mid_out)
   
        hazards = self.head(high_out).squeeze()  # (n_bins,) 每个 bin 的风险概率
        return low_out, mid_out, high_out, low_vec, mid_vec, high_vec, hazards


def dist_pair(student_feat, teacher_feat, eps=1e-8):
    """
    (1 - cos_sim) + SmoothL1 loss，防止除以0导致NaN
    """
    # 防止 norm 很小导致的 NaN
    student_feat = student_feat.squeeze(0)
    student_norm = student_feat.norm(dim=-1, keepdim=True).clamp(min=eps)
    teacher_norm = teacher_feat.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos_sim = (student_feat * teacher_feat).sum(dim=-1) / (student_norm * teacher_norm).squeeze(-1)
    cos_term = 1.0 - cos_sim.mean()

    smooth_l1 = nn.HuberLoss()(student_feat, teacher_feat)
    return cos_term + smooth_l1
def distill(out1, out2, out3, low_vecs, mid_vecs, high_vecs):
    """
    out1, out2, out3: Student 输出的融合特征（low_out, mid_out, high_out），shape = [1, d_model]
    *_vecs: 每个 teacher 聚合的特征向量（List[Tensor[d_model]]）
    """
    loss = 0
    for i in range(len(low_vecs)):
        # tqdm.write(f"[distill] low_vecs[{i}].shape = {low_vecs[i].shape}")
        # tqdm.write(f"out.shape = {out1.shape}")
        # print("low_vecs[i]",low_vecs[i].shape)
        loss += dist_pair(out1, low_vecs[i])
        loss += dist_pair(out2, mid_vecs[i])
        loss += dist_pair(out3, high_vecs[i])
    return loss / (3 * len(low_vecs))
# def distill(out1, out2, out3, bag, model):
#     """
#     out1, out2, out3: model输出的low/mid/high (每个都是 [d_model])
#     bag: {'low': [teacher1_patches, teacher2_patches, ...], ...}
#     model: Student模型，用来取map_low/map_mid/map_high
#     """
#     loss = 0
#     # with torch.no_grad(): 
#     for i in range(len(bag['low'])):
#         # 1. teacher i 的 low/mid/high patch 特征
#         low_feat = bag['low'][i]   # shape: [npatch, d_teacher]
#         mid_feat = bag['mid'][i]
#         high_feat = bag['high'][i]
#         low_proj = model.map_low[i](low_feat)    # [npatch, d_model]
#         mid_proj = model.map_mid[i](mid_feat)
#         high_proj = model.map_high[i](high_feat)
#         low_vec = model.pool_low[i](low_proj).unsqueeze(0)
#         mid_vec = model.pool_mid[i](mid_proj).unsqueeze(0)
#         high_vec = model.pool_high[i](high_proj).unsqueeze(0)
#         # n_patch = high_vec.size(0)
#         # print("low_proj",low_proj.shape)
#         # print("ou1",out1.shape)
#         # # 3. 每个 teacher 自己做 ABMIL池化成 wsi级特征
#         # low_wsi_feat = model.pool_low[i](low_proj)    # [d_model]
#         # mid_wsi_feat = model.pool_mid[i](mid_proj)
#         # high_wsi_feat = model.pool_high[i](high_proj)
#         # print("low_wsi_feat",low_wsi_feat.shape)
#         # print(out1.shape)
#         # 4. 蒸馏 loss
#         # loss += dist_pair(out1, low_proj)
#         # loss += dist_pair(out2, mid_proj)
#         # loss += dist_pair(out3, high_proj)
#         # if not torch.isfinite(loss).all():
#         #     print(f"[NaN detected in final distill loss]")
#         #     raise ValueError("Final distill loss NaN!")
#         # print('low_vec',low_vec.shape)
#         # print('out1',out1.shape)
        
#         loss += dist_pair(out1, low_vec)
   

   
#         loss += dist_pair(out2, mid_vec) 
   

        
#         loss += dist_pair(out3, high_vec) 
   
#         del low_feat, mid_feat, high_feat, low_proj, mid_proj, high_proj, low_vec, mid_vec, high_vec
#         # torch.cuda.empty_cache()
#         # print(loss)
#         # print(bag['low'][0].shape[0])
#         # print(3 * len(bag['low']))
#         # print(loss / (3 * len(bag['low'])))
#     return loss / (3 * len(bag['low']))



# -------------- train & eval helpers -------------- #
def train_epoch(model, ldr, opt, lmbd, dev, n_bins, alpha=0.4):
    model.train()
    scaler = GradScaler(device='cuda')
    loader_bar = tqdm(ldr, desc="Training", dynamic_ncols=True, leave=False)

    all_hazards, all_S, all_distills, all_Y, all_c, all_times = [], [], [], [], [], []
    total_loss = 0
    cindex_sum = 0.0
    cindex_count = 0
    acc_steps = 0
    accumulate_steps = 8
    opt.zero_grad(set_to_none=True)
    
    for step, (feats, times, censorships, Y_bins, _) in enumerate(loader_bar):
        # 跳过空批次（collate 可能在整批样本均无特征时返回 None）
        if feats is None or times is None or censorships is None or Y_bins is None:
            continue
        batch_size = times.shape[0]
        for i in range(batch_size):  # 遍历 batch 内每个 WSI
            bag_i = {level: [tensor[i].to(dev) for tensor in ts_list] for level, ts_list in feats.items()}
            
            with autocast(device_type='cuda'):
                l_s, m_s, h_s, l_v, m_v, h_v, hazards = model(bag_i)
                distill_loss = distill(l_s, m_s, h_s, l_v, m_v, h_v)
            
            # hazards shape: (n_bins,) -> 需要变成 (1, n_bins)
            hazards = hazards.unsqueeze(0)  # (1, n_bins)
            
            # 计算生存函数 S
            S = torch.cumprod(1.0 - hazards.clamp(min=1e-7, max=1.0-1e-7), dim=1)  # (1, n_bins)
            
            # 获取标签
            Yi = Y_bins[i].to(dev).long().view(1).clamp(min=1, max=n_bins)  # (1,) 1-based
            ci = censorships[i].to(dev).float().view(1)  # (1,)
            ti = times[i].to(dev).float()  # ()
            
            all_hazards.append(hazards)  # (1, n_bins)
            all_S.append(S)  # (1, n_bins)
            all_Y.append(Yi)  # (1,)
            all_c.append(ci)  # (1,)
            all_times.append(ti)  # ()
            all_distills.append(distill_loss)
            acc_steps += 1
            
            is_last_sample = (step == len(ldr) - 1) and (i == batch_size - 1)
            if acc_steps == accumulate_steps or is_last_sample:
                # 合并成 batch
                hazards_batch = torch.cat(all_hazards, dim=0)  # (B, n_bins)
                S_batch = torch.cat(all_S, dim=0)  # (B, n_bins)
                Y_batch = torch.cat(all_Y, dim=0).view(-1, 1)  # (B, 1) 1-based
                c_batch = torch.cat(all_c, dim=0).view(-1, 1)  # (B, 1)
                times_batch = torch.stack(all_times)  # (B,)
                
                # NLL loss
                with autocast(device_type='cuda'):
                    survival_loss = nll_loss(
                        hazards=hazards_batch, 
                        S=S_batch,
                        Y=Y_batch,  # 1-based
                        c=c_batch,  # 1=删失, 0=死亡
                        alpha=alpha
                    )
                
                distill_mean = sum(all_distills) / len(all_distills)
                loss = lmbd * distill_mean + survival_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                
                total_loss += loss.item()
                
                # 计算 C-index (参考代码风格)
                risk = (-torch.sum(S_batch, dim=1)).detach()  # 生存概率越低，风险越高
                try:
                    times_np = times_batch.detach().cpu().numpy()
                    events_np = (1.0 - c_batch.view(-1)).detach().cpu().numpy()  # 1=事件, 0=删失
                    risks_np = risk.detach().cpu().numpy()
                    cidx = c_index(times_np, events_np, risks_np)
                    cindex_sum += cidx
                    cindex_count += 1
                except Exception:
                    pass
                
                all_hazards.clear()
                all_S.clear()
                all_Y.clear()
                all_c.clear()
                all_times.clear()
                all_distills.clear()
                acc_steps = 0

    avg_loss = total_loss / max(1, len(ldr))
    avg_cindex = cindex_sum / max(1, cindex_count) if cindex_count > 0 else 0.0
    return avg_loss, avg_cindex

# def train_epoch(model, ldr, opt, cox, lmbd, dev):


@torch.no_grad()
def evaluate(model, ldr, dev):
    """
    评估模型：使用生存函数 S 计算风险分数用于 C-index
    风险分数 = -sum(S)，生存概率越低，风险越高
    """
    model.eval()
    risks, times, events = [], [], []
    loader_bar = tqdm(ldr, desc="Evaluating", dynamic_ncols=True, leave=False)
    
    for feats, t_batch, c_batch, Y_batch, _ in loader_bar:
        # 跳过空批次
        if feats is None or t_batch is None or c_batch is None or Y_batch is None:
            continue
        for i in range(t_batch.shape[0]):
            bag_i = {level: [ts_list[i].to(dev) for ts_list in feats[level]] for level in ['low', 'mid', 'high']}
            _, _, _, _, _, _, hazards = model(bag_i)
            
            # 计算生存函数 S
            hazards = hazards.unsqueeze(0)  # (1, n_bins)
            S = torch.cumprod(1.0 - hazards.clamp(min=1e-7, max=1.0-1e-7), dim=1)  # (1, n_bins)
            
            # 风险分数：负的生存概率总和（参考代码风格）
            risk_score = (-torch.sum(S)).item()
            
            risks.append(risk_score)
            times.append(t_batch[i].item())
            events.append((1.0 - c_batch[i]).item())  # 转换为 event: 1=死亡, 0=删失

    return c_index(np.array(times), np.array(events), np.array(risks))

@torch.no_grad()
def inference(model, ldr, dev, save_path):
    """
    推理函数，保存预测结果到JSON文件
    
    输出字段说明：
    - slide_id: WSI切片ID
    - pred_hazards: 模型预测的每个时间区间的风险概率
    - pred_survival: 预测的生存概率 S(t) 
    - pred_risk: 风险分数（-sum(S)，越高风险越大）
    - survival_months: 生存时间（月）
    - Y: 离散化的bin索引 (1-based)
    - censorship: 删失标记 (1=删失, 0=死亡)
    """
    model.eval()
    results = []
    
    for feats, t_batch, c_batch, Y_batch, slides in tqdm(ldr, desc="Infer", leave=False):
        # 跳过空批次
        if feats is None or t_batch is None or c_batch is None or Y_batch is None or slides is None:
            continue
        B = t_batch.size(0)
        for i in range(B):
            bag = {lvl: [t[i].to(dev) for t in feats[lvl]] for lvl in feats}
            *_, hazards = model(bag)
            
            # 计算生存函数
            hazards = hazards.unsqueeze(0)
            S = torch.cumprod(1.0 - hazards.clamp(min=1e-7, max=1.0-1e-7), dim=1)
            risk_score = (-torch.sum(S)).item()
            
            results.append({
                "slide_id": slides[i],
                "pred_hazards": hazards.squeeze(0).cpu().tolist(),
                "pred_survival": S.squeeze(0).cpu().tolist(),
                "pred_risk": risk_score,
                "survival_months": t_batch[i].item(),
                "Y": Y_batch[i].item(),  # 1-based
                "censorship": c_batch[i].item()
            })
    
    output_data = {
        "description": {
            "pred_hazards": "predicted hazard probabilities h(t) for each time bin",
            "pred_survival": "predicted survival probabilities S(t) = prod(1-h)",
            "pred_risk": "risk score = -sum(S), higher=worse prognosis",
            "censorship": "1=censored (alive), 0=death event",
            "Y": "discretized bin index (1-based, for nll_loss)",
            "survival_months": "survival time in months"
        },
        "results": results
    }
    
    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=2)

# ------------------------- main -------------------------- #
def main():
    set_seed(args.seed)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_df = pd.read_csv(args.csv)
    
    # 离散化处理现在由WSIDataset自动完成
    n_bins = args.n_bins
    print(f"使用 {n_bins} 个bins进行生存时间离散化")
    
    # 推断特征维度
    first_slide_path = full_df.loc[0, 'slide_id']
    slide_id = os.path.splitext(os.path.basename(first_slide_path))[0]
    d_teachers = []
    for teacher_dir in TEACHER_DIRS:
        sample = torch.load(teacher_dir / f"{slide_id}.pt", weights_only=True)
        low, _, _ = sample
        d_teachers.append(low.shape[1])
    
    print(f"教师特征维度: {d_teachers}")
    
    # 定义splits_dir路径
    splits_dir = Path(args.splits_dir)
    
    dataset_name = Path(args.csv).stem.split('_')[1]
    ckpt_root = Path("checkpoints_brca") / dataset_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    
    for fold in folds:
        print(f'\n===== Fold {fold} =====')
        
        # 使用新的WSIDataset.create_fold_datasets方法创建数据集
        # 自动展开多个slide为独立样本，使用bool类型的fold文件
        train_dataset, val_dataset, test_dataset = WSIDataset.create_fold_datasets(
            full_df, TEACHER_DIRS, splits_dir, fold, n_bins=n_bins, 
            create_Y=True, expand_multi_slides=True
        )
        
        tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        tst_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        model = Student(d_teachers, n_bins=n_bins).to(dev)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_val = 0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for ep in epoch_bar:
            tr_loss, tr_cindex = train_epoch(model, tr_loader, opt, args.lambda_dist, dev, n_bins)
            val_c = evaluate(model, val_loader, dev)
            
            if val_c > best_val:
                best_val = val_c
                pt = ckpt_root / f"fold{fold}_multi_nll_bestval.pt"
                torch.save(model.state_dict(), pt)
            
            sch.step()
            tqdm.write(f'Epoch {ep:02d} | loss: {tr_loss:.4f} | train C-index: {tr_cindex:.4f} | val C-index: {val_c:.4f}')

        print(f'Fold {fold} | Best val C-index: {best_val:.4f}')
        val_scores.append(best_val)
        
        # 测试集评估
        ckpt = ckpt_root / f"fold{fold}_multi_nll_bestval.pt"
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        test_c = evaluate(model, tst_loader, dev)
        test_scores.append(test_c)
        
        out_json = ckpt_root / f"fold{fold}_multi_nll_test_preds.json"
        inference(model, tst_loader, dev, out_json)

    print('\\n===== 5‑fold summary =====')
    # print('val  mean {:.4f} ± {:.4f}'.format(np.mean(val_scores), np.std(val_scores)))
    # print('test mean {:.4f} ± {:.4f}'.format(np.mean(test_scores),np.std(test_scores)))
    val_mean = np.mean(val_scores)
    val_std = np.std(val_scores, ddof=1)
    val_ci95 = stats.t.ppf(0.975, df=4) * val_std / np.sqrt(5)


    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores, ddof=1)
    test_ci95 = stats.t.ppf(0.975, df=4) * test_std / np.sqrt(5)

    print('\n===== 5-fold Summary =====')
    print(f'val mean {val_mean:.4f} ± {val_std:.4f} (95% CI: [{val_mean-val_ci95:.4f}, {val_mean+val_ci95:.4f}])')
    print(f'test mean {test_mean:.4f} ± {test_std:.4f} (95% CI: [{test_mean-test_ci95:.4f}, {test_mean+test_ci95:.4f}])')

if __name__=='__main__':
    main()


# =============================================================================
# 多教师模型蒸馏训练指令 - 使用更新后的WSIDataset类
# =============================================================================
# 
# 基本命令格式:
# CUDA_VISIBLE_DEVICES=<GPU_ID> python shazam_survival_analysis.py \
#   --csv <CSV文件路径> \
#   --splits_dir <fold分割目录> \
#   --root <特征根目录> \
#   --teachers <教师模型列表> \
#   --epochs <训练轮数> \
#   --lr <学习率> \
#   --batch_size <批次大小> \
#   --fold_idx <指定fold> \
#   --n_bins <时间分箱数>
#
# 参数说明:
# --csv: 包含slide_id, survival_months, censorship列的CSV文件
# --splits_dir: 包含splits_0_bool.csv到splits_4_bool.csv的目录
# --root: 包含各教师模型特征目录的根目录
# --teachers: 教师模型目录名称列表，用空格分隔
# --epochs: 训练轮数，默认20
# --lr: 学习率，默认2e-4
# --batch_size: 批次大小，默认8
# --fold_idx: 指定单个fold训练，不指定则训练所有fold
# --n_bins: 生存时间分箱数，默认4
#
# =============================================================================

# 示例训练命令 (按数据集分组):

# === 肾脏相关数据集 ===
# CUDA_VISIBLE_DEVICES=3 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/KIRC_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=0 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_SKCM_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_SKCM_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_SKCM_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/SKCM_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=4 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRP_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRP_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRP_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 8 \
#   | tee multilog_final/KIRP_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=5 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KICH_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KICH_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KICH_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 8 \
#   | tee multilog_final/KICH_multi_teacher.txt

# === 肺癌相关数据集 ===
# CUDA_VISIBLE_DEVICES=0 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_LUAD_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_LUAD_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_LUAD_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/LUAD_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=2 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_LUSC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_LUSC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_LUSC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/LUSC_multi_teacher.txt

# === 其他癌症数据集 ===
# CUDA_VISIBLE_DEVICES=3 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_BLCA_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_BLCA_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/BLCA_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=6 python shazam_survival_analysis.py \
#   --csv /data2/tanyusheng/Code/Survival/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv \
#   --splits_dir /data2/tanyusheng/Code/Survival/splits82/TCGA_BRCA_survival_100 \
#   --root /data2/tanyusheng/Data/Extracted_Feature/TCGA_BRCA_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/BRCA_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=3 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_STAD_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_STAD_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_STAD_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/STAD_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=1 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_CESC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_CESC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_CESC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/CESC_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=3 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_GBMLGG_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_GBMLGG_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_GBM_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/GBMLGG_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=0 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_HNSC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_HNSC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_HNSC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 8 \
#   | tee multilog_final/HNSC_multi_teacher.txt

# CUDA_VISIBLE_DEVICES=1 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_COADREAD_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_COADREAD_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_COADREAD_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4 --batch_size 4 \
#   | tee multilog_final/COADREAD_multi_teacher.txt

# =============================================================================
# 单fold训练示例 (用于调试或快速测试):
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 python shazam_survival_analysis.py \
#   --csv /nas/leiwenhui/tys/survival_analysis/dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv \
#   --splits_dir /nas/leiwenhui/tys/survival_analysis/splits82/TCGA_KIRC_survival_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 5 --lr 2e-4 --batch_size 4 --fold_idx 0 \
#   | tee multilog_final/KIRC_fold0_test.txt

# =============================================================================
# 注意事项:
# =============================================================================
# 1. 确保所有路径存在且可访问
# 2. 根据GPU内存调整batch_size
# 3. 使用不同的CUDA_VISIBLE_DEVICES避免GPU冲突
# 4. 训练日志会自动保存到multilog_final/目录
# 5. 模型检查点会保存到checkpoints_nll/目录
# 6. 新版本支持自动展开多个slide为独立样本
# 7. 使用bool类型的fold文件进行数据分割
# 8. 自动进行生存时间离散化处理
