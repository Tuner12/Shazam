#!/usr/bin/env python3
"""
DDP-version of WSI-level survival (5-fold CV).
Launch with:  torchrun --standalone --nproc_per_node 8 this_script.py <args>
"""
# ---------- std libs ----------
import argparse, os, random, math
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- torch ----------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sksurv.metrics import concordance_index_censored

# ========== 你的 Student / ABMIL / MoE 等所有网络组件保持不变 ==========
# ……（此处粘贴你原来的网络定义与 distill()、collate()、WSIDataset … 源码）……
ap = argparse.ArgumentParser()
ap.add_argument('--csv',       required=True, help='full meta CSV: wsi_id,survival_months,censorship')
ap.add_argument('--splits_dir',required=True, help='folder containing split0.csv … split4.csv')
ap.add_argument('--root',      required=True, help='root with <teacher>_features/pt_files')
ap.add_argument('--teachers',  nargs='+',    required=True, help='teacher folder names')
ap.add_argument('--epochs',    type=int,   default=25)
ap.add_argument('--lr',        type=float, default=3e-4)
ap.add_argument('--lambda_dist', type=float, default=0.1)
ap.add_argument('--seed',      type=int,   default=42)
ap.add_argument('--batch_size', type=int, default=2)
args = ap.parse_args()

TEACHER_DIRS = [Path(args.root)/f'{t}/merged_pt_files' for t in args.teachers]
N_TEACHERS   = len(TEACHER_DIRS)
D_MODEL      = 128
# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def c_index(times, events, risks):
    events_bool = (events == 1)
    return concordance_index_censored(events_bool, times, risks, tied_tol=1e-08)[0]

class CoxPHLoss(nn.Module):
    def forward(self, risk, time, event):
        # Step 1: manually build the "at risk" matrix
        n = len(time)
        R_mat = torch.zeros((n, n), dtype=torch.float32, device=risk.device)
        for i in range(n):
            for j in range(n):
                if time[j] >= time[i]:
                    R_mat[i, j] = 1.0

        # Step 2: compute loss based on R_mat
        theta = risk.reshape(-1)
        exp_theta = torch.exp(theta)

        # numerator: theta_i
        # denominator: log(sum_j exp(theta_j) where j >= i)
        loss = -((theta - torch.log(torch.matmul(R_mat, exp_theta))) * event).sum() / event.sum()

        return loss

# ---------------- split parser ------------------- #
# def load_split_df(split_idx:int)->pd.DataFrame:
#     path = Path(args.splits_dir)/f'splits_{split_idx}.csv'
#     if not path.exists():
#         raise FileNotFoundError(path)
#     # keep_blank_values to preserve empty cells ⇒ NaN
#     df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
#     return df

# def get_ids(df_split:pd.DataFrame, col:str)->List[str]:
#     return df_split[col].dropna().unique().tolist()
# def extract_case_id(slide_path):
#     filename = os.path.basename(slide_path)
#     return '-'.join(filename.split('-')[:3])
# def prepare_fold(full_df:pd.DataFrame, split_idx:int):
#     df_split=load_split_df(split_idx)
#     tr_ids = get_ids(df_split,'train')
#     val_ids= get_ids(df_split,'val')
#     tst_ids= get_ids(df_split,'test')
#     return (
#         full_df[full_df['slide_id'].apply(extract_case_id).isin(tr_ids)],
#         full_df[full_df['slide_id'].apply(extract_case_id).isin(val_ids)],
#         full_df[full_df['slide_id'].apply(extract_case_id).isin(tst_ids)]
#     )

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
    def __init__(self, input_dims, output_dim=512):
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
        concat_feats = torch.cat(features, dim=-1)  # [num_experts * output_dim]
        gate_scores = torch.softmax(self.gate(concat_feats), dim=-1)  # [num_experts]
        weighted_features = []
        for w, f in zip(gate_scores.transpose(0, 1), features):
            weighted_feat = self.out_ln(w.unsqueeze(-1) * f)  # [n_patch, d_model]
            weighted_features.append(weighted_feat)
        # 3. 加权融合
        # out = [self.out_ln(w * f) for w, f in zip(gate_scores, features)]

        # 4. 最后归一化
        return torch.stack(weighted_features, dim=1)



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
    def __init__(self, d_teachers, d_model=128, n_layers=4):
        super().__init__()
        self.d_model = d_model

        # 每个 teacher 一个 ABMIL 聚合器
        # self.pool_low  = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])
        # self.pool_mid  = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])
        # self.pool_high = nn.ModuleList([ABMIL(d_model) for _ in d_teachers])

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
            nn.Linear(128, 128//2),  # 先降维
            nn.GELU(),                     # 激活
            nn.Linear(128//2, 1)          # 再输出风险分数
        )

        self.abmil = ABMIL(d_model)
    def forward(self, bag):
        """
        bag: {'low': List of 5 tensors [n_patch, d_feat], 'mid': ..., 'high': ...}
        """
        # 1. 先把 patch-level特征映射到d_model
        low_proj  = [mapper(feat) for mapper, feat in zip(self.map_low,  bag['low'])]
        mid_proj  = [mapper(feat) for mapper, feat in zip(self.map_mid,  bag['mid'])]
        high_proj = [mapper(feat) for mapper, feat in zip(self.map_high, bag['high'])]
        # for idx, tensor in enumerate(low_proj):
        #     print(tensor.shape)
        # 2. 每个 teacher 自己 ABMIL pooling成一个向量
        low_feat = self.moe_low(low_proj)
        mid_feat = self.moe_mid(mid_proj)
        high_feat = self.moe_high(high_proj)

        # 3. Attention Cross — 输入 [n_patch, 5, d_model]，输出 [n_patch, d_model]
        low_out = self.cross_low(low_feat)  # [n_patch, d_model]
        mid_out = self.cross_mid(mid_feat)
        mid_out = self.res_ln2(mid_out + self.alpha * low_out)
        high_out = self.cross_high(high_feat)
        high_out = self.res_ln3(high_out + self.beta * mid_out)
        bag_feat = self.abmil(high_out) 
        # print("low_out",low_out.shape)
        risk = self.head(bag_feat).squeeze()
        
        return low_out, mid_out, high_out, risk


cos,huber=nn.CosineSimilarity(-1),nn.HuberLoss()
def dist_pair(s,t): return ((1 - cos(s, t)) + huber(s, t)).mean()
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
    for i in range(len(bag['low'])):
        # 1. teacher i 的 low/mid/high patch 特征
        low_feat = bag['low'][i]   # shape: [npatch, d_teacher]
        mid_feat = bag['mid'][i]
        high_feat = bag['high'][i]

        # 2. 通过 Student里的map_low等，统一到 d_model
        low_proj = model.map_low[i](low_feat)    # [npatch, d_model]
        mid_proj = model.map_mid[i](mid_feat)
        high_proj = model.map_high[i](high_feat)
        # print("low_proj",low_proj.shape)
        # print("ou1",out1.shape)
        # # 3. 每个 teacher 自己做 ABMIL池化成 wsi级特征
        # low_wsi_feat = model.pool_low[i](low_proj)    # [d_model]
        # mid_wsi_feat = model.pool_mid[i](mid_proj)
        # high_wsi_feat = model.pool_high[i](high_proj)
        # print("low_wsi_feat",low_wsi_feat.shape)
        # print(out1.shape)
        # 4. 蒸馏 loss
        loss += dist_pair(out1, low_proj)
        loss += dist_pair(out2, mid_proj)
        loss += dist_pair(out3, high_proj)

    return loss / (3 * len(bag['low']))



# ------------------- DDP helpers -------------------
def setup_ddp(rank: int, world_size: int):
    """Init default process group (single-node assumed)."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def all_gather_list(obj_list_local: list) -> list:
    """all_gather on Python objects (e.g. metric lists)."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj_list_local)
    # flatten
    return [x for sub in gathered for x in sub]

# ------------------- training loop -----------------
def train_one_epoch(model, loader, opt, cox_loss_fn, lmbd, device):
    model.train()
    sampler: DistributedSampler = loader.sampler      # type: ignore
    sampler.set_epoch(train_one_epoch.epoch)          # shuffle 每 epoch 变化
    train_one_epoch.epoch += 1                        # static var

    total_loss = 0.0
    loader_bar = tqdm(loader, desc="Training", dynamic_ncols=True, leave=False)
    for feats, times, events in loader_bar:
        batch_risks, distill_loss = [], 0.0
        batch_valid = 0
        for i in range(times.shape[0]):
            bag_i = {lvl: [ts_list[i].to(device) for ts_list in feats[lvl]]
                      for lvl in ('low', 'mid', 'high')}
            l_s, m_s, h_s, risk = model(bag_i)
            if not torch.isfinite(risk):
                tqdm.write(f"[Train Rank {dist.get_rank()}] ⚠️ Skipping NaN risk at batch_i={i}")
                continue
            batch_risks.append(risk)
            distill_loss += distill(l_s, m_s, h_s, bag_i, model.module)
        if batch_valid == 0:
            continue  # 当前batch全是NaN，跳过
        batch_risks = torch.stack(batch_risks)        # [B]
        times, events = times.to(device), events.to(device)
        # loss = cox_loss_fn(batch_risks, times, events) \
        #      + lmbd * distill_loss / times.shape[0]
        loss = cox_loss_fn(batch_risks, times[:batch_valid], events[:batch_valid]) \
             + lmbd * distill_loss / batch_valid
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()

    # 聚合 loss（取平均）
    loss_tensor = torch.tensor(total_loss / len(loader), device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    mean_loss = (loss_tensor / dist.get_world_size()).item()
    return mean_loss
train_one_epoch.epoch = 0  # static variable

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    risk_list, time_list, event_list = [], [], []
    loader_bar = tqdm(loader, desc="Evaluating", dynamic_ncols=True, leave=False)
    for feats, t_batch, e_batch in loader_bar:
        for i in range(t_batch.shape[0]):
            bag_i = {lvl: [ts_list[i].to(device) for ts_list in feats[lvl]]
                      for lvl in ('low', 'mid', 'high')}
            _, _, _, risk = model(bag_i)
            risk_list.append(risk.item())
            if not math.isfinite(risk.item()):
                tqdm.write(f"[Rank {dist.get_rank()}] ⚠️ NaN risk found at batch_i={i}, skipping.")
                continue  # 跳过这个样本
            time_list.append(t_batch[i].item())
            event_list.append(e_batch[i].item())

    # gather to rank0
    risk_all  = all_gather_list(risk_list)
    time_all  = all_gather_list(time_list)
    event_all = all_gather_list(event_list)

    if dist.get_rank() == 0:
        return c_index(np.array(time_all), np.array(event_all), np.array(risk_all))
    else:
        return 0.0    # 非 rank0 不用此值

# ------------------- main per-process ----------------
def run_ddp(rank: int, world_size: int, args):
    setup_ddp(rank, world_size)
    set_seed(args.seed + rank)          # 每进程不同 seed

    # ---------- 根目录与 TEACHER_DIRS ----------
    teacher_dirs = [Path(args.root) / f"{t}/merged_pt_files" for t in args.teachers]

    # ---------- 读 meta CSV ----------
    full_df = pd.read_csv(args.csv)
    slide_id0 = Path(full_df.loc[0, "slide_id"]).stem
    d_teachers = []
    for d in teacher_dirs:
        low, _, _ = torch.load(d / f"{slide_id0}.pt", map_location="cpu", weights_only=True)
        d_teachers.append(low.shape[1])

    # ---------- 5 folds ----------
    best_val_all, best_test_all = [], []
    for fold in range(5):
        if rank == 0:
            print(f"\n===== Fold {fold} (rank0 prints only) =====")

        # ------- split & datasets -------
        def load_split_df(i):
            df = pd.read_csv(Path(args.splits_dir) / f"splits_{i}.csv",
                             dtype=str, keep_default_na=False, na_values=[''])
            return df
        def get_ids(df, col): return df[col].dropna().unique().tolist()
        def extract_case_id(p): return "-".join(Path(p).stem.split("-")[:3])

        df_split = load_split_df(fold)
        tr_ids = get_ids(df_split, "train")
        val_ids= get_ids(df_split, "val")
        tst_ids= get_ids(df_split, "test")

        def sel(df, ids): return df[df["slide_id"].apply(extract_case_id).isin(ids)]
        tr_df,val_df,tst_df = sel(full_df,tr_ids), sel(full_df,val_ids), sel(full_df,tst_ids)

        # ------- samplers & loaders -------
        def make_loader(df, shuffle):
            dataset = WSIDataset(df)
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            return DataLoader(dataset,
                              batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=collate)
        tr_loader = make_loader(tr_df, True)
        val_loader= make_loader(val_df, False)
        tst_loader= make_loader(tst_df, False)

        # ------- model / opt -------
        device = torch.device('cuda', rank)
        model = Student(d_teachers).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        cox = CoxPHLoss()

        best_val, best_test = 0.0, 0.0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for ep in epoch_bar:
            tr_loss = train_one_epoch(model, tr_loader, opt, cox, args.lambda_dist, device)
            val_c   = eval_epoch(model, val_loader, device)
            test_c  = eval_epoch(model, tst_loader, device)
            sch.step()

            # 只在 rank0 打印与记录
            if rank == 0:
                print(f"[Fold {fold} | Ep {ep:02d}] "
                      f"loss {tr_loss:.4f}  valC {val_c:.4f}  testC {test_c:.4f}")
                if val_c > best_val:
                    best_val, best_test = val_c, test_c

        # 记录
        best_val_all.append(best_val); best_test_all.append(best_test)

    # ---------- 5-fold summary ----------
    if rank == 0:
        def summar(x):
            m, s = np.mean(x), np.std(x)
            ci95  = 1.96 * s / np.sqrt(5)
            return f"{m:.4f} ± {s:.4f} (95% CI [{m-ci95:.4f},{m+ci95:.4f}])"
        print("\n===== 5-fold Summary (rank0) =====")
        print("val  :", summar(best_val_all))
        print("test :", summar(best_test_all))

    cleanup_ddp()

# -------------------- CLI & LAUNCH --------------------
def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--splits_dir', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--teachers', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda_dist', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=2)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))   # torchrun 会自动设置
    run_ddp(rank, world_size, args) 
# torchrun --standalone --nproc_per_node 8 multi_moe_distill_DDP.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 100 --lr 2e-4 --lambda_dist 0.01 | tee runBLCA_ddp.txt
# torchrun --standalone --nproc_per_node 8 multi_moe_distill_DDP.py --csv dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir splits82/TCGA_KIRC_survival_100 --root TCGA_KIRC_multi_features --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 100 --lr 2e-4 --lambda_dist 0.01 | tee runKIRC_ddp.txt