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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
# from torch import amp
# from torch import autocast
# from torch.amp import GradScaler
from torch.amp import GradScaler, autocast
# scaler = amp.GradScaler(device='cuda')

# -------------------- CLI ------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument('--csv',       required=True, help='full meta CSV: wsi_id,survival_months,censorship')
ap.add_argument('--splits_dir',required=True, help='folder containing split0.csv … split4.csv')
ap.add_argument('--root',      nargs='+', required=True, help='root(s) with <teacher>_features/pt_files')
ap.add_argument('--teachers',  nargs='+',    required=True, help='teacher folder names')
ap.add_argument('--epochs',    type=int,   default=20)
ap.add_argument('--lr',        type=float, default=2e-4)
ap.add_argument('--lambda_dist', type=float, default=0.01)
ap.add_argument('--seed',      type=int,   default=42)
ap.add_argument('--fold_idx', type=int, default=None, help='Specify a single fold to run (0-4); if not set, all folds will be run')

ap.add_argument('--batch_size', type=int, default=4)
args = ap.parse_args()

TEACHER_DIRS = [Path(r)/f'{t}/merged_pt_files' for r in args.root for t in args.teachers]
N_TEACHERS   = len(TEACHER_DIRS)
D_MODEL      = 128
# BATCH_SIZE   = 1

# ---------------- reproducibility ---------------- #
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)




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

def prepare_fold(full_df: pd.DataFrame, split_idx: int):
    df_split = load_split_df(split_idx)
    tr_ids = get_ids(df_split, 'train')
    val_ids = get_ids(df_split, 'val')
    tst_ids = get_ids(df_split, 'test')

    def is_valid_slide(slide_path):
        # 检查WSI是否在某个完整的数据集中存在（所有teacher目录）
        # 假设每个数据集有相同数量的teacher
        teachers_per_dataset = len(args.teachers)
        
        for i in range(0, len(TEACHER_DIRS), teachers_per_dataset):
            # 检查一个完整的数据集
            dataset_teachers = TEACHER_DIRS[i:i+teachers_per_dataset]
            if all((d / f'{slide_path}.pt').exists() for d in dataset_teachers):
                return True
        return False

    train_df = full_df[full_df['slide_id'].isin(tr_ids)]
    val_df   = full_df[full_df['slide_id'].isin(val_ids)]
    test_df  = full_df[full_df['slide_id'].isin(tst_ids)]

    # 过滤掉缺失的文件
    train_df = train_df[train_df['slide_id'].apply(is_valid_slide)]
    val_df   = val_df[val_df['slide_id'].apply(is_valid_slide)]
    test_df  = test_df[test_df['slide_id'].apply(is_valid_slide)]

    return train_df, val_df, test_df


# ---------------- Dataset ------------------------ #
class WSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row['slide_id']
        
        bag = {'low': [], 'mid': [], 'high': []}
        
        # 找到包含该WSI的完整数据集
        teachers_per_dataset = len(args.teachers)
        wsi_found = False
        
        for i in range(0, len(TEACHER_DIRS), teachers_per_dataset):
            # 检查一个完整的数据集
            dataset_teachers = TEACHER_DIRS[i:i+teachers_per_dataset]
            if all((d / f'{slide_path}.pt').exists() for d in dataset_teachers):
                # 找到了包含该WSI的完整数据集，读取所有teacher的特征
                wsi_found = True
                for teacher_dir in dataset_teachers:
                    pt_file = teacher_dir / f"{slide_path}.pt"
                    low_t, mid_t, high_t = torch.load(pt_file, map_location='cpu', weights_only=True)
                    bag['low'].append(low_t)
                    bag['mid'].append(mid_t)
                    bag['high'].append(high_t)
                break
        
        if not wsi_found:
            raise FileNotFoundError(f"No complete dataset found for WSI {slide_path}")

        label_idx = torch.tensor(self.label_to_idx[row['label']],dtype=torch.long)
        return bag, label_idx, slide_path


def collate(batch):
    feats = {'low': [], 'mid': [], 'high': []}
    labels, slides = [], []

    for bag, label, slide in batch:
        for k in feats:
            for i, t in enumerate(bag[k]):
                if len(feats[k]) <= i:
                    feats[k].append([])
                feats[k][i].append(t)
        labels.append(label)
        slides.append(slide)

    # 不要stack！！！直接保持list of tensors
    return feats, torch.stack(labels),slides


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
    def __init__(self, d_teachers, num_classes, d_model=128, n_layers=4):
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
        # 使用每个数据集的teacher数量（5个）
        teachers_per_dataset = len(args.teachers)
        self.moe_low  = MoE([d_model] * teachers_per_dataset, d_model)
        self.moe_mid  = MoE([d_model] * teachers_per_dataset, d_model)
        self.moe_high = MoE([d_model] * teachers_per_dataset, d_model)

        # attention
        self.cross_low = MultiCross(d_model, n_layers)
        self.cross_mid = MultiCross(d_model, n_layers)
        self.cross_high = MultiCross(d_model, n_layers)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        # residual
        self.res_ln2 = nn.LayerNorm(d_model)
        self.res_ln3 = nn.LayerNorm(d_model)

        # # 分类 head → 输出 num_classes 个 logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        # self.abmil = ABMIL(d_model)


    def forward(self, bag):
        """
        bag: {'low': List of tensors [n_patch, d_feat], 'mid': ..., 'high': ...}
        """

        # 只处理实际存在的teacher特征
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
   
        logits = self.classifier(high_out).squeeze()
        return low_out, mid_out, high_out,low_vec,mid_vec,high_vec, logits


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



# -------------- train & eval helpers -------------- #
def train_epoch(model, ldr, opt, criterion, lmbd, dev):
    model.train()
    scaler = GradScaler(device='cuda')
    loader_bar = tqdm(ldr, desc="Training", dynamic_ncols=True, leave=False)
    total_loss = 0
    acc_steps = 0
    accumulate_steps = 16
    opt.zero_grad(set_to_none=True)
    all_logits, all_labels, all_distills = [], [], []

    for step, (feats, labels, _) in enumerate(loader_bar):
        batch_size = labels.shape[0]
        for i in range(batch_size):  # 遍历 batch 内每个 WSI
            bag_i = {level: [tensor[i].to(dev) for tensor in ts_list] for level, ts_list in feats.items()}
            # with autocast(device_type='cuda'):
            with autocast(device_type='cuda'):
                l_s, m_s, h_s,l_v,m_v,h_v, logits = model(bag_i)
                distill_loss = distill(l_s, m_s, h_s, l_v,m_v,h_v)
            all_logits.append(logits)  
            all_labels.append(labels[i].to(dev)) 
            all_distills.append(distill_loss)
            acc_steps += 1
            is_last_sample = (step == len(ldr) - 1) and (i == batch_size - 1)
            if acc_steps == accumulate_steps or is_last_sample:
                # opt.zero_grad()
                logits_batch = torch.stack(all_logits)
                labels_batch = torch.stack(all_labels)
                
    
                with autocast(device_type='cuda'):
                    cls_loss = criterion(logits_batch, labels_batch)
                    distill_mean = sum(all_distills) / len(all_distills)
                    loss = cls_loss + lmbd * distill_mean

                   
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                
                total_loss += loss.item()

                
                acc_steps = 0
                all_logits, all_labels, all_distills = [], [], []



    return total_loss / max(1, len(ldr))



# def train_epoch(model, ldr, opt, cox, lmbd, dev):


@torch.no_grad()
def evaluate(model, ldr, dev, criterion=None, lmbd=None):
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    total_loss = 0
    total_samples = 0
    loader_bar = tqdm(ldr, desc="Evaluating", dynamic_ncols=True, leave=False)
    for feats, labels, _ in loader_bar:
        for i in range(labels.shape[0]):

            bag_i = {level: [ts_list[i].to(dev) for ts_list in feats[level]] for level in ['low', 'mid', 'high']}
 
            l_s, m_s, h_s, l_v, m_v, h_v, logits = model(bag_i)

            # 只计算分类loss，不包含distill loss
            if criterion is not None:
                cls_loss = criterion(logits.unsqueeze(0), labels[i].unsqueeze(0).to(dev))
                total_loss += cls_loss.item()
                total_samples += 1

            pred = torch.argmax(logits, dim=-1).item()
            all_preds.append(pred)
            all_labels.append(labels[i].item())
            all_logits.append(logits.cpu().numpy())
    
    # 计算概率用于ROC AUC
    all_logits = np.array(all_logits)
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
    
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 处理ROC AUC计算
    if len(set(all_labels)) > 1:
        if all_probs.shape[1] == 2:  # 二分类问题
            # 对于二分类，使用正类的概率
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:  # 多分类问题
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        auc = 0.0
    
    # 如果计算了loss，返回loss；否则返回None
    avg_loss = total_loss / max(1, total_samples) if total_samples > 0 else None
    
    return balanced_acc, f1, auc, avg_loss
    

@torch.no_grad()
def inference(model,ldr,dev,save_path):
    model.eval(); results=[]
    for feats,labels,slides in tqdm(ldr,desc="Infer",leave=False):
        B=labels.size(0)
        for i in range(B):
            bag={lvl:[t[i].to(dev) for t in feats[lvl]] for lvl in feats}
            *_,logits = model(bag)
            pred = torch.argmax(logits, dim=-1).item()
            results.append({
                "slide_id": slides[i],
                "gt_label": labels[i].item(),
                "pred_label": pred
            })
    with open(save_path,'w') as f: json.dump(results,f,indent=2)

# ------------------------- main -------------------------- #
def main():
    set_seed(args.seed)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    full_df=pd.read_csv(args.csv)
    
    # 处理label，将数据集名称转换为012
    unique_labels = full_df['label'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    num_classes = len(label_to_idx)
    print(f"Label mapping: {label_to_idx}")
    print(f"Number of classes: {num_classes}")
    
    # first_slide_path = full_df.loc[0, 'slide_id']
    # slide_id = os.path.splitext(os.path.basename(first_slide_path))[0]
    # infer feature dims
    # 正确提取每个teacher的特征维度
    d_teachers = []
    
    # 遍历所有teacher目录，获取维度信息
    for teacher_dir in TEACHER_DIRS:
        # 尝试找到该teacher目录中的任何pt文件来获取维度信息
        pt_files = list(teacher_dir.glob("*.pt"))
        if pt_files:
            # 使用第一个找到的pt文件来获取维度信息
            sample = torch.load(pt_files[0], map_location='cpu', weights_only=True)
            low, _, _ = sample
            d_teachers.append(low.shape[1])
        else:
            print(f"Warning: No pt files found in {teacher_dir}")
    
    # 确保我们找到了足够的teacher维度信息
    if len(d_teachers) == 0:
        raise ValueError(f"No feature files found in any teacher directory")
    
    print(f"Found {len(d_teachers)} teachers with dimensions: {d_teachers}")
    csv_stem = Path(args.csv).stem
    if '_' in csv_stem:
        dataset_name = csv_stem.split('_')[1]
    else:
        dataset_name = csv_stem  # 如果没有下划线，使用整个文件名
    ckpt_root = Path("checkpoints4cls")/dataset_name
    ckpt_root.mkdir(parents=True,exist_ok=True)

    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    print(args.batch_size)
    for fold in folds:
        print(f'===== Fold {fold} =====')
        tr_df,val_df,tst_df = prepare_fold(full_df, fold)
        
        tr_loader = DataLoader(WSIDataset(tr_df, label_to_idx), batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
        val_loader= DataLoader(WSIDataset(val_df, label_to_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        tst_loader= DataLoader(WSIDataset(tst_df, label_to_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        # model=Student(d_teachers).to(dev)
        model = Student(d_teachers, num_classes)
        # model = nn.DataParallel(model)
        model = model.to(dev)
        opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_acc, best_f1, best_auc = 0, 0, 0
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        for ep in epoch_bar:
            # train_epoch_dbg(model, tr_loader, opt, cox, args.lambda_dist, dev)
            tr_loss=train_epoch(model,tr_loader,opt,criterion,args.lambda_dist,dev)
            val_acc, val_f1, val_auc, val_loss = evaluate(model,val_loader,dev,criterion)
            # test_c =evaluate(model,tst_loader,dev)
            if val_auc > best_auc: 
                best_acc = val_acc
                best_f1 = val_f1
                best_auc = val_auc
                pt = ckpt_root/f"fold{fold}_multi_bestval.pt"
                torch.save(model.state_dict(),pt)
            sch.step()
            tqdm.write(f'ep{ep:02d} tr_loss{tr_loss:.4f} val_loss{val_loss:.4f} val_acc{val_acc:.4f} val_f1{val_f1:.4f} val_auc{val_auc:.4f}')
            test_acc, test_f1, test_auc, test_loss =evaluate(model,tst_loader,dev,criterion)
            tqdm.write(f'ep{ep:02d} test_acc {test_acc:.4f} test_f1 {test_f1:.4f} test_auc {test_auc:.4f} test_loss {test_loss:.4f}')
        print(f'Fold {fold} | best val_loss {best_val_loss:.4f} best val_acc {best_acc:.4f} best val_f1{best_f1:.4f} best val_auc{best_auc:.4f}')
        val_scores.append(best_acc)
        ckpt = ckpt_root/f"fold{fold}_multi_bestval.pt"
        model.load_state_dict(torch.load(ckpt,map_location=dev))
        test_acc, test_f1, test_auc, _ =evaluate(model,tst_loader,dev,criterion)
        print(f'Fold {fold} | tst acc {test_acc:.4f} tst f1 {test_f1:.4f} tst auc {test_auc:.4f}')
        # test_scores.append(test_c)
        out_json = ckpt_root/f"fold{fold}_multi_test_preds.json"
        inference(model,tst_loader,dev,out_json)

    # print('\\n===== 5‑fold summary =====')
    # # print('val  mean {:.4f} ± {:.4f}'.format(np.mean(val_scores), np.std(val_scores)))
    # # print('test mean {:.4f} ± {:.4f}'.format(np.mean(test_scores),np.std(test_scores)))
    # val_mean = np.mean(val_scores)
    # val_std = np.std(val_scores, ddof=1)
    # val_ci95 = stats.t.ppf(0.975, df=4) * val_std / np.sqrt(5)


    # test_mean = np.mean(test_scores)
    # test_std = np.std(test_scores, ddof=1)
    # test_ci95 = stats.t.ppf(0.975, df=4) * test_std / np.sqrt(5)

    # print('\n===== 5-fold Summary =====')
    # print(f'val mean {val_mean:.4f} ± {val_std:.4f} (95% CI: [{val_mean-val_ci95:.4f}, {val_mean+val_ci95:.4f}])')
    # print(f'test mean {test_mean:.4f} ± {test_std:.4f} (95% CI: [{test_mean-test_ci95:.4f}, {test_mean+test_ci95:.4f}])')

if __name__=='__main__':
    main()


# python wsi_survival_splitcv.py \
#   --csv       all_cases.csv \
#   --splits_dir splits82/TCGA_BLCA_survival_100 \
#   --root      TCGA_BLCA_multi_features \
#   --teachers  gigapath_features hoptimus0_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 30 --lr 3e-4 --lambda_dist 0.1
# CUDA_VISIBLE_DEVICES=2 python multi_moe_distill_v3.py --csv dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv --splits_dir splits82/TCGA_KIRC_survival_100 --root /nas/share/Extracted_Feature/multi_features4KIRC --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 | tee multilog/KIRC0-4.txt
# CUDA_VISIBLE_DEVICES=0 python multi_moe_distill_v2.py --csv dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv --splits_dir splits82/TCGA_BLCA_survival_100 --root TCGA_BLCA_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 > runBLCA.txt 2>&1
# CUDA_VISIBLE_DEVICES=0 python multi_moe_distill4cls.py --csv dataset_csv/RCC.csv --splits_dir splits712/RCC_100 --root  /data2/leiwenhui/Data/Extracted_Feature/TCGA_KICH_multi_features /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRP_multi_features /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 | tee multilog/RCC0-4.txt
# CUDA_VISIBLE_DEVICES=2 python multi_moe_distill4cls.py --csv dataset_csv/BRCA_subtyping.csv --splits_dir splits712/TCGA_BRCA_subtyping_100 --root  /data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 | tee multilog/BRCA4cls0-4.txt
# CUDA_VISIBLE_DEVICES=0 python multi_moe_distill4cls.py --csv dataset_csv/LUAD_LUSC.csv --splits_dir splits712/LUAD_LUSC_100 --root  /data2/leiwenhui/Data/Extracted_Feature/TCGA_LUAD_multi_features /data2/leiwenhui/Data/Extracted_Feature/TCGA_LUSC_multi_features --teacher gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features --epochs 20 --lr 2e-4 | tee multilog/LUAD_LUSC4cls0-4.txt
