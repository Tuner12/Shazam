#!/usr/bin/env python3
"""
WSI-level classification · 5-fold CV using split<i>.csv  (ID-enumeration format)
------------------------------------------------------------------------------

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
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast

# -------------------- CLI ------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument('--csv',       required=True, help='full meta CSV: wsi_id,label')
ap.add_argument('--splits_dir',required=True, help='folder containing split0.csv … split4.csv')
ap.add_argument('--root',      nargs='+', required=True, help='multiple roots with <teacher>_features/pt_files')
ap.add_argument('--teachers',  nargs='+',    required=True, help='teacher folder names')
ap.add_argument('--epochs',    type=int,   default=20)
ap.add_argument('--lr',        type=float, default=2e-4)
ap.add_argument('--seed',      type=int,   default=42)
ap.add_argument('--fold_idx', type=int, default=None, help='Specify a single fold to run (0-4); if not set, all folds will be run')
ap.add_argument('--batch_size', type=int, default=4)
args = ap.parse_args()

# 构建所有teacher目录路径
TEACHER_DIRS = []
for root_path in args.root:
    for teacher in args.teachers:
        TEACHER_DIRS.append(Path(root_path) / f'{teacher}/merged_pt_files')
N_TEACHERS   = len(TEACHER_DIRS)

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
class ABMIL(nn.Module):
    def __init__(self, C, hidden=128, embed_dim=128, dropout=0.25, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(C, embed_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, hidden)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, 1, bias=False)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        a = self.dropout2(self.tanh(self.fc2(x)))
        a = self.fc3(a)
        w = torch.softmax(a, 0)
        z = (w * x).sum(0)  # [embed_dim]
        
        logits = self.classifier(z)
        return logits

# -------------- train & eval helpers -------------- #
def train_epoch(model, ldr, opt, criterion, dev):
    model.train()
    scaler = GradScaler(device='cuda')
    loader_bar = tqdm(ldr, desc="Training", dynamic_ncols=True, leave=False)
    total_loss = 0

    for step, (feats, labels, _) in enumerate(loader_bar):
        batch_size = labels.shape[0]
        for i in range(batch_size):  # 遍历 batch 内每个 WSI
            bag_i = {level: [tensor[i].to(dev) for tensor in ts_list] for level, ts_list in feats.items()}
            
            # 简单平均融合所有teacher的特征
            low_feat = torch.stack([feat.to(dev) for feat in bag_i['low']]).mean(0)   # [n_patch, d_feat]
            mid_feat = torch.stack([feat.to(dev) for feat in bag_i['mid']]).mean(0)   # [n_patch, d_feat]
            high_feat = torch.stack([feat.to(dev) for feat in bag_i['high']]).mean(0) # [n_patch, d_feat]
            
            # 使用high-level特征进行分类
            with autocast(device_type='cuda'):
                logits = model(high_feat)
                loss = criterion(logits.unsqueeze(0), labels[i].to(dev).unsqueeze(0))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

    return total_loss / max(1, len(ldr))

@torch.no_grad()
def evaluate(model, ldr, dev, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []
    total_loss = 0
    total_samples = 0
    loader_bar = tqdm(ldr, desc="Evaluating", dynamic_ncols=True, leave=False)
    for feats, labels, _ in loader_bar:
        for i in range(labels.shape[0]):
            bag_i = {level: [ts_list[i].to(dev) for ts_list in feats[level]] for level in ['low', 'mid', 'high']}
            high_feat = torch.stack([feat.to(dev) for feat in bag_i['high']]).mean(0)
            logits = model(high_feat)
            pred = torch.argmax(logits, dim=-1).item()
            all_preds.append(pred)
            all_labels.append(labels[i].item())
            all_logits.append(logits.cpu().numpy())
            if criterion is not None:
                loss = criterion(logits.unsqueeze(0), labels[i].to(dev).unsqueeze(0))
                total_loss += loss.item()
                total_samples += 1
    all_logits = np.array(all_logits)
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    unique_labels = set(all_labels)
    print(f"Debug - Unique labels in evaluation: {unique_labels}")
    print(f"Debug - Label counts: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    if len(unique_labels) > 1:
        if all_probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        auc = 0.0
        print(f"⚠️  WARNING: Only one class found in evaluation, AUC set to 0.0")
    avg_loss = total_loss / max(1, total_samples) if total_samples > 0 else None
    return balanced_acc, f1, auc, avg_loss

@torch.no_grad()
def inference(model,ldr,dev,save_path):
    model.eval(); results=[]
    for feats,labels,slides in tqdm(ldr,desc="Infer",leave=False):
        B=labels.size(0)
        for i in range(B):
            bag={lvl:[t[i].to(dev) for t in feats[lvl]] for lvl in feats}
            
            # 简单平均融合所有teacher的特征
            high_feat = torch.stack([feat.to(dev) for feat in bag['high']]).mean(0)
            
            logits = model(high_feat)
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
    model_name = args.teachers
    # 获取teacher特征维度
    d_teachers = []
    for teacher_dir in TEACHER_DIRS:
        pt_files = list(teacher_dir.glob("*.pt"))
        if pt_files:
            sample = torch.load(pt_files[0], map_location='cpu', weights_only=True)
            low, _, high = sample
            d_teachers.append(high.shape[1])  # 使用high-level特征的维度
        else:
            print(f"Warning: No pt files found in {teacher_dir}")
    
    if len(d_teachers) == 0:
        raise ValueError(f"No feature files found in any teacher directory")
    
    print(f"Found {len(d_teachers)} teachers with dimensions: {d_teachers}")
    
    # 创建checkpoint目录
    csv_stem = Path(args.csv).stem
    if '_' in csv_stem:
        dataset_name = csv_stem.split('_')[1]
    else:
        dataset_name = csv_stem
    ckpt_root = Path("checkpoints4cls")/dataset_name
    ckpt_root.mkdir(parents=True,exist_ok=True)

    val_scores, test_scores = [], []
    folds = [args.fold_idx] if args.fold_idx is not None else range(5)
    print(args.batch_size)
    
    for fold in folds:
        print(f'===== Fold {fold} =====')
        tr_df,val_df,tst_df = prepare_fold(full_df, fold)

        # 打印数据分布信息
        print(f"Train samples: {len(tr_df)}, Train labels: {tr_df['label'].value_counts().to_dict()}")
        print(f"Val samples: {len(val_df)}, Val labels: {val_df['label'].value_counts().to_dict()}")
        print(f"Test samples: {len(tst_df)}, Test labels: {tst_df['label'].value_counts().to_dict()}")
        
        # 检查数据是否平衡
        if len(val_df['label'].unique()) == 1:
            print("⚠️  WARNING: Validation set contains only one class!")
        if len(tst_df['label'].unique()) == 1:
            print("⚠️  WARNING: Test set contains only one class!")

        tr_loader = DataLoader(WSIDataset(tr_df, label_to_idx), batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
        val_loader= DataLoader(WSIDataset(val_df, label_to_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        tst_loader= DataLoader(WSIDataset(tst_df, label_to_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        # 使用第一个teacher的维度作为输入维度
        model = ABMIL(d_teachers[0], num_classes=num_classes)
        model = model.to(dev)
        opt=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc,best_f1,best_auc=0,0,0
        best_val_loss = float('inf')
        epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold}", dynamic_ncols=True, leave=False)
        
        for ep in epoch_bar:
            tr_loss=train_epoch(model,tr_loader,opt,criterion,dev)
            val_acc, val_f1, val_auc, val_loss = evaluate(model,val_loader,dev,criterion)
            if val_auc > best_auc:
                best_val_loss = val_loss
                best_acc = val_acc
                best_f1 = val_f1
                best_auc = val_auc
                pt = ckpt_root/f"{model_name}_fold{fold}_single_bestval.pt"
                torch.save(model.state_dict(),pt)
            sch.step()
            tqdm.write(f'ep{ep:02d} loss{tr_loss:.4f}  val_loss{val_loss:.4f} val_acc{val_acc:.4f} val_f1{val_f1:.4f} val_auc{val_auc:.4f}')
        print(f'Fold {fold} | best val loss {best_val_loss:.4f} best val acc {best_acc:.4f} best val_f1{best_f1:.4f} best val_auc{best_auc:.4f}')
        val_scores.append(best_acc)
        
        # 加载最佳模型进行测试
        ckpt = ckpt_root/f"{model_name}_fold{fold}_single_bestval.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt,map_location=dev))
        test_acc, test_f1, test_auc, test_loss =evaluate(model,tst_loader,dev) 
        print(f'Fold {fold} | tst acc {test_acc:.4f} tst f1 {test_f1:.4f} tst auc {test_auc:.4f}')
        test_scores.append(test_acc)
        
        # 保存预测结果
        out_json = ckpt_root/f"{model_name}_fold{fold}_single_test_preds.json"
        inference(model,tst_loader,dev,out_json)

    # 输出5-fold总结
    if len(val_scores) > 1:
        val_mean = np.mean(val_scores)
        val_std = np.std(val_scores, ddof=1)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores, ddof=1)

        print('\n===== 5-fold Summary =====')
        print(f'val mean {val_mean:.4f} ± {val_std:.4f}')
        print(f'test mean {test_mean:.4f} ± {test_std:.4f}')

if __name__=='__main__':
    main()

# 使用示例：
# python single_model4cls.py \
#   --csv dataset_csv/RCC.csv \
#   --splits_dir splits712/RCC_100 \
#   --root /data2/leiwenhui/Data/Extracted_Feature/TCGA_KICH_multi_features \
#        /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRP_multi_features \
#        /data2/leiwenhui/Data/Extracted_Feature/TCGA_KIRC_multi_features \
#   --teachers gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features \
#   --epochs 20 --lr 2e-4
