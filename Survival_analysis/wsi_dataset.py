#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI Dataset类 - 用于多教师特征融合的生存分析
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path


class WSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, teacher_dirs: List[Path], n_bins: int = 4, create_Y: bool = True, 
                 splits_dir: Path = None, fold_idx: int = None, split_type: str = 'train', 
                 expand_multi_slides: bool = True):
        """
        WSI数据集类，用于加载多教师特征进行生存分析
        
        Args:
            df: DataFrame with columns ['slide_id', 'survival_months', 'censorship']
                - slide_id: 切片文件路径
                - survival_months: 生存时间（月）
                - censorship: 删失标记 (1=删失, 0=死亡)
            teacher_dirs: 教师特征目录列表，每个目录包含.pt文件
            n_bins: 生存时间离散化的bins数量，默认4
            create_Y: 是否自动创建Y标签，默认True
            splits_dir: fold分割文件目录，如果提供则使用bool类型fold文件进行分割
            fold_idx: fold索引 (0-4)，如果提供则只使用指定的fold
            split_type: 分割类型 ('train', 'val', 'test')，默认'train'
        """
        self.df = df.reset_index(drop=True)
        self.teacher_dirs = teacher_dirs
        self.n_bins = n_bins
        self.splits_dir = splits_dir
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.expand_multi_slides = expand_multi_slides
        
        # 如果提供了splits_dir，则使用bool类型fold文件进行分割
        if splits_dir is not None:
            self._apply_fold_split()
        
        # 如果DataFrame中没有Y列且需要创建，则自动构造Y标签
        if create_Y and 'Y' not in self.df.columns:
            self._create_Y_labels()
        
        # 展开多个slide为独立样本
        if expand_multi_slides:
            self._expand_multi_slides()
    
        # 过滤缺失文件的样本
        self._filter_missing_files()
    
    def _apply_fold_split(self):
        """
        使用bool类型的fold文件进行数据分割
        """
        if self.fold_idx is None:
            raise ValueError("当提供splits_dir时，必须指定fold_idx")
        
        fold_file = self.splits_dir / f"splits_{self.fold_idx}_bool.csv"
        if not fold_file.exists():
            raise FileNotFoundError(f"Fold文件不存在: {fold_file}")
        
        # 读取fold文件
        fold_df = pd.read_csv(fold_file)
        
        # fold文件的第一列是case_id（没有列名）
        case_id_col = fold_df.columns[0]  # 第一列
        
        # 根据split_type选择对应的列
        if self.split_type not in ['train', 'val', 'test']:
            raise ValueError("split_type必须是 'train', 'val', 或 'test'")
        
        # 获取该fold中指定split_type的case_id列表
        split_cases = fold_df[fold_df[self.split_type] == True][case_id_col].tolist()
        
        # 从原始DataFrame中提取对应的case_id
        def extract_case_id_from_slide(slide_path):
            filename = os.path.basename(slide_path)
            return '-'.join(filename.split('-')[:3])
        
        # 过滤DataFrame，只保留指定split_type的样本
        original_size = len(self.df)
        self.df['case_id'] = self.df['slide_id'].apply(extract_case_id_from_slide)
        self.df = self.df[self.df['case_id'].isin(split_cases)].reset_index(drop=True)
        
        print(f"Fold {self.fold_idx} - {self.split_type}集:")
        print(f"  原始样本数: {original_size}")
        print(f"  过滤后样本数: {len(self.df)}")
        print(f"  使用的case_id数量: {len(split_cases)}")
        
        # 删除临时添加的case_id列
        if 'case_id' in self.df.columns:
            self.df = self.df.drop('case_id', axis=1)
    
    @classmethod
    def create_fold_datasets(cls, df: pd.DataFrame, teacher_dirs: List[Path], splits_dir: Path, 
                           fold_idx: int, n_bins: int = 4, create_Y: bool = True, 
                           expand_multi_slides: bool = True):
        """
        便利方法：为指定fold创建训练、验证、测试数据集
        
        Args:
            df: 原始DataFrame
            teacher_dirs: 教师特征目录列表
            splits_dir: fold分割文件目录
            fold_idx: fold索引 (0-4)
            n_bins: 生存时间离散化的bins数量
            create_Y: 是否自动创建Y标签
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = cls(df, teacher_dirs, n_bins, create_Y, splits_dir, fold_idx, 'train', expand_multi_slides)
        val_dataset = cls(df, teacher_dirs, n_bins, create_Y, splits_dir, fold_idx, 'val', expand_multi_slides)
        test_dataset = cls(df, teacher_dirs, n_bins, create_Y, splits_dir, fold_idx, 'test', expand_multi_slides)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_Y_labels(self):
        """
        构造Y标签：将生存时间离散化为n_bins个区间
        参考multi_moe_distill_nll_v2.py中的方法
        """
        survival_times = self.df['survival_months'].values
        censorship = self.df['censorship'].values
        
        # 只使用未删失样本的生存时间来确定 bin 边界
        uncensored_times = survival_times[censorship == 0]
        
        if len(uncensored_times) < self.n_bins:
            print(f"   警告: 只有 {len(uncensored_times)} 个未删失事件，使用所有样本进行分箱")
            times_for_binning = survival_times
        else:
            times_for_binning = uncensored_times
        
        # 使用分位数创建 bin 边界
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = np.quantile(times_for_binning, quantiles)
        bin_edges[0] = 0  # 第一个 bin 从 0 开始
        bin_edges[-1] = np.inf  # 最后一个 bin 到无穷
        
        # 使用 digitize 分配 bin 索引（直接生成 1-based）
        Y = np.digitize(survival_times, bin_edges[1:], right=False) + 1
        Y = np.clip(Y, 1, self.n_bins)
        
        # 将Y添加到DataFrame中
        self.df['Y'] = Y
        
        print(f"✓  生存时间离散化完成: {self.n_bins} bins")
        print(f"   Bin 边界: {bin_edges}")
        print(f"   Y 范围: [1, {self.n_bins}] (1-based)")
        
        # 打印标签分布
        print(f"   Bin 分布 (Y):")
        for i in range(1, self.n_bins + 1):
            count = (Y == i).sum()
            print(f"     Bin {i}: {count} 样本")

    def _expand_multi_slides(self):
        """
        将包含多个slide的case展开为独立的样本
        每个slide将成为一个独立的训练样本
        """
        expanded_rows = []
        
        for idx, row in self.df.iterrows():
            slide_paths = row['slide_id']
            
            # 处理多个slide_id的情况（用分号分隔）
            if isinstance(slide_paths, str) and ';' in slide_paths:
                slide_paths = slide_paths.split(';')
            else:
                slide_paths = [slide_paths]
            
            # 为每个slide创建一行
            for slide_path in slide_paths:
                slide_path = slide_path.strip()
                new_row = row.copy()
                new_row['slide_id'] = slide_path
                expanded_rows.append(new_row)
        
        # 更新DataFrame
        original_size = len(self.df)
        self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        print(f"✓  多slide展开完成:")
        print(f"  原始样本数: {original_size}")
        print(f"  展开后样本数: {len(self.df)}")
        print(f"  平均每个case的slide数: {len(self.df) / original_size:.2f}")

    def _filter_missing_files(self):
        """
        过滤掉在所有教师目录中都缺失特征文件的样本
        记录被跳过的样本信息
        """
        original_size = len(self.df)
        valid_indices = []
        skipped_samples = []
        
        for idx, row in self.df.iterrows():
            slide_path = row['slide_id']
            wsi = os.path.splitext(os.path.basename(slide_path))[0]
            
            # 检查所有教师特征文件是否存在
            all_exist = True
            missing_in = []
            for d in self.teacher_dirs:
                pt_file = d / f"{wsi}.pt"
                if not pt_file.exists():
                    all_exist = False
                    # 获取教师名称：从路径中提取（merged_pt_files的父目录名）
                    teacher_name = d.parent.name if d.name == 'merged_pt_files' else d.name
                    missing_in.append(teacher_name)
            
            if all_exist:
                valid_indices.append(idx)
            else:
                skipped_samples.append({
                    'wsi': wsi,
                    'slide_path': slide_path,
                    'missing_in': missing_in
                })
        
        # 过滤DataFrame
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        if skipped_samples:
            print(f"⚠  过滤缺失文件:")
            print(f"  原始样本数: {original_size}")
            print(f"  过滤后样本数: {len(self.df)}")
            print(f"  跳过的样本数: {len(skipped_samples)}")
            if len(skipped_samples) <= 10:
                print(f"  跳过的样本列表:")
                for i, sample in enumerate(skipped_samples):
                    print(f"    {i+1}. {sample['wsi']}")
                    print(f"       缺失于: {', '.join(sample['missing_in'])}")
            else:
                print(f"  跳过的样本示例 (前5个):")
                for i, sample in enumerate(skipped_samples[:5]):
                    print(f"    {i+1}. {sample['wsi']}")
                    print(f"       缺失于: {', '.join(sample['missing_in'])}")
        else:
            print(f"✓  所有样本的特征文件都存在")

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row['slide_id']
        
        # 现在每个样本只包含一个slide
        wsi = os.path.splitext(os.path.basename(slide_path))[0]
        
        # 注意：文件存在性检查已在初始化时完成，这里不再检查
        # 如果文件不存在，说明初始化时的过滤逻辑有问题

        # 读取教师特征
        all_bags = {'low': [], 'mid': [], 'high': []}
        for d in self.teacher_dirs:
            low_t, mid_t, high_t = torch.load(d / f'{wsi}.pt', map_location='cpu', weights_only=True)
            all_bags['low'].append(low_t)
            all_bags['mid'].append(mid_t)
            all_bags['high'].append(high_t)

        # 标签处理
        label = {
            'survival_times': torch.tensor(row['survival_months'], dtype=torch.float32),
            'censorship': torch.tensor(row['censorship'], dtype=torch.float32),  # 0=死亡, 1=删失
            'Y': torch.tensor(int(row['Y']), dtype=torch.long)  # 1-based bin 索引
        }
        
        # 返回单个slide的特征和标签
        return all_bags, label, wsi


def collate(batch):
    """
    批处理函数，将多个样本组合成批次
    
    Args:
        batch: List of (bag, label, wsi_names) tuples
               - bag: Dict with 'low', 'mid', 'high' keys, each containing List of tensors
               - label: Dict with survival info
               - wsi_names: List of WSI names (可能包含多个slide)
        
    Returns:
        feats: Dict with 'low', 'mid', 'high' keys, each containing List of tensors
        times: Tensor of survival times
        censorships: Tensor of censorship indicators
        Y_bins: Tensor of bin indices
        slides: List of slide IDs (每个元素可能是多个slide的列表)
    """
    feats = {'low': [], 'mid': [], 'high': []}
    times, censorships, Y_bins, slides = [], [], [], []

    for bag, label, wsi_name in batch:  # wsi_name现在是单个字符串
        for k in feats:
            for i, t in enumerate(bag[k]):
                if len(feats[k]) <= i:
                    feats[k].append([])
                feats[k][i].append(t)
        times.append(label['survival_times'])
        censorships.append(label['censorship'])
        Y_bins.append(label['Y'])
        slides.append(wsi_name)  # 现在wsi_name是单个字符串

    return feats, torch.stack(times), torch.stack(censorships), torch.stack(Y_bins), slides


def extract_case_id(slide_path):
    """
    从切片路径中提取case ID
    
    Args:
        slide_path: 切片文件路径
        
    Returns:
        case_id: 提取的case ID (例如: TCGA-06-1086)
    """
    filename = os.path.basename(slide_path)
    return '-'.join(filename.split('-')[:3])


def is_valid_slide(slide_path, teacher_dirs):
    """
    检查切片是否在所有教师目录中都有对应的特征文件
    
    Args:
        slide_path: 切片文件路径
        teacher_dirs: 教师特征目录列表
        
    Returns:
        bool: 如果所有教师都有该切片的特征文件则返回True
    """
    wsi = os.path.splitext(os.path.basename(slide_path))[0]
    return all((d / f'{wsi}.pt').exists() for d in teacher_dirs)


if __name__ == "__main__":
    """
    测试WSIDataset类的基本功能，支持不同数据集
    """
    import pandas as pd
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    # 可以修改这里的数据集名称来测试不同数据集
    dataset_name = "BRCA"  # 可选: KIRC, BRCA, BLCA, STAD, CESC, GBM, HNSC, KICH, KIRP, LGG, LUAD, LUSC等
    # KIRC STAD LUAD LUSC
    print(f"=== WSI Dataset 功能测试 - {dataset_name}数据集 ===")
    
    # 数据集路径（支持不同数据集）
    dataset_csv = Path(f"/data2/tanyusheng/Code/Survival/dataset_csv/survival_by_case/TCGA_{dataset_name}_Splits.csv")
    features_root = Path(f"/data2/tanyusheng/Data/Extracted_Feature/TCGA_{dataset_name}_multi_features")
    splits_dir = Path(f"/data2/tanyusheng/Code/Survival/splits82/TCGA_{dataset_name}_survival_100")
    
    # 检查路径存在性
    print(f"CSV文件存在: {dataset_csv.exists()}")
    print(f"特征目录存在: {features_root.exists()}")
    print(f"Fold目录存在: {splits_dir.exists()}")
    
    if not all([dataset_csv.exists(), features_root.exists(), splits_dir.exists()]):
        print("路径不存在，跳过测试")
        exit()
    
    # 加载数据
    df = pd.read_csv(dataset_csv)
    print(f"\n数据信息:")
    print(f"  总case数: {len(df)}")
    
    # 设置教师特征目录
    teacher_dirs = []
    for feat_dir in features_root.iterdir():
        if feat_dir.is_dir():
            merged_pt_dirs = list(feat_dir.glob("**/merged_pt_files"))
            if merged_pt_dirs:
                teacher_dirs.append(merged_pt_dirs[0])
    
    print(f"\n找到的教师模型: {len(teacher_dirs)} 个")
    for i, teacher_dir in enumerate(teacher_dirs):
        print(f"  {i+1}. {teacher_dir.parent.name}: {teacher_dir}")
    
    if not teacher_dirs:
        print("未找到教师特征目录，跳过测试")
        exit()
    
    # 统计总slide数并检查特征文件存在性
    total_slides = 0
    missing_features = []
    existing_features = 0
    partial_features = []  # 部分教师模型有特征的slide
    
    for slide_id in df['slide_id']:
        if isinstance(slide_id, str) and ';' in slide_id:
            slide_paths = slide_id.split(';')
        else:
            slide_paths = [slide_id]
        
        for slide_path in slide_paths:
            slide_path = slide_path.strip()
            wsi = os.path.splitext(os.path.basename(slide_path))[0]
            total_slides += 1
            
            # 检查每个教师特征文件是否存在
            existing_teachers = []
            missing_teachers = []
            
            for i, teacher_dir in enumerate(teacher_dirs):
                pt_file = teacher_dir / f"{wsi}.pt"
                if pt_file.exists():
                    existing_teachers.append(i)
                else:
                    missing_teachers.append(i)
            
            if len(existing_teachers) == len(teacher_dirs):
                # 所有教师模型都有特征
                existing_features += 1
            elif len(existing_teachers) > 0:
                # 部分教师模型有特征
                partial_features.append({
                    'wsi': wsi,
                    'existing_teachers': existing_teachers,
                    'missing_teachers': missing_teachers
                })
            else:
                # 所有教师模型都缺失特征
                missing_features.append(wsi)
    
    print(f"\n特征文件检查结果:")
    print(f"  总slide数: {total_slides}")
    print(f"  所有教师模型都有特征的slide数: {existing_features}")
    print(f"  部分教师模型有特征的slide数: {len(partial_features)}")
    print(f"  所有教师模型都缺失特征的slide数: {len(missing_features)}")
    
    # if partial_features:
    #     print(f"\n部分教师模型有特征的slide示例 (前5个):")
    #     for i, item in enumerate(partial_features):
    #         existing_names = [teacher_dirs[j].parent.name for j in item['existing_teachers']]
    #         missing_names = [teacher_dirs[j].parent.name for j in item['missing_teachers']]
    #         print(f"  {i+1}. {item['wsi']}")
    #         print(f"     有特征: {existing_names}")
    #         print(f"     缺失特征: {missing_names}")
    
    if missing_features:
        print(f"\n所有教师模型都缺失特征的slide示例 (前10个): {missing_features}")
    
    if existing_features == total_slides:
        print(f"\n✓ 所有slide在所有教师模型中都已提取特征！")
    elif len(missing_features) == 0:
        print(f"\n✓ 所有slide至少在一个教师模型中有特征！")
    
    # 测试fold 0的数据集创建
    fold_idx = 0
    print(f"\n=== 测试Fold {fold_idx} ===")
    
    try:
        # 创建训练、验证、测试数据集（展开多个slide为独立样本）
        train_dataset, val_dataset, test_dataset = WSIDataset.create_fold_datasets(
            df, teacher_dirs, splits_dir, fold_idx, n_bins=4, create_Y=True, expand_multi_slides=True
        )
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate)
        
        print(f"\nDataLoader信息:")
        print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        # 获取第一个batch的信息
        for batch_idx, (feats, times, censorships, Y_bins, slides) in enumerate(train_loader):
            print(f"\n第{batch_idx+1}个batch:")
            print(f"  特征形状: low={len(feats['low'])}, mid={len(feats['mid'])}, high={len(feats['high'])}")
            print(f"  时间形状: {times.shape}")
            print(f"  删失形状: {censorships.shape}")
            print(f"  Y_bins形状: {Y_bins.shape}")
            print(f"  slides: {slides}")
            
            # 显示特征详细信息
            print(f"\n特征详细信息:")
            for level in ['low', 'mid', 'high']:
                print(f"  {level}层特征:")
                for i, teacher_feat in enumerate(feats[level]):
                    if isinstance(teacher_feat, list):
                        print(f"    教师{i+1}: 列表，包含{len(teacher_feat)}个tensor")
                        if teacher_feat:
                            print(f"      第一个tensor形状: {teacher_feat[0].shape}")
                    else:
                        print(f"    教师{i+1}: {teacher_feat.shape}")
            break  # 只显示第一个batch
            
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ 测试完成！")
