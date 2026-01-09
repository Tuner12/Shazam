#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Teacher WSI Dataset类 - 用于单个教师模型的生存分析，只使用high层特征
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path


class SingleWSIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, teacher_dir: Path, n_bins: int = 4, create_Y: bool = True, 
                 splits_dir: Path = None, fold_idx: int = None, split_type: str = 'train', 
                 expand_multi_slides: bool = True):
        """
        单教师WSI数据集类，用于加载单个教师特征进行生存分析，只使用high层特征
        
        Args:
            df: DataFrame with columns ['slide_id', 'survival_months', 'censorship']
                - slide_id: 切片文件路径
                - survival_months: 生存时间（月）
                - censorship: 删失标记 (1=删失, 0=死亡)
            teacher_dir: 单个教师特征目录，包含.pt文件
            n_bins: 生存时间离散化的bins数量，默认4
            create_Y: 是否自动创建Y标签，默认True
            splits_dir: fold分割文件目录，如果提供则使用bool类型fold文件进行分割
            fold_idx: fold索引 (0-4)，如果提供则只使用指定的fold
            split_type: 分割类型 ('train', 'val', 'test')，默认'train'
            expand_multi_slides: 是否展开多个slide为独立样本，默认True
        """
        self.df = df.reset_index(drop=True)
        self.teacher_dir = teacher_dir
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
        
        original_size = len(self.df)
        self.df['case_id'] = self.df['slide_id'].apply(extract_case_id_from_slide)
        self.df = self.df[self.df['case_id'].isin(split_cases)].reset_index(drop=True)
        
        print(f"✓  Fold {self.fold_idx} {self.split_type}集分割完成:")
        print(f"  原始样本数: {original_size}")
        print(f"  过滤后样本数: {len(self.df)}")
        print(f"  使用的case_id数量: {len(split_cases)}")
        
        # 删除临时添加的case_id列
        if 'case_id' in self.df.columns:
            self.df = self.df.drop('case_id', axis=1)
    
    @classmethod
    def create_fold_datasets(cls, df: pd.DataFrame, teacher_dir: Path, splits_dir: Path, 
                           fold_idx: int, n_bins: int = 4, create_Y: bool = True, 
                           expand_multi_slides: bool = True):
        """
        便利方法：为指定fold创建训练、验证、测试数据集
        
        Args:
            df: 原始DataFrame
            teacher_dir: 单个教师特征目录
            splits_dir: fold分割文件目录
            fold_idx: fold索引 (0-4)
            n_bins: 生存时间离散化的bins数量
            create_Y: 是否自动创建Y标签
            expand_multi_slides: 是否展开多个slide为独立样本
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = cls(df, teacher_dir, n_bins, create_Y, splits_dir, fold_idx, 'train', expand_multi_slides)
        val_dataset = cls(df, teacher_dir, n_bins, create_Y, splits_dir, fold_idx, 'val', expand_multi_slides)
        test_dataset = cls(df, teacher_dir, n_bins, create_Y, splits_dir, fold_idx, 'test', expand_multi_slides)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_Y_labels(self):
        """
        创建生存时间离散化标签Y
        使用未删失样本的生存时间来确定bin边界
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

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_path = row['slide_id']
        
        # 现在每个样本只包含一个slide
        wsi = os.path.splitext(os.path.basename(slide_path))[0]
        
        # 检查教师特征文件是否存在，如果缺失则跳过
        pt_file = self.teacher_dir / f"{wsi}.pt"
        if not pt_file.exists():
            # print(f"跳过: WSI {wsi} 缺失特征文件 {pt_file}")
            # 返回None表示跳过此样本
            return None
        
        # 读取教师特征，只使用high层特征
        _, _, high_t = torch.load(pt_file, map_location='cpu', weights_only=True)
        
        # 标签处理
        label = {
            'survival_times': torch.tensor(row['survival_months'], dtype=torch.float32),
            'censorship': torch.tensor(row['censorship'], dtype=torch.float32),  # 0=死亡, 1=删失
            'Y': torch.tensor(int(row['Y']), dtype=torch.long)  # 1-based bin 索引
        }
        
        # 返回单个slide的high层特征和标签
        return high_t, label, wsi


def single_collate(batch):
    """
    单教师批处理函数，将多个样本组合成批次，过滤掉None值
    
    Args:
        batch: List of (high_feat, label, wsi_name) tuples or None
               - high_feat: high层特征tensor
               - label: Dict with 'survival_times', 'censorship', 'Y' keys
               - wsi_name: Single slide name string
    
    Returns:
        high_feats: List of high层特征tensors
        times: Tensor of survival times
        censorships: Tensor of censorship flags
        Y_bins: Tensor of bin indices
        slides: List of slide names
    """
    # 过滤掉None值
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # 如果所有样本都被跳过，返回空批次
        return None, None, None, None, None
    
    high_feats, times, censorships, Y_bins, slides = [], [], [], [], []

    for high_feat, label, wsi_name in valid_batch:
        high_feats.append(high_feat)
        times.append(label['survival_times'])
        censorships.append(label['censorship'])
        Y_bins.append(label['Y'])
        slides.append(wsi_name)

    return high_feats, torch.stack(times), torch.stack(censorships), torch.stack(Y_bins), slides


def extract_case_id(slide_path):
    """从slide路径中提取case_id"""
    filename = os.path.basename(slide_path)
    return '-'.join(filename.split('-')[:3])


def is_valid_slide(slide_path, teacher_dir):
    """检查slide对应的特征文件是否存在"""
    if isinstance(slide_path, str) and ';' in slide_path:
        # 处理多个slide的情况
        slide_paths = slide_path.split(';')
        for path in slide_paths:
            wsi = os.path.splitext(os.path.basename(path.strip()))[0]
            pt_file = teacher_dir / f"{wsi}.pt"
            if not pt_file.exists():
                return False
        return True
    else:
        # 单个slide
        wsi = os.path.splitext(os.path.basename(slide_path))[0]
        pt_file = teacher_dir / f"{wsi}.pt"
        return pt_file.exists()


if __name__ == "__main__":
    """
    测试SingleWSIDataset类的基本功能，支持不同数据集
    """
    import pandas as pd
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    # 可以修改这里的数据集名称来测试不同数据集
    dataset_name = "LUSC"  # 可选: KIRC, BRCA, BLCA, STAD, CESC, GBM, HNSC, KICH, KIRP, LGG, LUAD, LUSC等
    
    print(f"=== Single WSI Dataset 功能测试 - {dataset_name}数据集 ===")
    
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
    
    # 使用第一个教师模型
    teacher_dir = teacher_dirs[0]
    print(f"\n使用教师模型: {teacher_dir.parent.name}")
    print(f"教师特征目录: {teacher_dir}")
    
    # 统计总slide数并检查特征文件存在性
    total_slides = 0
    missing_features = []
    existing_features = 0
    
    for slide_id in df['slide_id']:
        if isinstance(slide_id, str) and ';' in slide_id:
            slide_paths = slide_id.split(';')
        else:
            slide_paths = [slide_id]
        
        for slide_path in slide_paths:
            slide_path = slide_path.strip()
            wsi = os.path.splitext(os.path.basename(slide_path))[0]
            total_slides += 1
            
            # 检查教师特征文件是否存在
            pt_file = teacher_dir / f"{wsi}.pt"
            if pt_file.exists():
                existing_features += 1
            else:
                missing_features.append(wsi)
    
    print(f"\n特征文件检查结果 (教师模型: {teacher_dir.parent.name}):")
    print(f"  总slide数: {total_slides}")
    print(f"  已提取特征的slide数: {existing_features}")
    print(f"  缺失特征的slide数: {len(missing_features)}")
    
    if missing_features:
        print(f"  缺失特征的slide示例 (前10个): {missing_features[:10]}")
    else:
        print(f"  ✓ 所有slide都已提取特征！")
    
    # 测试fold 0的数据集创建
    fold_idx = 0
    print(f"\n=== 测试Fold {fold_idx} ===")
    
    try:
        # 创建训练、验证、测试数据集（展开多个slide为独立样本）
        train_dataset, val_dataset, test_dataset = SingleWSIDataset.create_fold_datasets(
            df, teacher_dir, splits_dir, fold_idx, n_bins=4, create_Y=True, expand_multi_slides=True
        )
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=single_collate)
        
        print(f"\nDataLoader信息:")
        print(f"  训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        # 获取第一个batch的信息
        for batch_idx, (high_feats, times, censorships, Y_bins, slides) in enumerate(train_loader):
            print(f"\n第{batch_idx+1}个batch:")
            print(f"  high特征数量: {len(high_feats)}")
            print(f"  时间形状: {times.shape}")
            print(f"  删失形状: {censorships.shape}")
            print(f"  Y_bins形状: {Y_bins.shape}")
            print(f"  slides: {slides}")
            
            # 显示特征详细信息
            print(f"\n特征详细信息:")
            for i, high_feat in enumerate(high_feats):
                print(f"  样本{i+1} high特征形状: {high_feat.shape}")
            break  # 只显示第一个batch
            
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ 测试完成！")