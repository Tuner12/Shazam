#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Hugging Face PathVQA数据集的实际格式
"""

import os
from datasets import load_dataset

# # 设置代理
# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'

# def check_dataset_format():
#     """检查数据集格式"""
#     print("=== 检查Hugging Face PathVQA数据集格式 ===\n")
    
#     # 加载数据集
#     print("正在加载数据集...")
#     dataset = load_dataset("flaviagiammarino/path-vqa")
    
#     print(f"数据集分割: {list(dataset.keys())}")
    
#     # 检查每个分割
#     for split_name, split_data in dataset.items():
#         print(f"\n--- {split_name.upper()} 分割 ---")
#         print(f"样本数量: {len(split_data)}")
        
#         if len(split_data) > 0:
#             # 检查第一个样本的字段
#             sample = split_data[0]
#             print(f"字段列表: {list(sample.keys())}")
            
#             # 显示每个字段的类型和示例值
#             for field_name, field_value in sample.items():
#                 if field_name == 'image':
#                     print(f"  {field_name}: {type(field_value)} - 形状: {field_value.size if hasattr(field_value, 'size') else 'N/A'}")
#                 else:
#                     print(f"  {field_name}: {type(field_value)} - 值: {field_value}")
            
#             # 显示前几个样本的问题和答案
#             print(f"\n前3个样本的问题和答案:")
#             for i in range(min(3, len(split_data))):
#                 sample = split_data[i]
#                 print(f"  样本 {i}:")
#                 print(f"    问题: {sample['question']}")
#                 print(f"    答案: {sample['answer']}")
#                 if 'id' in sample:
#                     print(f"    ID: {sample['id']}")
#                 print()

# def inspect_image_field(split='train', num_samples=10, cache_dir=None):
#     ds = load_dataset("flaviagiammarino/path-vqa", split=split, cache_dir=cache_dir)
#     print(f"{split}集样本总数: {len(ds)}")
#     for i in range(num_samples):
#         item = ds[i]
#         img = item['image']
#         print(f"\n样本{i}:")
#         print(f"类型: {type(img)}")
#         if hasattr(img, 'filename'):
#             print(f"filename: {img.filename}")
#         if isinstance(img, str):
#             print(f"字符串内容: {img[:100]}")
#         if hasattr(img, 'size'):
#             print(f"size: {img.size}")
#         if hasattr(img, 'mode'):
#             print(f"mode: {img.mode}")
#         if hasattr(img, 'shape'):
#             print(f"shape: {img.shape}")
#         # 你可以根据filename、字符串路径、size+mode、shape等做去重

# if __name__ == "__main__":
#     check_dataset_format() 
#     inspect_image_field(split='train', num_samples=10, cache_dir="/nas/leiwenhui/tys/PathVQA/Dataset/cache") 

ds = load_dataset("flaviagiammarino/path-vqa", split="train", cache_dir="/nas/leiwenhui/tys/PathVQA/Dataset/cache")
for i in range(5):
    item = ds[i]
    img = item['image']
    print(f"样本{i}: type={type(img)}")
    if hasattr(img, 'filename'):
        print("  filename:", img.filename)
    if isinstance(img, str):
        print("  路径字符串:", img)
    if hasattr(img, 'size'):
        print("  size:", img.size)
    if hasattr(img, 'mode'):
        print("  mode:", img.mode)
    if hasattr(img, 'shape'):
        print("  shape:", img.shape)
    print("  keys:", list(item.keys()))
    if 'id' in item:
        print("  id:", item['id']) 