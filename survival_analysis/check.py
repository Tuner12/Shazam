# import torch

# # 修改为你想查看的 .pt 文件路径
# pt_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_multi_features/virchow2_features/part0/pt_files/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.pt"

# # 加载 .pt 文件
# low, mid, high = torch.load(pt_path, map_location='cpu')
# print("Low feature shape:", low.shape, "| size:", low.size())
# print("Mid feature shape:", mid.shape, "| size:", mid.size())
# print("High feature shape:", high.shape, "| size:", high.size())

# # # 打印 shape 信息
# # print("Low feature shape:", low.shape)
# # print("Mid feature shape:", mid.shape)
# # print("High feature shape:", high.shape)

# # 总 patch 数
# total_patches = high.shape[0]
# print("Total patches:", total_patches)

# # 每个张量的大小（float32 是 4 字节）
# def tensor_memory_MB(tensor):
#     return tensor.numel() * 4 / 1024**2  # MB

# mem_low = tensor_memory_MB(low)
# mem_mid = tensor_memory_MB(mid)
# mem_high = tensor_memory_MB(high)
# mem_total = mem_low + mem_mid + mem_high

# print(f"Memory (Low):  {mem_low:.2f} MB")
# print(f"Memory (Mid):  {mem_mid:.2f} MB")
# print(f"Memory (High): {mem_high:.2f} MB")
# print(f"Total Memory:  {mem_total:.2f} MB / {mem_total / 1024:.2f} GB")
# import openslide

# # 替换为你具体的 .svs 文件路径
# svs_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs"
# svs_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_directory/TCGA-A3-3306-01Z-00-DX1.bfd320d3-f3ec-4015-b34a-98e9967ea80d.svs"

# # 打开 WSI
# slide = openslide.OpenSlide(svs_path)

# # 获取 Level 0 的尺寸
# width, height = slide.level_dimensions[0]
# print(f"Level 0 size: {width} x {height}")

# # Patch 尺寸
# patch_size = 512


# num_patches_x = width // patch_size
# num_patches_y = height // patch_size
# total_patches = num_patches_x * num_patches_y

# print(f"Patches along width: {num_patches_x}")
# print(f"Patches along height: {num_patches_y}")
# print(f"Total patch count: {total_patches}")

# h5_path =  "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_PATCH_DIR40/patches/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.h5"
# h5_path =  "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_PATCH_DIR40/patches/TCGA-A3-3306-01Z-00-DX1.bfd320d3-f3ec-4015-b34a-98e9967ea80d.h5"
# import h5py

# # h5_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_PATCH_DIR40/patches/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.h5"

# with h5py.File(h5_path, "r") as f:
#     coords = f['coords'][:]
#     patch_level = f['coords'].attrs.get('patch_level', 'N/A')
#     patch_size = f['coords'].attrs.get('patch_size', 'N/A')

# print(f"Total number of coordinates (patches): {coords.shape[0]}")
# print(f"Patch level: {patch_level}")
# print(f"Patch size: {patch_size}")
# print("First 5 patch coordinates:")
# print(coords[:5])
# import torch
# from pathlib import Path

# # 五个teacher的目录
# root_dir = Path('/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_multi_features')  # 比如 'TCGA_BLCA_multi_features'
# teacher_names = ['gigapath_features', 'hoptimus0_features', 'phikon_v2_features', 'uni_v2_features', 'virchow2_features']

# # Slide名字
# slide_id = 'TCGA-S5-A6DX-01Z-00-DX1.70418D45-0396-4838-BF0C-588C7719A131'

# for teacher in teacher_names:
#     pt_path = root_dir / f"{teacher}/merged_pt_files/{slide_id}.pt"
    
#     try:
#         sample = torch.load(pt_path, map_location='cpu')
#         low, mid, high = sample
#         print(f"[{teacher}]")
#         print(f"  low  shape: {low.shape}")
#         print(f"  mid  shape: {mid.shape}")
#         print(f"  high shape: {high.shape}")
#         print("-" * 50)
#     except Exception as e:
#         print(f"[{teacher}] Failed to load: {e}")
#!/usr/bin/env python3
"""
检查一批.pt特征文件中是否存在 NaN 或 Inf，找出损坏的文件。
"""

import torch
import os
from pathlib import Path
from tqdm import tqdm

# ----------------- 配置区 ----------------- #
feature_dirs = [  # 你的 teacher 特征文件夹
    "TCGA_BLCA_multi_features/gigapath_features/merged_pt_files",
    "TCGA_BLCA_multi_features/hoptimus0_features/merged_pt_files",
    "TCGA_BLCA_multi_features/phikon_v2_features/merged_pt_files",
    "TCGA_BLCA_multi_features/uni_v2_features/merged_pt_files",
    "TCGA_BLCA_multi_features/virchow2_features/merged_pt_files",
]
save_bad_list = "bad_files.txt"  # 检查出坏文件后保存到这里
# ------------------------------------------ #

bad_files = []

for dir_path in feature_dirs:
    dir_path = Path(dir_path)
    pt_files = list(dir_path.glob("*.pt"))
    print(f"Checking {len(pt_files)} files in {dir_path}...")

    for pt_file in tqdm(pt_files):
        try:
            low, mid, high = torch.load(pt_file, map_location='cpu', weights_only=True)

            for name, feat in zip(['low', 'mid', 'high'], [low, mid, high]):
                if not torch.isfinite(feat).all():
                    print(f"[BAD] {pt_file.name} ({name}) contains NaN/Inf")
                    bad_files.append(str(pt_file))
                    break  # 只要任意一层坏了，就标记坏文件

        except Exception as e:
            print(f"[ERROR] Failed to load {pt_file}: {e}")
            bad_files.append(str(pt_file))  # 读都读不出来直接算坏

print("\n====== Summary ======")
print(f"Total bad files: {len(bad_files)}")

if bad_files:
    with open(save_bad_list, "w") as f:
        for item in bad_files:
            f.write(f"{item}\n")
    print(f"Bad file list saved to {save_bad_list}")
else:
    print("No bad files found.")

