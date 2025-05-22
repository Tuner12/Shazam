# import openslide

# svs_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs"
# slide = openslide.OpenSlide(svs_path)

# # 打印层数
# print("Level count:", slide.level_count)

# # 打印每层的尺寸和缩放比例
# for i in range(slide.level_count):
#     print(f"Level {i}: size = {slide.level_dimensions[i]}, downsample = {slide.level_downsamples[i]}")

# # 获取放大倍数
# objective_power = slide.properties.get("aperio.AppMag", "Unknown")
# print("Objective magnification:", objective_power + "x")


# from openslide import OpenSlide
# from PIL import Image
# wsi = OpenSlide("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs")
# # # 原始放大倍数
# # orig_mag = float(slide.properties.get("openslide.objective-power", 0))
# # # level 1 的下采样倍率
# # downsample1 = float(slide.properties.get("openslide.level[3].downsample", 1.0))
# # # downsample1 = slide.level_downsamples[1]
# # # 计算 level 1 对应的放大倍数
# # mag_level1 = orig_mag / downsample1
# # print(f"Level 0: {orig_mag}×")
# # print(f"Level 1 downsample: ×{downsample1}")
# # print(f"→ Level 1: {mag_level1}×")
# coord = (10000, 10000)  # 替换为你自己的坐标
# patch_level = 0         # 40×
# patch_size = 512

# # 提取原图 patch（40×，512x512）
# img_40x = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')

# # Resize 成 256x256 → 等效于 20×
# img_20x = img_40x.resize((patch_size // 2, patch_size // 2), Image.BILINEAR)

# # 打印尺寸信息
# print(f"[INFO] 原图尺寸: {img_40x.size} (应为 512x512)")
# print(f"[INFO] 下采样后尺寸: {img_20x.size} (应为 256x256)")
# print(f"[INFO] 下采样倍率: {img_40x.size[0] // img_20x.size[0]}×")
import openslide
import h5py
import os

# === 替换为你的路径 ===
svs_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_directory/TCGA-B0-4710-01Z-00-DX1.e1440c30-b28d-42a8-b126-5abab7e0e3b2.svs"
h5_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_PATCH_DIR40/patches/TCGA-B0-4710-01Z-00-DX1.e1440c30-b28d-42a8-b126-5abab7e0e3b2.h5"

# === 打开 slide 文件 ===
print("Opening WSI...")
slide = openslide.OpenSlide(svs_path)
print("Slide opened successfully.")

# === 打印 slide 尺寸 ===
print("Level count:", slide.level_count)
print("Level 0 dimensions:", slide.level_dimensions[0])

# === 读取 patch 坐标 ===
print("Reading patch coordinates from h5...")
with h5py.File(h5_path, "r") as f:
    coords = f['coords'][:]
    print(f"Total patches: {len(coords)}")

# === 尝试读取所有 patch 区域 ===
patch_size = 512
error_count = 0
for i, (x, y) in enumerate(coords):
    try:
        _ = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
    except openslide.OpenSlideError as e:
        print(f"[Error] Failed to read patch {i} at ({x}, {y}): {e}")
        error_count += 1
    if i > 0 and i % 100 == 0:
        print(f"Checked {i} patches...")

print(f"\n✅ Done. Total errors: {error_count} / {len(coords)}")


# import h5py
# import numpy as np

# # 文件路径
# file_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_PATCH_DIR40/patches/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.h5"

# # 打开 HDF5 文件
# with h5py.File(file_path, "r") as f:
#     print("[INFO] Keys in the file:")
#     for key in f.keys():
#         data = f[key]
#         print(f"  - {key}: shape = {data.shape}, dtype = {data.dtype}")
        
#     print("\n[INFO] Example values:")
#     for key in f.keys():
#         data = f[key]
#         # 仅打印前几个数据预览
#         print(f"{key}[:5] =\n{data[:5]}")