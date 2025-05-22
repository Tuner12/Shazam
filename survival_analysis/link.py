# import pandas as pd
# import os
# import glob

# # 1. 路径设置
# csv_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv"  # 你刚上传的 CSV
# wsi_root = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/Pathology/TCGA/TCGA-BLCA"
# link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory"

# os.makedirs(link_dir, exist_ok=True)

# # 2. 读取 CSV 中的 slide_id 字段
# df = pd.read_csv(csv_path)
# pt_paths = df['slide_id'].astype(str)

# # 3. 遍历所有路径
# for pt_path in pt_paths:
#     if not pt_path.endswith('.pt'):
#         continue
#     svs_name = os.path.basename(pt_path).replace('.pt', '.svs')

#     # 在 wsi_root 中递归查找这个 .svs 文件
#     matches = glob.glob(os.path.join(wsi_root, '**', svs_name), recursive=True)
#     if len(matches) == 0:
#         print(f"❌ Not found: {svs_name}")
#         continue

#     src_path = matches[0]
#     dst_path = os.path.join(link_dir, svs_name)

#     try:
#         if not os.path.exists(dst_path):
#             os.symlink(src_path, dst_path)
#             print(f"✅ Linked: {svs_name}")
#         else:
#             print(f"⚠️ Already exists: {svs_name}")
#     except Exception as e:
#         print(f"⚠️ Failed to link {svs_name}: {e}")
import pandas as pd
import os
import glob

# 1. 路径设置
csv_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv"
wsi_root = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/Pathology/TCGA/TCGA-BRCA"
link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BRCA_directory"

os.makedirs(link_dir, exist_ok=True)

# 2. 读取 CSV 中的 slide_id 字段
df = pd.read_csv(csv_path)
pt_paths = df['slide_id'].astype(str)

# 3. 初始化计数器
total = 0
linked = 0
skipped = 0
not_found = 0
failed = 0

# 4. 遍历所有路径
for pt_path in pt_paths:
    if not pt_path.endswith('.pt'):
        continue

    total += 1
    svs_name = os.path.basename(pt_path).replace('.pt', '.svs')

    matches = glob.glob(os.path.join(wsi_root, '**', svs_name), recursive=True)
    if len(matches) == 0:
        print(f"❌ Not found: {svs_name}")
        not_found += 1
        continue

    src_path = matches[0]
    dst_path = os.path.join(link_dir, svs_name)

    try:
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)
            print(f"✅ Linked: {svs_name}")
            linked += 1
        else:
            print(f"⚠️ Already exists: {svs_name}")
            skipped += 1
    except Exception as e:
        print(f"⚠️ Failed to link {svs_name}: {e}")
        failed += 1

# 5. 打印汇总信息
print("\n📊 Summary:")
print(f"Total .pt entries processed: {total}")
print(f"✅ Linked: {linked}")
print(f"⚠️ Skipped (already exists): {skipped}")
print(f"❌ Not found: {not_found}")
print(f"💥 Failed to link: {failed}")
