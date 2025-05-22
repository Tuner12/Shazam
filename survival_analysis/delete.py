import os
import pandas as pd
import glob

# 1. 路径设置
csv_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv"
wsi_root = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/Pathology/TCGA/TCGA-BLCA"

# 2. 读取 CSV 中需要保留的 .svs 文件名
df = pd.read_csv(csv_path)
pt_paths = df['slide_id'].astype(str)
keep_svs_set = set([os.path.basename(p).replace('.pt', '.svs') for p in pt_paths if p.endswith('.pt')])
print(f"✅ Total .svs files to keep: {len(keep_svs_set)}")
# 3. 遍历 wsi_root 中所有 .svs 文件，删除不在 keep_svs_set 中的
all_svs_files = glob.glob(os.path.join(wsi_root, "**", "*.svs"), recursive=True)

removed = 0
for svs_path in all_svs_files:
    svs_name = os.path.basename(svs_path)
    if svs_name not in keep_svs_set:
        try:
            os.remove(svs_path)
            print(f"🗑️ Deleted: {svs_path}")
            removed += 1
        except Exception as e:
            print(f"❌ Failed to delete {svs_path}: {e}")

print(f"\n✅ Cleanup complete. Total deleted: {removed}")
