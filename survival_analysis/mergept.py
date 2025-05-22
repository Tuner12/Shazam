import os
import shutil
from glob import glob

# 所有模型目录路径（你可以添加更多）
base_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_multi_features"
model_names = ["virchow2_features", "uni_v2_features", "phikon_v2_features", "gigapath_features", "hoptimus0_features", "hoptimus1_features"]
# model_names = ["hoptimus1_features"]

for model in model_names:
    model_dir = os.path.join(base_dir, model)
    merged_dir = os.path.join(model_dir, "merged_pt_files")
    os.makedirs(merged_dir, exist_ok=True)

    # 遍历所有 part*/pt_files/*.pt 文件
    part_dirs = glob(os.path.join(model_dir, "part*/pt_files/*.pt"))

    print(f"🔄 Moving {len(part_dirs)} files for {model}...")

    for pt_file in part_dirs:
        filename = os.path.basename(pt_file)
        dst = os.path.join(merged_dir, filename)

        # 避免重复移动
        if not os.path.exists(dst):
            shutil.move(pt_file, dst)
        else:
            print(f"⚠️ Skipped existing: {filename}")

    print(f"✅ Done moving for {model}: {merged_dir}")
