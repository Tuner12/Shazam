# import pandas as pd
# import os

# csv_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_PATCH_DIR/process_list_autogen.csv"
# output_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_PATCH_DIR"
# num_splits = 16

# df = pd.read_csv(csv_path)
# splits = [df.iloc[i::num_splits] for i in range(num_splits)]

# for i, split in enumerate(splits):
#     split_path = os.path.join(output_dir, f"process_list_autogen_part{i}.csv")
#     split.to_csv(split_path, index=False)


import pandas as pd
import os

csv_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_PATCH_DIR40/process_list_autogen.csv"
output_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_PATCH_DIR40"
num_splits = 16

df = pd.read_csv(csv_path)
splits = [df.iloc[i::num_splits] for i in range(num_splits)]

for i, split in enumerate(splits):
    split_path = os.path.join(output_dir, f"process_list_autogen_part{i}.csv")
    split.to_csv(split_path, index=False)
    print(f"Saved part {i} with {len(split)} entries to {split_path}")

