#!/bin/bash

CSV_PATH="dataset_csv/BRCA_subtyping.csv"
SPLITS_DIR="splits712/TCGA_BRCA_subtyping_100"
ROOT_DIR="/data2/leiwenhui/Data/Extracted_Feature/TCGA_BRCA_multi_features"
SCRIPT_NAME="single_model4cls.py"
EPOCHS=20
LR=2e-4
# Fold=1

# teacher列表 + 分配的GPU
declare -A teacher_gpu_map=(
  ["virchow2_features"]=0
  ["uni_v2_features"]=0
  ["phikon_v2_features"]=1
  ["gigapath_features"]=2
  ["hoptimus1_features"]=3
)

# 创建日志文件夹
mkdir -p single_log/BRCA4cls

# 创建一个新的 tmux 会话
if tmux has-session -t BRCA4cls 2>/dev/null; then
    echo "Session BRCA4cls exists, killing it..."
    tmux kill-session -t BRCA4cls
fi

tmux new-session -d -s "BRCA4cls"
if [ $? -eq 0 ]; then
    echo "tmux session 'BRCA4cls' created successfully."
else
    echo "Failed to create tmux session 'BRCA4cls'."
    exit 1
fi

# 遍历每个teacher，并为每个任务创建一个 tmux 窗口
for teacher in "${!teacher_gpu_map[@]}"; do
    gpu_id=${teacher_gpu_map[$teacher]}
    
    # 在 tmux 中为每个任务创建一个新窗口
    tmux new-window -t "BRCA4cls" -n "$teacher" "
        CUDA_VISIBLE_DEVICES=$gpu_id python $SCRIPT_NAME \
            --csv $CSV_PATH \
            --splits_dir $SPLITS_DIR \
            --root $ROOT_DIR \
            --teachers $teacher \
            --epochs $EPOCHS \
            --fold_idx 0 \
            --lr $LR | tee single_log/BRCA4cls/${teacher}.log; exec bash
    "
    echo "Launched training for $teacher on GPU $gpu_id"
done

# 回到第一个窗口，保持 tmux 会话运行
tmux select-window -t "BRCA4cls:1"

echo "All training sessions launched. Use 'tmux ls' to see them."
