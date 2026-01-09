CSV_PATH="/data2/tanyusheng/Code/Survival/dataset_csv/survival_by_case/TCGA_BRCA_Splits.csv"
SPLITS_DIR="/data2/tanyusheng/Code/Survival/splits82/TCGA_BRCA_survival_100"
ROOT_DIR="/data2/tanyusheng/Data/Extracted_Feature/TCGA_BRCA_multi_features"
SCRIPT_NAME="single_model.py"
EPOCHS=20
LR=2e-4
# Fold=1

# teacher列表 + 分配的GPU
declare -A teacher_gpu_map=(
  ["virchow2_features"]=0
  ["uni_v2_features"]=1
  ["phikon_v2_features"]=2
  ["gigapath_features"]=3
  ["hoptimus1_features"]=4
)

# 创建日志文件夹
mkdir -p single_log_nll/BRCA

# 创建一个新的 tmux 会话
if tmux has-session -t BRCA_NLL 2>/dev/null; then
    echo "Session BRCA_NLL exists, killing it..."
    tmux kill-session -t BRCA_NLL
fi

tmux new-session -d -s "BRCA_NLL"
# tmux new-session -d -s "BLCA" 
if [ $? -eq 0 ]; then
    echo "tmux session 'BRCA_NLL' created successfully."
else
    echo "Failed to create tmux session 'BRCA_NLL'."
    exit 1
fi

# 遍历每个teacher，并为每个任务创建一个 tmux 窗口
for teacher in "${!teacher_gpu_map[@]}"; do
    gpu_id=${teacher_gpu_map[$teacher]}
    
    # 在 tmux 中为每个任务创建一个新窗口
    tmux new-window -t "BRCA_NLL" -n "$teacher" "
        CUDA_VISIBLE_DEVICES=$gpu_id python $SCRIPT_NAME \
            --csv $CSV_PATH \
            --splits_dir $SPLITS_DIR \
            --root $ROOT_DIR \
            --teacher $teacher \
            --epochs $EPOCHS \
            --lr $LR | tee single_log_nll/BRCA/${teacher}.log; exec bash
    "
    echo "Launched training for $teacher on GPU $gpu_id"
done

# 回到第一个窗口，保持 tmux 会话运行
tmux select-window -t "BRCA:1"

echo "All training sessions launched. Use 'tmux ls' to see them."
