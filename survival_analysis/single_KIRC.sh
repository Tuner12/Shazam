#!/bin/bash
# 脚本：launch_five_models_tmux.sh
# 功能：每个teacher开一个独立的tmux session，分别训练，分别日志

# 通用参数
CSV_PATH="dataset_csv/survival_by_case/TCGA_KIRC_Splits.csv"
SPLITS_DIR="splits82/TCGA_KIRC_survival_100"
ROOT_DIR="TCGA_KIRC_multi_features"
SCRIPT_NAME="single_model.py"
EPOCHS=100
LR=2e-4

# teacher列表 + 分配的GPU
declare -A teacher_gpu_map=(
  ["virchow2_features"]=0
  ["uni_v2_features"]=1
  ["phikon_v2_features"]=2
  ["gigapath_features"]=3
  ["hoptimus1_features"]=4
)

# 创建日志文件夹
mkdir -p single_log

# 遍历每个teacher
for teacher in "${!teacher_gpu_map[@]}"; do
    gpu_id=${teacher_gpu_map[$teacher]}
    session_name="train_${teacher}"

    # 先杀掉可能存在的同名session
    tmux kill-session -t $session_name 2>/dev/null

    # 创建新的 tmux session 并在里面执行命令
    tmux new-session -d -s $session_name "
        CUDA_VISIBLE_DEVICES=$gpu_id python $SCRIPT_NAME \
            --csv $CSV_PATH \
            --splits_dir $SPLITS_DIR \
            --root $ROOT_DIR \
            --teacher $teacher \
            --epochs $EPOCHS \
            --lr $LR | tee single_log/KIRC/${teacher}.log
    "
    echo "Launched $session_name on GPU $gpu_id"
done

echo "All training sessions launched. Use 'tmux ls' to see them."
