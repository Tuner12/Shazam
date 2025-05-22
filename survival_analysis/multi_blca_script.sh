#!/bin/bash
# 脚本名：launch_five_folds.sh

# ---------------- 通用参数 ---------------- #
CSV_PATH="dataset_csv/survival_by_case/TCGA_BLCA_Splits.csv"
SPLITS_DIR="splits82/TCGA_BLCA_survival_100"
ROOT_DIR="TCGA_BLCA_multi_features"
TEACHERS="gigapath_features hoptimus1_features phikon_v2_features uni_v2_features virchow2_features"

EPOCHS=20
LR=2e-4
LAMBDA_DIST=0.01

PYTHON_SCRIPT="multi_moe_distill.py"

# 创建日志文件夹
mkdir -p logs

# ---------------- 启动每个 fold ---------------- #
for FOLD in 0 1 2 3 4; do
    GPU=$FOLD   # fold0 → GPU0, fold1 → GPU1, ...

    echo "Launching fold ${FOLD} on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=$GPU stdbuf -oL python $PYTHON_SCRIPT \
        --csv "$CSV_PATH" \
        --splits_dir "$SPLITS_DIR" \
        --root "$ROOT_DIR" \
        --teachers $TEACHERS \
        --epochs $EPOCHS \
        --lr $LR \
        --lambda_dist $LAMBDA_DIST \
        --fold_idx $FOLD \
        > logs/blcafold${FOLD}.log 2>&1 &
done

echo "All folds launched. Check logs/ for real-time output."
