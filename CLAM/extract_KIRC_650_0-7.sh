#!/bin/bash

# 设置模型列表
MODELS=("virchow2" "uni_v2" "phikon_v2" "gigapath" "hoptimus0")
MODELS=("hoptimus1")
# 设置数据 CSV 文件偏移量（如果在另一个服务器上，可设为 8）
OFFSET=0

# 遍历模型
for model in "${MODELS[@]}"; do
  echo "==== Processing model: $model ===="
  # 设置每个模型的 batch size（hoptimus0 用较小值）
  if [ "$model" = "hoptimus1" ]; then
    BATCH_SIZE=256
  else
    BATCH_SIZE=512
  fi
  # 启动 8 张卡并行提取同一模型的不同数据部分
  for i in {0..7}; do
    csv_index=$((OFFSET + i))
    feat_dir="../survival_analysis/TCGA_KIRC_multi_features/${model}_features/part${csv_index}"
    mkdir -p "$feat_dir"
    CUDA_VISIBLE_DEVICES=$i python extract_multi_features.py \
      --model_name $model \
      --data_h5_dir ../survival_analysis/TCGA_KIRC_PATCH_DIR40 \
      --data_slide_dir ../survival_analysis/TCGA_KIRC_directory \
      --csv_path ../survival_analysis/TCGA_KIRC_PATCH_DIR40/process_list_autogen_part${csv_index}.csv \
      --feat_dir "$feat_dir" \
      --batch_size $BATCH_SIZE \
      --slide_ext .svs \
      > ../survival_analysis/logs/KIRC/${model}_gpu${csv_index}.txt 2>&1 &
  done

  # 等待当前模型的 8 个子进程全部结束
  wait
  echo "==== Completed model: $model ===="
done
