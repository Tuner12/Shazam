#!/bin/bash
# export LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

MODELS=("virchow2" "uni_v2" "phikon_v2"  "gigapath" "hoptimus0")
# MODELS=("phikon_v2" "hoptimus0")
for model in "${MODELS[@]}"
do
  echo "==== Processing with model: $model ===="

  python extract_multi_features.py \
    --model_name $model \
    --data_h5_dir ../survival_analysis/TCGA_BLCA_PATCH_DIR1 \
    --data_slide_dir ../survival_analysis/TCGA_BLCA_directory \
    --csv_path ../survival_analysis/TCGA_BLCA_PATCH_DIR1/process_list_autogen.csv \
    --feat_dir ../survival_analysis/TCGA_BLCA_multi_features/${model}_features \
    --batch_size 512 \
    --slide_ext .svs
done
