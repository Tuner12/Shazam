import argparse
import os
import sys
import ruamel.yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from utils import cosine_lr_schedule

from vqaEvaluate import compute_vqa_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='multi_moe_pathvqa', help='choose medical vqa dataset(rad, pathvqa, slake)')
    parser.add_argument('--is_save_path', default=False)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_suffix', default='', help='output suffix, eg. ../rad_29_1')
    parser.add_argument('--output_dir', default='', help='the final output path, need not to assign')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # args.output_dir = '/nas/leiwenhui/tys/PathVQA/MUMC/output/vqa/' + args.dataset_use + args.output_suffix
    args.output_dir = '/nas/leiwenhui/tys/PathVQA/MUMC/output/' + args.dataset_use 
    # 修复YAML加载兼容性问题
    try:
        # 尝试使用新的YAML API
        from ruamel.yaml import YAML
        yaml_loader = YAML(typ='rt')
        with open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
            config = yaml_loader.load(f)
    except ImportError:
        # 回退到旧的API
        config = yaml.load(open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r'), Loader=yaml.Loader)
    
    res_file_path = '/nas/leiwenhui/tys/PathVQA/MUMC/output/multi_moe_pathvqa/result/multi_moe_pathvqa_vqa_result_35.json'
    # res_file_path = '%s/result/%s_vqa_result_<epoch>.json' % (args.output_dir, prefix)
    compute_vqa_acc(answer_list_path=config['pathvqa']['test_file'][0], epoch=35, res_file_path=res_file_path)
    