import argparse
import os
import json
import time
from pathlib import Path

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_vqa_multi_moe import MUMC_VQA_Multi_MoE
from models.tokenization_bert import BertTokenizer
from dataset import create_dataset1, create_sampler, create_loader, multi_moe_collate_fn
import utils

# 评测与指标
from vqaEvaluate import compute_vqa_acc, compute_vqa_acc_topk


def evaluation(model, data_loader, device, config, model_dims_low, model_dims_mid, model_dims_high):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Inference: VQA multi-MoE'
    print_freq = 50

    result = []
    result_topk = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    with torch.no_grad():
        for n, (low_feature, mid_feature, high_feature, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            low_feature = low_feature.to(device, non_blocking=True)
            mid_feature = mid_feature.to(device, non_blocking=True)
            high_feature = high_feature.to(device, non_blocking=True)

            features = []
            low_features, mid_features, high_features = [], [], []
            start_idx = 0
            for dim in model_dims_low:
                end_idx = start_idx + dim
                low_features.append(low_feature[:, start_idx:end_idx])
                start_idx = end_idx
            start_idx = 0
            for dim in model_dims_mid:
                end_idx = start_idx + dim
                mid_features.append(mid_feature[:, start_idx:end_idx])
                start_idx = end_idx
            start_idx = 0
            for dim in model_dims_high:
                end_idx = start_idx + dim
                high_features.append(high_feature[:, start_idx:end_idx])
                start_idx = end_idx
            for j in range(5):
                features.append(low_features[j])
                features.append(mid_features[j])
                features.append(high_features[j])

            topk_ids, topk_probs, out1, out2, out3 = model(features, question, answer_list, train=False, k=config['k_test'])

            for qid, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                if hasattr(qid, 'item'):
                    qid = int(qid.item())
                else:
                    qid = int(qid)

                topk_id_list = topk_id.tolist()
                topk_prob_list = topk_prob.detach().cpu().tolist()

                topk_pairs = sorted(
                    zip(topk_id_list, topk_prob_list),
                    key=lambda pair: pair[1],
                    reverse=True
                )

                topk_entries = []
                for idx, prob in topk_pairs[:min(5, len(topk_pairs))]:
                    answer_text = data_loader.dataset.answer_list[idx]
                    topk_entries.append({
                        "answer": answer_text,
                        "prob": float(prob)
                    })

                if not topk_entries:
                    continue

                result.append({"qid": qid, "answer": topk_entries[0]["answer"]})
                result_topk.append({"qid": qid, "answers": topk_entries})
    return result, result_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='multi_moe_pathvqa', help='multi_moe_pathvqa')
    parser.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
    parser.add_argument('--output_dir', default='/data2/tanyusheng/Code/PathVQA/MUMC/tys', help='inference output root')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # 读取配置
    try:
        from ruamel.yaml import YAML
        yaml_loader = YAML(typ='rt')
        with open('/data2/tanyusheng/Code/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
            config = yaml_loader.load(f)
    except Exception:
        import ruamel.yaml as yaml
        config = yaml.load(open('/data2/tanyusheng/Code/PathVQA/MUMC/configs/VQA.yaml', 'r'), Loader=yaml.Loader)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)

    # 数据集/Loader
    print('Creating multi-moe vqa datasets for inference')
    datasets = create_dataset1('multi_moe_pathvqa', config, 'multi_moe')
    train_dataset, val_dataset, test_dataset = datasets
    datasets = [train_dataset, val_dataset, test_dataset]
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                                          batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[multi_moe_collate_fn, multi_moe_collate_fn, None])

    # 自动读取各模型特征维度（与训练保持一致）
    model_names = ['virchow2', 'uni_v2', 'phikon_v2', 'hoptimus1', 'gigapath']
    base_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'
    dim_list_low, dim_list_mid, dim_list_high = [], [], []
    for model_name in model_names:
        feature_file = os.path.join(base_path, f'train_{model_name}_features.pt')
        feature_data = torch.load(feature_file, map_location='cpu')
        dim_list_low.append(feature_data[0].shape[1])
        dim_list_mid.append(feature_data[1].shape[1])
        dim_list_high.append(feature_data[2].shape[1])

    tokenizer = BertTokenizer.from_pretrained('/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    model = MUMC_VQA_Multi_MoE(
        text_encoder='/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model',
        text_decoder='/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model',
        tokenizer=tokenizer,
        config=config,
        dim_list_low=dim_list_low,
        dim_list_mid=dim_list_mid,
        dim_list_high=dim_list_high,
        d_model=config.get('d_model', 128),
        num_layers=config.get('num_layers', 15)
    ).to(device)

    print('Loading checkpoint:', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print('load state dict msg:', msg)

    out_root = args.output_dir
    result_dir = os.path.join(out_root, 'result')
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    result, result_topk = evaluation(model, test_loader, device, config,
                                     model_dims_low=dim_list_low, model_dims_mid=dim_list_mid, model_dims_high=dim_list_high)
    print('Inference done, time: %.2fs' % (time.time() - start))

    prefix = 'multi_moe_pathvqa'
    result_path = os.path.join(result_dir, f'{prefix}_vqa_result_0.json')
    result_topk_path = os.path.join(result_dir, f'{prefix}_vqa_result_0_topk.json')
    json.dump(result, open(result_path, 'w'))
    json.dump(result_topk, open(result_topk_path, 'w'))

    # 计算指标（acc与compare，以及acc@1/3/5汇总）
    answer_list_path = config['pathvqa']['test_file'][0]
    res_file_tmpl = os.path.join(result_dir, f'{prefix}_vqa_result_<epoch>.json')
    compute_vqa_acc(answer_list_path=answer_list_path, epoch=1, res_file_path=res_file_tmpl)
    compute_vqa_acc_topk(answer_list_path=answer_list_path, epoch=1, res_file_path=res_file_tmpl)

    print('All done. Outputs in:', out_root)


if __name__ == '__main__':
    main() 