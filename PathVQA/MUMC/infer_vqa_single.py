import argparse
import os
import json
import time
from pathlib import Path

import torch

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset import create_dataset1, create_sampler, create_loader, vqa_collate_fn
import utils

# 评测与指标
from vqaEvaluate import compute_vqa_acc, compute_vqa_acc_topk
# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'

def evaluation(model, data_loader, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Inference: VQA single model'
    print_freq = 50

    result = []
    result_topk = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    with torch.no_grad():
        for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

            for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
                if hasattr(ques_id, 'item'):
                    ques_id = int(ques_id.item())
                else:
                    ques_id = int(ques_id)
                topk_id_list = topk_id.tolist()
                topk_prob_list = topk_prob.detach().cpu().tolist()

                # 将答案和概率配对并按概率降序排序
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

                # Top-1答案
                result.append({"qid": ques_id, "answer": topk_entries[0]["answer"]})
                # 保存Top-K详情（包括概率，便于后续误差分析）
                result_topk.append({"qid": ques_id, "answers": topk_entries})
    return result, result_topk


def main():
    # os.environ['http_proxy'] = 'http://192.168.1.18:7890'
    # os.environ['https_proxy'] = 'http://192.168.1.18:7890'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='pathvqa', help='pathvqa')
    parser.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
    parser.add_argument('--output_dir', default='/data2/tanyusheng/Code/PathVQA/MUMC/tys', help='inference output root')
    # parser.add_argument('--text_encoder', default='bert-base-uncased')
    # parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--text_encoder', default='/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model')
    parser.add_argument('--text_decoder', default='/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_name', default='gigapath', help='name tag for output prefix, e.g., gigapath/virchow2')
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

    # 设备、随机种子
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)

    # 创建数据集/加载器
    print('Creating VQA datasets for inference:', args.dataset_use)
    datasets = create_dataset1(args.dataset_use, config, args.model_name)
    if args.dataset_use == 'pathvqa':
        train_dataset, val_dataset, test_dataset = datasets
        datasets = [train_dataset, val_dataset, test_dataset]
        test_dataset = test_dataset
    else:
        train_dataset, test_dataset = datasets
    feature_dim = train_dataset.get_feature_dim()

    samplers = [None, None, None] if args.dataset_use == 'pathvqa' else [None, None]
    if args.dataset_use == 'pathvqa':
        _, _, test_loader = create_loader(datasets, samplers,
                                          batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                          collate_fns=[vqa_collate_fn, vqa_collate_fn, None])
    else:
        _, test_loader = create_loader(datasets, samplers,
                                       batch_size=[config['batch_size_train'], config['batch_size_test']],
                                       num_workers=[4, 4], is_trains=[True, False],
                                       collate_fns=[vqa_collate_fn, None])

    # 构建与加载模型
    tokenizer = BertTokenizer.from_pretrained('/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder,
                     tokenizer=tokenizer, image_embeds_dim=feature_dim)
    model = model.to(device)

    print('Loading checkpoint:', args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    # 兼容位置编码
    if 'visual_encoder.pos_embed' in state_dict:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    msg = model.load_state_dict(state_dict, strict=False)
    print('load state dict msg:', msg)

    # 输出目录
    out_root = os.path.join(args.output_dir, args.model_name)
    result_dir = os.path.join(out_root, 'result')
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # 推理
    start = time.time()
    result, result_topk = evaluation(model, test_loader, device, config)
    print('Inference done, time: %.2fs' % (time.time() - start))

    # 统一命名：epoch=0，便于沿用compute_vqa_acc接口
    prefix = args.model_name
    result_path = os.path.join(result_dir, f'{prefix}_vqa_result_0.json')
    result_topk_path = os.path.join(result_dir, f'{prefix}_vqa_result_0_topk.json')

    json.dump(result, open(result_path, 'w'))
    json.dump(result_topk, open(result_topk_path, 'w'))

    # 计算指标（acc与compare，以及acc@1/3/5汇总）
    if args.dataset_use in config and 'test_file' in config[args.dataset_use]:
        answer_list_path = config[args.dataset_use]['test_file'][0]
    else:
        # 回退：与multi_moe一致
        answer_list_path = config['pathvqa']['test_file'][0]

    res_file_tmpl = os.path.join(result_dir, f'{prefix}_vqa_result_<epoch>.json')
    compute_vqa_acc(answer_list_path=answer_list_path, epoch=1, res_file_path=res_file_tmpl)
    compute_vqa_acc_topk(answer_list_path=answer_list_path, epoch=1, res_file_path=res_file_tmpl)

    print('All done. Outputs in:', out_root)


if __name__ == '__main__':
    # os.environ['http_proxy'] = 'http://192.168.1.18:7890'
    # os.environ['https_proxy'] = 'http://192.168.1.18:7890'
    main() 