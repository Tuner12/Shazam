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
from dataset import create_dataset1, create_sampler, create_loader, vqa_collate_fn
from utils import cosine_lr_schedule

from vqaEvaluate import compute_vqa_acc
# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question, answer, train=True, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation(model, data_loader, device, config):
    # validation
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    print_freq = 50

    total_loss = 0
    num_batches = 0

    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        
        # 计算验证损失
        loss = model(image, question, answer, train=True, alpha=config['alpha'])
        
        total_loss += loss.item()
        num_batches += 1
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print("Validation stats:", metric_logger.global_avg())
    return avg_loss


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            # 处理ques_id，可能是字符串或张量
            if hasattr(ques_id, 'item'):
                ques_id = int(ques_id.item())
            else:
                ques_id = int(ques_id)
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def main(args, config,model_name):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa {} datasets'.format(args.dataset_use))
    datasets = create_dataset1(args.dataset_use, config,model_name)
    
    # 处理数据集返回值，pathvqa返回3个数据集，其他数据集返回2个
    if args.dataset_use == 'pathvqa':
        train_dataset, val_dataset, test_dataset = datasets
        datasets = [train_dataset, val_dataset, test_dataset]
        print('train dataset size: ', len(train_dataset))
        print('val dataset size: ', len(val_dataset))
        print('test dataset size: ', len(test_dataset))
        feature_dim = train_dataset.get_feature_dim()
    else:
        train_dataset, test_dataset = datasets
        val_dataset = None
        datasets = [train_dataset, test_dataset]
        print('train dataset size: ', len(train_dataset))
        print('test dataset size: ', len(test_dataset))
        feature_dim = train_dataset.get_feature_dim()
    
    print('feature_dim: ', feature_dim)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dataset_use == 'pathvqa':
            samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
        else:
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        if args.dataset_use == 'pathvqa':
            samplers = [None, None, None]
        else:
            samplers = [None, None]

    if args.dataset_use == 'pathvqa':
        train_loader, val_loader, test_loader = create_loader(datasets, samplers,
                                                  batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                  num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                  collate_fns=[vqa_collate_fn, vqa_collate_fn, None])
    else:
        train_loader, test_loader = create_loader(datasets, samplers,
                                                  batch_size=[config['batch_size_train'], config['batch_size_test']],
                                                  num_workers=[4, 4], is_trains=[True, False],
                                                  collate_fns=[vqa_collate_fn, None])
        val_loader = None

    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    tokenizer = BertTokenizer.from_pretrained('/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    #### Creating Model ####
    print("Creating model")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer,image_embeds_dim=feature_dim)
    model = model.to(device)
    # print(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # state_dict = checkpoint

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]

                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    start_epoch = 0
    print("\nStart training\n")
    start_time = time.time()
    
    # 初始化最佳验证损失追踪
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(start_epoch, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config)
            
            # 验证集评估（仅对pathvqa数据集）
            if args.dataset_use == 'pathvqa' and val_loader is not None:
                val_loss = validation(model, val_loader, device, config)
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    print(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch}")
                    
                    if utils.is_main_process():
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss,
                        }
                        prefix = args.checkpoint.split('/')[-1].split('.')[0] if args.checkpoint else 'model'
                        torch.save(save_obj, os.path.join(args.output_dir, f'{prefix}_best_val.pth'))
                        print(f"Saved best model to {prefix}_best_val.pth")

        if args.evaluate:
            break

        if utils.is_main_process():

            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'config': config,
                # 'epoch': epoch,
            }
            prefix = args.checkpoint.split('/')[-1].split('.')[0] if args.checkpoint else 'model'
            if args.is_save_path and epoch > 20:
                torch.save(save_obj, os.path.join(args.output_dir, '%s_rad_%02d.pth' % (prefix, epoch)))
            vqa_result = evaluation(model, test_loader, device, config)
            json.dump(vqa_result, open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))

        if args.distributed:
            dist.barrier()

    # 训练结束后报告最佳结果
    if args.dataset_use == 'pathvqa':
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # compute acc
    res_file_path = '%s/result/%s_vqa_result_<epoch>.json' % (args.output_dir, prefix)
    compute_vqa_acc(answer_list_path=config[args.dataset_use]['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='pathvqa', help='choose medical vqa dataset(rad, pathvqa, slake)')
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
    model_name = 'gigapath'
    args.output_dir = '/nas/leiwenhui/tys/PathVQA/MUMC/output1/' + model_name  + args.dataset_use + args.output_suffix
    os.makedirs(args.output_dir, exist_ok=True)
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
  
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("config: ", config)
    print("args: ", args)
    
    main(args, config, model_name)
