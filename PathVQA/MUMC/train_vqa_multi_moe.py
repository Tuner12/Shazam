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
from models.model_vqa_multi_moe import MUMC_VQA_Multi_MoE, multi_level_distillation_loss
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset1, create_sampler, create_loader, multi_moe_collate_fn
from utils import cosine_lr_schedule
from vqaEvaluate import compute_vqa_acc

# =============== 训练函数 ===============
def train(model, data_loader, optimizer, epoch, device, config, model_dims_low, model_dims_mid, model_dims_high, lambda_distill=0.01):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('distill_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (low_feature, mid_feature, high_feature, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 将特征移到设备上
        low_feature = low_feature.to(device, non_blocking=True)
        mid_feature = mid_feature.to(device, non_blocking=True)
        high_feature = high_feature.to(device, non_blocking=True)
        
        # 使用传入的维度列表分割特征，按照模型期望的顺序重新排列
        features = []
        
        # 按照模型期望的顺序：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        
        # 分割low特征
        low_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_low):
            end_idx = start_idx + dim
            low_features.append(low_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 分割mid特征
        mid_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_mid):
            end_idx = start_idx + dim
            mid_features.append(mid_feature[:, start_idx:end_idx])
            start_idx = end_idx
            
        # 分割high特征
        high_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_high):
            end_idx = start_idx + dim
            high_features.append(high_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 按照模型期望的顺序重新排列：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        for j in range(5):
            features.append(low_features[j])   # low_j
            features.append(mid_features[j])   # mid_j
            features.append(high_features[j])  # high_j

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        # 前向传播
        loss, out1, out2, out3 = model(features, question, answer, train=True, alpha=alpha)
        
        # 蒸馏损失（如果需要）
        distill_loss = multi_level_distillation_loss(
            out1, out2, out3, features, model.num_layers, model
        ) if lambda_distill > 0 else torch.tensor(0.0, device=device)
        
        total_loss = loss + lambda_distill * distill_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(distill_loss=distill_loss.item())
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# =============== 验证函数 ===============
@torch.no_grad()
def validation(model, data_loader, device, config, model_dims_low, model_dims_mid, model_dims_high, lambda_distill=0.01):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    print_freq = 50

    total_loss = 0
    num_batches = 0

    for i, (low_feature, mid_feature, high_feature, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 将特征移到设备上
        low_feature = low_feature.to(device, non_blocking=True)
        mid_feature = mid_feature.to(device, non_blocking=True)
        high_feature = high_feature.to(device, non_blocking=True)
        
        # 使用传入的维度列表分割特征，按照模型期望的顺序重新排列
        features = []
        
        # 分割low特征
        low_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_low):
            end_idx = start_idx + dim
            low_features.append(low_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 分割mid特征
        mid_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_mid):
            end_idx = start_idx + dim
            mid_features.append(mid_feature[:, start_idx:end_idx])
            start_idx = end_idx
            
        # 分割high特征
        high_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_high):
            end_idx = start_idx + dim
            high_features.append(high_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 按照模型期望的顺序重新排列
        for j in range(5):
            features.append(low_features[j])   # low_j
            features.append(mid_features[j])   # mid_j
            features.append(high_features[j])  # high_j
        
        # 计算验证损失
        loss, out1, out2, out3 = model(features, question, answer, train=True, alpha=config['alpha'])
        
        # 蒸馏损失（如果需要）
        distill_loss = multi_level_distillation_loss(
            out1, out2, out3, features, model.num_layers, model
        ) if lambda_distill > 0 else torch.tensor(0.0, device=device)
        
        total_loss += (loss + lambda_distill * distill_loss).item()
        num_batches += 1
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print("Validation stats:", metric_logger.global_avg())
    return avg_loss

# =============== 评估函数 ===============
@torch.no_grad()
def evaluation(model, data_loader, device, config, model_dims_low, model_dims_mid, model_dims_high):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []
    result_topk = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (low_feature, mid_feature, high_feature, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # 将特征移到设备上
        low_feature = low_feature.to(device, non_blocking=True)
        mid_feature = mid_feature.to(device, non_blocking=True)
        high_feature = high_feature.to(device, non_blocking=True)
        
        # 使用传入的维度列表分割特征，按照模型期望的顺序重新排列
        features = []
        
        # 按照模型期望的顺序：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        
        # 分割low特征
        low_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_low):
            end_idx = start_idx + dim
            low_features.append(low_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 分割mid特征
        mid_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_mid):
            end_idx = start_idx + dim
            mid_features.append(mid_feature[:, start_idx:end_idx])
            start_idx = end_idx
            
        # 分割high特征
        high_features = []
        start_idx = 0
        for j, dim in enumerate(model_dims_high):
            end_idx = start_idx + dim
            high_features.append(high_feature[:, start_idx:end_idx])
            start_idx = end_idx
        
        # 按照模型期望的顺序重新排列：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        for j in range(5):
            features.append(low_features[j])   # low_j
            features.append(mid_features[j])   # mid_j
            features.append(high_features[j])  # high_j

        topk_ids, topk_probs, out1, out2, out3 = model(features, question, answer_list, train=False, k=config['k_test'])

        for qid, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            if hasattr(qid, 'item'):
                qid = int(qid.item())
            else:
                qid = int(qid)
            _, pred = topk_prob.max(dim=0)
            # 记录top-k文本答案
            topk_id_list = topk_id.tolist()
            topk_texts = [data_loader.dataset.answer_list[idx] for idx in topk_id_list]
            result_topk.append({"qid": qid, "answers": topk_texts})
            result.append({"qid": qid, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    
    return result, result_topk

# =============== 主函数 ===============
def main(args, config, dataset_use):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating multi-moe vqa {} datasets'.format(dataset_use))
    
    # 对于multi-moe，直接创建数据集，不需要指定model_name
    if dataset_use == 'multi_moe_pathvqa':
        datasets = create_dataset1(dataset_use, config, 'multi_moe')  # 使用一个标识符表示multi-moe模式
    else:
        # 对于其他数据集，正常调用
        datasets = create_dataset1(dataset_use, config, dataset_use)
    
    # 处理数据集返回值，multi_moe_pathvqa返回3个数据集，其他数据集返回2个
    if dataset_use == 'multi_moe_pathvqa':
        train_dataset, val_dataset, test_dataset = datasets
        datasets = [train_dataset, val_dataset, test_dataset]
        print('train dataset size: ', len(train_dataset))
        print('val dataset size: ', len(val_dataset))
        print('test dataset size: ', len(test_dataset))
    else:
        train_dataset, test_dataset = datasets
        val_dataset = None
        datasets = [train_dataset, test_dataset]
        print('train dataset size: ', len(train_dataset))
        print('test dataset size: ', len(test_dataset))
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if dataset_use == 'multi_moe_pathvqa':
            samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
        else:
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        if dataset_use == 'multi_moe_pathvqa':
            samplers = [None, None, None]
        else:
            samplers = [None, None]

    if dataset_use == 'multi_moe_pathvqa':
        train_loader, val_loader, test_loader = create_loader(datasets, samplers, 
                                                batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                                num_workers=[4, 4, 4], is_trains=[True, False, False], 
                                                collate_fns=[multi_moe_collate_fn, multi_moe_collate_fn, None])
    else:
        train_loader, test_loader = create_loader(datasets, samplers, batch_size=[config['batch_size_train'], config['batch_size_test']],
                                                num_workers=[4, 4], is_trains=[True, False], collate_fns=[multi_moe_collate_fn, None])
        val_loader = None
    
    # 获取特征维度
    feature_dims = datasets[0].get_feature_dims()
    print('Feature dimensions:', feature_dims)
    
    # 自动获取5个教师模型的特征维度
    model_names = ['virchow2', 'uni_v2', 'phikon_v2', 'hoptimus1', 'gigapath']
    base_path = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/'
    
    dim_list_low = []
    dim_list_mid = []
    dim_list_high = []
    
    for model_name in model_names:
        feature_file = os.path.join(base_path, f'train_{model_name}_features.pt')
        feature_data = torch.load(feature_file, map_location='cpu')
        dim_list_low.append(feature_data[0].shape[1])   # low特征维度
        dim_list_mid.append(feature_data[1].shape[1])   # mid特征维度
        dim_list_high.append(feature_data[2].shape[1])  # high特征维度
    
    print(f'自动获取的各模型特征维度:')
    for i, model_name in enumerate(model_names):
        print(f'  {model_name}: Low={dim_list_low[i]}, Mid={dim_list_mid[i]}, High={dim_list_high[i]}')
    tokenizer = BertTokenizer.from_pretrained('/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    model = MUMC_VQA_Multi_MoE(
        text_encoder=args.text_encoder,
        text_decoder=args.text_decoder,
        tokenizer=tokenizer,
        config=config,
        dim_list_low=dim_list_low,
        dim_list_mid=dim_list_mid,
        dim_list_high=dim_list_high,
        d_model=config.get('d_model', 128),
        num_layers=config.get('num_layers', 15)
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #### Optimization ####
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    # 添加学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'], eta_min=config['min_lr'])

    #### Training ####
    print("Start training")
    start_time = time.time()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 初始化最佳验证损失追踪
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config, 
                              model_dims_low=dim_list_low, model_dims_mid=dim_list_mid, model_dims_high=dim_list_high, 
                              lambda_distill=config.get('lambda_distill', 0.01))

            # 验证集评估（仅对multi_moe_pathvqa数据集）
            if dataset_use == 'multi_moe_pathvqa' and val_loader is not None:
                val_loss = validation(model, val_loader, device, config, 
                                    model_dims_low=dim_list_low, model_dims_mid=dim_list_mid, model_dims_high=dim_list_high,
                                    lambda_distill=config.get('lambda_distill', 0.01))
                print(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    print(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch}")
                    
                    if utils.is_main_process():
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'args': args,
                            'config': config,
                            'epoch': epoch,
                            'val_loss': val_loss,
                        }
                        
                        # 获取prefix，与single model代码保持一致
                        if args.checkpoint:
                            prefix = args.checkpoint.split('/')[-1].split('.')[0]
                        else:
                            prefix = 'multi_moe_pathvqa'
                            
                        torch.save(save_obj, os.path.join(args.output_dir, f'{prefix}_best_val.pth'))
                        print(f"Saved best model to {prefix}_best_val.pth")

        if args.evaluate:
            break

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # 保存模型
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'config': config,
                'epoch': epoch,
            }
            
            # 保存checkpoint
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
            utils.save_on_master(save_obj, checkpoint_path)
            
            # 获取prefix，与single model代码保持一致
            if args.checkpoint:
                prefix = args.checkpoint.split('/')[-1].split('.')[0]
            else:
                prefix = 'multi_moe_pathvqa'
            
            # 如果指定了保存路径且epoch大于20，保存额外的模型文件
            if hasattr(args, 'is_save_path') and args.is_save_path and epoch > 20:
                torch.save(save_obj, os.path.join(args.output_dir, '%s_%02d.pth' % (prefix, epoch)))
            
            # 每个epoch都进行评估
            print("Start evaluation")
            test_stats, test_stats_topk = evaluation(model, test_loader, device, config, dim_list_low, dim_list_mid, dim_list_high)
            
            # 保存评估结果到json文件（仿照train_vqa.py）
            json.dump(test_stats, open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))
            json.dump(test_stats_topk, open(os.path.join(args.result_dir, '%s_vqa_result_topk_%s.json' % (prefix, epoch)), 'w'))

        if args.distributed:
            dist.barrier()

    # 训练结束后报告最佳结果
    if dataset_use == 'multi_moe_pathvqa':
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # compute acc
    if args.checkpoint:
        prefix = args.checkpoint.split('/')[-1].split('.')[0]
    else:
        prefix = 'multi_moe_pathvqa'
    res_file_path = '%s/result/%s_vqa_result_<epoch>.json' % (args.output_dir, prefix)
    compute_vqa_acc(answer_list_path=config['pathvqa']['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)
    
    # 额外：计算top-k准确率
    try:
        from vqaEvaluate import compute_vqa_acc_topk
        compute_vqa_acc_topk(answer_list_path=config['pathvqa']['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)
    except Exception as e:
        print('compute_vqa_acc_topk failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--dataset_use', default='multi_moe_pathvqa')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--is_save_path', default=False, type=bool, help='whether to save model path')
    args = parser.parse_args()
    # args.output_dir = '/nas/leiwenhui/tys/PathVQA/MUMC/output/' + args.dataset_use 
    args.output_dir = '/nas/leiwenhui/tys/PathVQA/MUMC/output1/multi_moe' 

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



    main(args, config, args.dataset_use) 