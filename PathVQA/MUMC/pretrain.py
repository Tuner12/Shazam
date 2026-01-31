import argparse
import os
import sys
import ruamel_yaml as yaml
import time
import datetime
import json
from pathlib import Path


import torch
import torch.distributed as dist

from models.model_pretrain import MUMC_Pretrain
from models.model import load_checkpoint

import utils
from dataset import create_dataset, create_sampler, create_loader

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch == 0:   # 预热学习率
            utils.warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha'] * min(1, (epoch * len(data_loader) + i) / (2 * len(data_loader)))

        loss_mlm, loss_ita, loss_itm = model(image, text, alpha=alpha)
        loss = loss_mlm + loss_ita + loss_itm

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]
    print('number of training samples: %d' % len(datasets[0]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(datasets, samplers, batch_size=[config['batch_size']], num_workers=[4],
                                is_trains=[True], collate_fns=[None])[0]

    ''' Model '''
    print("Creating model")
    model = MUMC_Pretrain(config=config).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    start_epoch = 0
    if args.checkpoint:
        model, msg = load_checkpoint(model, args.checkpoint)
        print('missing_keys = {}, unexpected_keys = {}'.format(msg.missing_keys, msg.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, config['max_epoch']):
        utils.step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        train_stats = train(model, data_loader, optimizer, epoch, device, config)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            save_obj = {
                'model': model_without_ddp.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'config': config,
                # 'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'med_pretrain_%02d.pth' % epoch))

            with open(os.path.join(args.output_dir, "loss.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        '''distributed Training'''
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='/mnt/sda/lpf/weights/pre_training/albef/ALBEF.pth')  #
    parser.add_argument('--output_dir', default='/mnt/sda/lpf/weights/output/V2/pretrain')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    args.output_dir = os.path.join(args.output_dir, datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H-%M'))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("config: ", config)
    print("args: ", args)

    main(args, config)
