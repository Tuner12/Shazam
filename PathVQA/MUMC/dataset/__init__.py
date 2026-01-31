import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from .vqa_dataset import vqa_dataset
from .vqa_feature_dataset import vqa_feature_dataset
from .multi_feature_dataset import multi_feature_dataset
from .pretrain_dataset import pretrain_dataset
from .randaugment import RandomAugment


def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # medical image-text pretraining datasets
    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform, image_root=config['image_root'])
        return dataset

    # vqa_rad
    elif dataset == 'rad':
        train_dataset = vqa_dataset(config['rad']['train_file'], train_transform, config['rad']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['rad']['test_file'], test_transform, config['rad']['vqa_root'], split='test',
                                   answer_list=config['rad']['answer_list'])
        return train_dataset, test_dataset

    # pathvqa
    elif dataset == 'pathvqa':
        train_dataset = vqa_dataset(config['pathvqa']['train_file'], train_transform, config['pathvqa']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['pathvqa']['test_file'], test_transform, config['pathvqa']['vqa_root'], split='test',
                                   answer_list=config['pathvqa']['answer_list'])
        return train_dataset, test_dataset
    # slake
    elif dataset == 'slake':
        train_dataset = vqa_dataset(config['slake']['train_file'], train_transform, config['slake']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['slake']['test_file'], test_transform, config['slake']['vqa_root'], split='test',
                                   answer_list=config['slake']['answer_list'])
        return train_dataset, test_dataset
    elif dataset == 'med2019':
        train_dataset = vqa_dataset(config['med2019']['train_file'], train_transform, config['med2019']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['med2019']['test_file'], test_transform, config['med2019']['vqa_root'], split='test',
                                   answer_list=config['med2019']['answer_list'])
        return train_dataset, test_dataset


def create_dataset1(dataset, config,model_name):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    feature_train_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'+'train_'+model_name+'_features.pt'
    feature_test_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'+'test_'+model_name+'_features.pt'
    # medical image-text pretraining datasets
    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform, image_root=config['image_root'])
        return dataset

    # vqa_rad
    elif dataset == 'rad':
        train_dataset = vqa_dataset(config['rad']['train_file'], train_transform, config['rad']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['rad']['test_file'], test_transform, config['rad']['vqa_root'], split='test',
                                   answer_list=config['rad']['answer_list'])
        return train_dataset, test_dataset

    # pathvqa
    elif dataset == 'pathvqa':
        feature_train_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'+'train_'+model_name+'_features.pt'
        feature_val_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'+'val_'+model_name+'_features.pt'
        feature_test_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'+'test_'+model_name+'_features.pt'
        
        # 创建训练集、验证集和测试集
        train_dataset = vqa_feature_dataset(feature_file=feature_train_path, split='train')
        val_dataset = vqa_feature_dataset(feature_file=feature_val_path, split='val')
        test_dataset = vqa_feature_dataset(feature_file = feature_test_path, split='test',answer_list=config['pathvqa']['answer_list'])
        return train_dataset, val_dataset, test_dataset
    
    # multi-moe pathvqa
    elif dataset == 'multi_moe_pathvqa':
        # 对于multi-moe，直接创建multi_feature_dataset，它会自动加载5个模型的特征
        train_dataset = multi_feature_dataset(feature_file='', split='train')  # feature_file参数在multi_feature_dataset中被忽略
        val_dataset = multi_feature_dataset(feature_file='', split='val')
        test_dataset = multi_feature_dataset(feature_file='', split='test', answer_list=config['pathvqa']['answer_list'])
        return train_dataset, val_dataset, test_dataset
    # slake
    elif dataset == 'slake':
        train_dataset = vqa_dataset(config['slake']['train_file'], train_transform, config['slake']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['slake']['test_file'], test_transform, config['slake']['vqa_root'], split='test',
                                   answer_list=config['slake']['answer_list'])
        return train_dataset, test_dataset
    elif dataset == 'med2019':
        train_dataset = vqa_dataset(config['med2019']['train_file'], train_transform, config['med2019']['vqa_root'], split='train')
        test_dataset = vqa_dataset(config['med2019']['test_file'], test_transform, config['med2019']['vqa_root'], split='test',
                                   answer_list=config['med2019']['answer_list'])
        return train_dataset, test_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list = [], [], []
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list += answer

    return torch.stack(image_list, dim=0), question_list, answer_list

def multi_moe_collate_fn(batch):
    low_list, mid_list, high_list, question_list, answer_list = [], [], [], [], []
    for low, mid, high, question, answer in batch:
        low_list.append(low)
        mid_list.append(mid)
        high_list.append(high)
        question_list.append(question)
        answer_list += answer

    return (torch.stack(low_list, dim=0), torch.stack(mid_list, dim=0), torch.stack(high_list, dim=0), 
            question_list, answer_list)


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, pin_memory=None):
    loaders = []
    # 自动判定pin_memory：若未显式指定，则根据CUDA可用性启用
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

if __name__ == '__main__':
    config_path = 'configs/VQA.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    dataset = create_dataset('pathvqa', config)
    print(dataset)