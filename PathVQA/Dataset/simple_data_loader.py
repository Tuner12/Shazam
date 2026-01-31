import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
sys.path.append('/nas/leiwenhui/tys/PathVQA/Dataset')
from pathvqa_dataset import PathVQADataset, create_datasets

def collate_fn(batch):
    """
    自定义的collate函数，用于处理不同长度的序列
    
    Args:
        batch: 批次数据列表
        
    Returns:
        collated_batch: 整理后的批次数据
    """
    # 分离不同类型的数据
    qids = [item['qid'] for item in batch]
    images = [item['image'] for item in batch]
    questions = [item['question'] for item in batch]
    question_words = [item['question_words'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # 堆叠图像张量
    images = torch.stack(images, dim=0)
    
    # 构建返回字典
    collated_batch = {
        'qids': qids,
        'images': images,
        'questions': questions,
        'question_words': question_words,
        'answers': answers
    }
    
    return collated_batch


def collate_fn_images(batch):
    """
    用于图像数据集的collate函数
    
    Args:
        batch: 批次数据列表
        
    Returns:
        collated_batch: 整理后的批次数据
    """
    images = [item['image'] for item in batch]
    image_indices = [item['image_idx'] for item in batch]
    
    # 堆叠图像张量
    images = torch.stack(images, dim=0)
    
    # 转换图像索引为张量
    image_indices = torch.tensor(image_indices, dtype=torch.long)
    
    return {
        'images': images,
        'image_indices': image_indices
    }


def create_data_loaders(image_size=224, 
                       batch_size=32, 
                       num_workers=4,
                       images_only=False,
                       cache_dir=None):
    """
    创建数据加载器
    
    Args:
        image_size: 图像尺寸
        batch_size: 批处理大小
        num_workers: 数据加载的工作进程数
        images_only: 是否仅返回图片
        cache_dir: 缓存目录路径
        
    Returns:
        train_loader, val_loader, test_loader: 三个数据加载器
        train_dataset, val_dataset, test_dataset: 三个数据集对象
    """
    
    print("创建数据集...")
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(
        image_size=image_size,
        images_only=images_only,
        cache_dir=cache_dir
    )
    
    # 选择collate函数
    if images_only:
        collate_func = collate_fn_images
    else:
        collate_func = collate_fn
    
    print("创建数据加载器...")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=True
    )
    
    print(f"数据加载器创建完成!")
    print(f"训练集: {len(train_loader)} 批次")
    print(f"验证集: {len(val_loader)} 批次")
    print(f"测试集: {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def test_data_loader():
    """测试数据加载器"""
    print("测试数据加载器...")
    
    # 测试完整数据加载器
    print("\n=== 测试完整数据加载器 ===")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        image_size=224,
        batch_size=4,
        num_workers=0,
        images_only=False
    )
    
    # 测试一个批次
    for batch in train_loader:
        print(f"批次键: {batch.keys()}")
        print(f"图像形状: {batch['images'].shape}")
        print(f"问题数量: {len(batch['questions'])}")
        print(f"答案数量: {len(batch['answers'])}")
        break
    
    # 测试图像数据加载器
    print("\n=== 测试图像数据加载器 ===")
    train_loader_img, val_loader_img, test_loader_img, train_dataset_img, val_dataset_img, test_dataset_img = create_data_loaders(
        image_size=224,
        batch_size=4,
        num_workers=0,
        images_only=True
    )
    
    # 测试一个批次
    for batch in train_loader_img:
        print(f"批次键: {batch.keys()}")
        print(f"图像形状: {batch['images'].shape}")
        print(f"图像索引形状: {batch['image_indices'].shape}")
        break
    
    print("\n数据加载器测试完成!")


if __name__ == "__main__":
    test_data_loader() 