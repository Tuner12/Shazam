import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class VQAMedDataset(Dataset):
    def __init__(self, root_dir, qa_file, image_dir, transform=None, max_length=512):
        """
        VQAMed数据集类
        
        Args:
            root_dir: 数据集根目录
            qa_file: QA对文件名
            image_dir: 图片文件夹名
            transform: 图像预处理
            max_length: 文本最大长度
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.max_length = max_length
        
        # 默认的图像预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.samples = []
        self._load_qa_pairs(qa_file)
        
    def _load_qa_pairs(self, qa_file):
        """加载QA对数据"""
        qa_path = os.path.join(self.root_dir, qa_file)
        
        if not os.path.exists(qa_path):
            print(f"Warning: QA file {qa_path} not found!")
            return
            
        with open(qa_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('|')
                if len(parts) == 3:  # 训练集和验证集: image_id|question|answer
                    image_id, question, answer = parts
                    self.samples.append({
                        'image_id': image_id,
                        'question': question,
                        'answer': answer,
                        'has_answer': True
                    })
                elif len(parts) == 4:  # 测试集: image_id|category|question|answer
                    image_id, category, question, answer = parts
                    self.samples.append({
                        'image_id': image_id,
                        'question': question,
                        'answer': answer,
                        'has_answer': True
                    })
                elif len(parts) == 2:  # 原始测试集: image_id|question
                    image_id, question = parts
                    self.samples.append({
                        'image_id': image_id,
                        'question': question,
                        'answer': '',  # 测试集没有答案
                        'has_answer': False
                    })
                else:
                    print(f"Warning: Invalid line format at line {line_num}: {line}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        question = sample['question']
        answer = sample['answer']
        
        # 加载图像
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'image_id': image_id,
            'has_answer': sample['has_answer']
        }
    
    def get_sample_info(self):
        """获取数据集基本信息"""
        total_samples = len(self.samples)
        samples_with_answer = sum(1 for s in self.samples if s['has_answer'])
        samples_without_answer = total_samples - samples_with_answer
        
        # 统计答案分布（仅对有答案的样本）
        answer_counts = {}
        for sample in self.samples:
            if sample['has_answer']:
                answer = sample['answer'].lower().strip()
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # 统计唯一图片数量
        unique_images = set(sample['image_id'] for sample in self.samples)
        
        return {
            'total_samples': total_samples,
            'samples_with_answer': samples_with_answer,
            'samples_without_answer': samples_without_answer,
            'unique_answers': len(answer_counts),
            'unique_images': len(unique_images),
            'top_answers': sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }


def create_datasets(data_root):
    """创建训练集、验证集和测试集"""
    
    # 训练集
    train_dataset = VQAMedDataset(
        root_dir=os.path.join(data_root, 'ImageClef-2019-VQA-Med-Training'),
        qa_file='All_QA_Pairs_train.txt',
        image_dir='Train_images'
    )
    
    # 验证集
    val_dataset = VQAMedDataset(
        root_dir=os.path.join(data_root, 'ImageClef-2019-VQA-Med-Validation'),
        qa_file='All_QA_Pairs_val.txt',
        image_dir='Val_images'
    )
    
    # 测试集
    test_dataset = VQAMedDataset(
        root_dir=os.path.join(data_root, 'VQAMed2019Test'),
        qa_file='VQAMed2019_Test_Questions_w_Ref_Answers.txt',
        image_dir='VQAMed2019_Test_Images'
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 设置数据根目录 - 修复路径问题
    data_root = "/nas/leiwenhui/tys/PathVQA/PathVQA"
    
    print("=== VQAMed Dataset 测试 ===")
    
    try:
        # 创建数据集
        print("正在创建数据集...")
        train_dataset, val_dataset, test_dataset = create_datasets(data_root)
        
        print(f"\n数据集大小:")
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        
        # 打印图片数量
        print(f"\n图片数量:")
        train_info = train_dataset.get_sample_info()
        val_info = val_dataset.get_sample_info()
        test_info = test_dataset.get_sample_info()
        print(f"训练集: {train_info['unique_images']} 张图片")
        print(f"验证集: {val_info['unique_images']} 张图片")
        print(f"测试集: {test_info['unique_images']} 张图片")
        print(f"总计: {train_info['unique_images'] + val_info['unique_images'] + test_info['unique_images']} 张图片")
        
        # 测试训练集
        print(f"\n=== 训练集测试 ===")
        train_info = train_dataset.get_sample_info()
        print(f"总样本数: {train_info['total_samples']}")
        print(f"有答案样本: {train_info['samples_with_answer']}")
        print(f"无答案样本: {train_info['samples_without_answer']}")
        print(f"唯一答案数: {train_info['unique_answers']}")
        print(f"唯一图片数: {train_info['unique_images']}")
        print(f"前10个最常见答案:")
        for answer, count in train_info['top_answers']:
            print(f"  {answer}: {count}")
        
        # 随机测试几个样本
        print(f"\n=== 随机样本测试 ===")
        for i, dataset_name in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
            print(f"\n{i.upper()} 集样本:")
            if len(dataset_name) == 0:
                print(f"  {i.upper()} 集为空，跳过测试")
                continue
            for j in range(min(3, len(dataset_name))):  # 确保不超过数据集大小
                idx = random.randint(0, len(dataset_name) - 1)
                sample = dataset_name[idx]
                print(f"  样本 {j+1}:")
                print(f"    Image ID: {sample['image_id']}")
                print(f"    Question: {sample['question']}")
                if sample['has_answer']:
                    print(f"    Answer: {sample['answer']}")
                else:
                    print(f"    Answer: (测试集无答案)")
                print(f"    Image shape: {sample['image'].shape}")
        
        print(f"\n=== 数据集测试完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 