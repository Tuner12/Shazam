import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
import os
import warnings
import hashlib
import json
import re
from transformers import BertTokenizer

# 抑制PIL的警告
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# 设置代理
os.environ['http_proxy'] = 'http://192.168.1.18:7890'
os.environ['https_proxy'] = 'http://192.168.1.18:7890'

# 全局eos标志
try:
    # 使用本地BERT模型路径
    bert_path = '/home/leiwenhui/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
    _tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    print(f"tokenizer: {_tokenizer.eos_token}")
    eos = _tokenizer.eos_token or '[SEP]'
except Exception:
    eos = '[SEP]'

# 预处理函数
def pre_question(question, max_ques_words=25):
    """预处理问题文本"""
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question

def pre_answer(answer):
    """预处理答案文本"""
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    return answer

class PathVQADataset(Dataset):
    """
    简洁的Hugging Face PathVQA数据集类
    支持两种模式：
    1. images_only=False: 返回完整的图-问-答三元组
    2. images_only=True: 仅返回去重后的图片
    """
    def __init__(self, 
                 split='train',
                 image_size=224,
                 images_only=False,
                 cache_dir=None,
                 max_ques_words=25,
                 use_preprocessing=True):
        self.split = split
        self.image_size = image_size
        self.images_only = images_only
        self.max_ques_words = max_ques_words
        self.use_preprocessing = use_preprocessing
        
        if cache_dir is None:
            cache_dir = "/nas/leiwenhui/tys/PathVQA/Dataset/cache"
        try:
            self.dataset = load_dataset("flaviagiammarino/path-vqa", 
                                    split=split,cache_dir=cache_dir)
        except Exception as e:
            print(f"数据集加载失败: {e}")
            raise
      
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        if self.images_only:
            self._build_unique_images()

    def _build_unique_images(self):
        # 构建去重图片列表，使用内容哈希，并记录图片与问答对的对应关系
        unique = {}
        self.image_hash_to_indices = {}
        for idx, item in enumerate(self.dataset):
            key = self._get_image_key(item['image'])
            if key not in unique:
                unique[key] = item['image']
                self.image_hash_to_indices[key] = []
            self.image_hash_to_indices[key].append(idx)
        self.unique_images = list(unique.values())
        self.unique_image_hashes = list(unique.keys())  # 保留哈希顺序与unique_images一致

    def get_qa_indices_for_image(self, image_idx):
        # 返回该图片对应的所有问答对在原始数据集中的索引
        key = self.unique_image_hashes[image_idx]
        return self.image_hash_to_indices[key]

    def _get_image_key(self, image):
        # 用图片内容的md5哈希作为唯一标识
        img = image.convert('RGB')
        img_bytes = img.tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    def _load_image(self, image):
        try:
            if isinstance(image, str):
                with Image.open(image) as img:
                    image = img.convert('RGB')
            elif hasattr(image, 'convert'):
                image = image.convert('RGB')
            else:
                image = Image.fromarray(image)
            return self.transform(image)
        except Exception as e:
            print(f"加载图像失败: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    def __len__(self):
        if self.images_only:
            return len(self.unique_images)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.images_only:
            image = self._load_image(self.unique_images[idx])
            return {
                'image': image,
                'image_idx': idx
            }
        else:
            item = self.dataset[idx]
            image = self._load_image(item['image'])
            
            # 预处理问题和答案
            if self.use_preprocessing:
                question = pre_question(item['question'], self.max_ques_words)
                answer = pre_answer(item['answer'])
                answer = answer + eos
            else:
                question = item['question']
                answer = item['answer']
            
            question_words = question.lower().split()
            
            # 为答案添加结束标志，与推理时的答案列表保持一致
       
            qa_type = 'close' if answer in ['yes', 'no'] else 'open'
            
            return {
                'qid': item.get('id', idx),
                'image': image,
                'question': question,
                'question_words': question_words,
                'answer': answer,  # 返回带</s>的答案
                'qa_type': qa_type
            }

def create_datasets(image_size=224, images_only=False, cache_dir=None):
    train_dataset = PathVQADataset(
        split='train',
        image_size=image_size,
        images_only=images_only,
        cache_dir=cache_dir
    )
    val_dataset = PathVQADataset(
        split='validation',
        image_size=image_size,
        images_only=images_only,
        cache_dir=cache_dir
    )
    test_dataset = PathVQADataset(
        split='test',
        image_size=image_size,
        images_only=images_only,
        cache_dir=cache_dir
    )
    return train_dataset, val_dataset, test_dataset

# 新增：保存三元组样本到唯一图片特征的映射json

def save_image_hash_mapping_to_json(split='train', image_size=224, cache_dir=None, json_path='image_hash_mapping.json'):
    """
    遍历三元组数据集，将每个样本的索引（或qid）与其图片哈希、唯一图片索引建立映射，并保存为json文件。
    """
    dataset = PathVQADataset(split=split, image_size=image_size, images_only=False)
    img_dedup = PathVQADataset(split=split, image_size=image_size, images_only=True)
    mapping = {}
    for i in range(len(dataset)):
        # 用原始PIL Image做哈希
        pil_image = dataset.dataset[i]['image']
        image_hash = img_dedup._get_image_key(pil_image)
        image_idx = img_dedup.unique_image_hashes.index(image_hash)
        qid = dataset.dataset[i].get('id', i)
        mapping[qid] = {
            'sample_idx': i,
            'image_hash': image_hash,
            'unique_image_idx': image_idx
        }
    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"保存三元组到唯一图片特征的映射到: {json_path}")

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = create_datasets(image_size=224)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    # for i in range(100):
    #     sample = train_dataset[i]
    #     print(f"问题: {sample['question']}")
    #     print(f"答案: {sample['answer']}")
    #     print(f"图像形状: {sample['image'].shape}")
    #     print(f"问题类型: {sample['qa_type']}")

    # 测试仅图片模式
    # train_images, val_images, test_images = create_datasets(image_size=224, images_only=True)
    # print(f"仅图片模式-训练集图片数: {len(train_images)}")
    # print(f"仅图片模式-验证集图片数: {len(val_images)}")
    # print(f"仅图片模式-测试集图片数: {len(test_images)}")
    # img_sample = train_images[0]
    # print(f"图片张量形状: {img_sample['image'].shape}")

    # # 新增：保存三元组到唯一图片特征的映射json
    # save_image_hash_mapping_to_json(split='train', image_size=224, cache_dir=None, json_path='train_image_hash_mapping.json')
    # save_image_hash_mapping_to_json(split='validation', image_size=224, cache_dir=None, json_path='val_image_hash_mapping.json')
    # save_image_hash_mapping_to_json(split='test', image_size=224, cache_dir=None, json_path='test_image_hash_mapping.json') 

    # 生成train唯一答案列表（使用预处理）
    # print("\n=== 生成train唯一答案列表 ===")
    # train_answers = set()
    # for i in range(len(train_dataset)):
    #     # 使用原始数据，然后预处理
    #     raw_ans = train_dataset.dataset[i]['answer']
    #     ans = pre_answer(raw_ans)
    #     ans_norm = ans.strip().lower()
    #     if ans_norm:
    #         train_answers.add(ans_norm)
    # train_answer_list = [ans + eos for ans in sorted(list(train_answers))]
    # with open("Dataset/train_answer_list.json", "w", encoding="utf-8") as f:
    #     json.dump(train_answer_list, f, indent=2, ensure_ascii=False)
    # print(f"train唯一答案数量: {len(train_answer_list)}，已保存到 Dataset/train_answer_list.json")
    # print("train前10个答案示例：", train_answer_list[:10])

    # print("\n=== 生成val唯一答案列表 ===")
    # val_answers = set()
    # for i in range(len(val_dataset)):
    #     # 使用原始数据，然后预处理
    #     raw_ans = val_dataset.dataset[i]['answer']
    #     ans = pre_answer(raw_ans)
    #     ans_norm = ans.strip().lower()
    #     if ans_norm:
    #         val_answers.add(ans_norm)
    # val_answer_list = [ans + eos for ans in sorted(list(val_answers))]
    # with open("Dataset/val_answer_list.json", "w", encoding="utf-8") as f:
    #     json.dump(val_answer_list, f, indent=2, ensure_ascii=False)
    # print(f"val唯一答案数量: {len(val_answer_list)}，已保存到 Dataset/val_answer_list.json")
    # print("val前10个答案示例：", val_answer_list[:10])

    # # 生成test唯一答案列表
    # print("\n=== 生成test唯一答案列表 ===")
    # test_answers = set()
    # for i in range(len(test_dataset)):
    #     # 使用原始数据，然后预处理
    #     raw_ans = test_dataset.dataset[i]['answer']
    #     ans = pre_answer(raw_ans)
    #     ans_norm = ans.strip().lower()
    #     if ans_norm:
    #         test_answers.add(ans_norm)
    # test_answer_list = [ans + eos for ans in sorted(list(test_answers))]
    # with open("Dataset/test_answer_list.json", "w", encoding="utf-8") as f:
    #     json.dump(test_answer_list, f, indent=2, ensure_ascii=False)
    # print(f"test唯一答案数量: {len(test_answer_list)}，已保存到 Dataset/test_answer_list.json")
    # print("test前10个答案示例：", test_answer_list[:10]) 