import os
import json
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .utils import pre_question, pre_answer

class vqa_feature_dataset(Dataset):
    def __init__(self, feature_file, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.max_ques_words = max_ques_words
        self.eos = eos

        # 加载预提取的图像特征
        print(f"Loading features from {feature_file}")
        feature_data = torch.load(feature_file, map_location='cpu')
        
        # 特征文件结构: (low_features, mid_features, high_features, questions, question_ids/answers)
        self.low_features = feature_data[0]   # 低级特征
        self.mid_features = feature_data[1]   # 中级特征  
        self.high_features = feature_data[2]  # 高级特征
        self.questions = feature_data[3]      # 问题列表
        
        # 使用高级特征作为主要特征
        self.features = self.high_features
        
        # print(f"Loaded {len(self.features)} features for {split}")
        # print(f"Feature dimensions - Low: {self.low_features.shape[1]}, Mid: {self.mid_features.shape[1]}, High: {self.high_features.shape[1]}")
        # print(f"Number of questions: {len(self.questions)}")

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.question_ids = feature_data[4]  # 测试集有问题ID
            if answer_list:
                self.answer_list = json.load(open(answer_list, 'r'))
        else:
            # 训练集和验证集都有答案
            self.answers = feature_data[4]  # 答案列表

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # 直接按索引访问特征
        image = self.features[index]
        question = self.questions[index]

        if self.split == 'test':
            question_id = self.question_ids[index]
            return image, question, question_id

        elif self.split in ['train', 'val']:
            # 训练集和验证集都有答案
            answers = self.answers[index]
            if isinstance(answers, tuple):
                answers = list(answers)
            answers = [pre_answer(answers)]
            answers = [answer + self.eos for answer in answers]
            # answer = answers[0]
            return image, question, answers

    def get_feature_dim(self):
        """返回特征维度"""
        return self.features.shape[1] if hasattr(self.features, 'shape') else None


if __name__ == '__main__':
    # 测试代码
    train_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/train_virchow2_features.pt'
    test_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/test_virchow2_features.pt'
    answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    
    # 测试训练集
    print("=== 训练集 ===")
    dataset = vqa_feature_dataset(train_feature_file, split='train')
    d1 = dataset[0]
    print(f"  Image feature shape: {d1[0].shape}")
    print(f"  Question: {d1[1]}")
    print(f"  Answer: {d1[2]}")
    print(f"  Feature dimension: {dataset.get_feature_dim()}")
    print(f"  Dataset length: {len(dataset)}")

    # 测试测试集
    print("\n=== 测试测试集 ===")
    dataset = vqa_feature_dataset(test_feature_file, answer_list=answer_list, split='test')
    d2 = dataset[0]
    print(f"  Image feature shape: {d2[0].shape}")
    print(f"  Question: {d2[1]}")
    print(f"  Question ID: {d2[2]}")
    print(f"  Dataset length: {len(dataset)}") 