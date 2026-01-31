import os
import json
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .utils import pre_question, pre_answer

class multi_feature_dataset(Dataset):
    def __init__(self, feature_file, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.max_ques_words = max_ques_words
        self.eos = eos

        # 定义5个模型名称
        model_names = ['virchow2', 'uni_v2', 'phikon_v2', 'hoptimus1', 'gigapath']
        base_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'
        
        # 加载5个模型的特征
        print(f"Loading multi-model features for {split} split...")
        all_low_features = []
        all_mid_features = []
        all_high_features = []
        
        for model_name in model_names:
            feature_file_path = os.path.join(base_path, f'{split}_{model_name}_features.pt')
            print(f"Loading features from {feature_file_path}")
            feature_data = torch.load(feature_file_path, map_location='cpu')
            
            all_low_features.append(feature_data[0])   # low features
            all_mid_features.append(feature_data[1])   # mid features  
            all_high_features.append(feature_data[2])  # high features
            
            # 使用第一个模型的问题和答案/ID
            if model_name == model_names[0]:
                self.questions = feature_data[3]      # 问题列表
                if split == 'test':
                    self.question_ids = feature_data[4]  # 测试集有问题ID
                else:
                    self.answers = feature_data[4]  # 训练集有答案列表
        
        # 合并5个模型的特征
        self.low_features = torch.cat(all_low_features, dim=1)   # 在特征维度上拼接
        self.mid_features = torch.cat(all_mid_features, dim=1)   # 在特征维度上拼接
        self.high_features = torch.cat(all_high_features, dim=1) # 在特征维度上拼接
        
        print(f"Loaded {len(self.low_features)} multi-model features")
        print(f"Combined feature dimensions - Low: {self.low_features.shape[1]}, Mid: {self.mid_features.shape[1]}, High: {self.high_features.shape[1]}")
        print(f"Number of questions: {len(self.questions)}")

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            if answer_list:
                self.answer_list = json.load(open(answer_list, 'r'))

    def __len__(self):
        return len(self.low_features)

    def __getitem__(self, index):
        # 返回三层特征
        low_feature = self.low_features[index]
        mid_feature = self.mid_features[index]
        high_feature = self.high_features[index]
        question = self.questions[index]

        if self.split == 'test':
            question_id = self.question_ids[index]
            return low_feature, mid_feature, high_feature, question, question_id

        elif self.split in ['train', 'val']:
            # 训练集和验证集都返回答案
            answers = self.answers[index]
            if isinstance(answers, tuple):
                answers = list(answers)
            answers = [pre_answer(answers)]
            answers = [answer + self.eos for answer in answers]
            return low_feature, mid_feature, high_feature, question, answers

    def get_feature_dims(self):
        """返回三层特征的维度"""
        return {
            'low': self.low_features.shape[1] if hasattr(self.low_features, 'shape') else None,
            'mid': self.mid_features.shape[1] if hasattr(self.mid_features, 'shape') else None,
            'high': self.high_features.shape[1] if hasattr(self.high_features, 'shape') else None
        }
    
    @staticmethod
    def get_model_feature_dims():
        """静态方法：获取5个模型的特征维度"""
        model_names = ['virchow2', 'uni_v2', 'phikon_v2', 'hoptimus1', 'gigapath']
        base_path = '/data2/tanyusheng/Code/PathVQA/features4pathvqa/images/'
        
        dim_list_low = []
        dim_list_mid = []
        dim_list_high = []
        
        for model_name in model_names:
            feature_file = os.path.join(base_path, f'train_{model_name}_features.pt')
            feature_data = torch.load(feature_file, map_location='cpu')
            dim_list_low.append(feature_data[0].shape[1])   # low特征维度
            dim_list_mid.append(feature_data[1].shape[1])   # mid特征维度
            dim_list_high.append(feature_data[2].shape[1])  # high特征维度
        
        return dim_list_low, dim_list_mid, dim_list_high


if __name__ == '__main__':
    # 测试代码
    train_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/train_virchow2_features.pt'
    test_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/test_virchow2_features.pt'
    answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    
    # 测试训练集
    print("=== 训练集 ===")
    dataset = multi_feature_dataset(train_feature_file, split='train')
    d1 = dataset[0]
    print(f"  Low feature shape: {d1[0].shape}")
    print(f"  Mid feature shape: {d1[1].shape}")
    print(f"  High feature shape: {d1[2].shape}")
    print(f"  Question: {d1[3]}")
    print(f"  Answer: {d1[4]}")
    print(f"  Feature dimensions: {dataset.get_feature_dims()}")
    print(f"  Dataset length: {len(dataset)}")

    # 测试测试集
    print("\n=== 测试测试集 ===")
    dataset = multi_feature_dataset(test_feature_file, answer_list=answer_list, split='test')
    d2 = dataset[0]
    print(f"  Low feature shape: {d2[0].shape}")
    print(f"  Mid feature shape: {d2[1].shape}")
    print(f"  High feature shape: {d2[2].shape}")
    print(f"  Question: {d2[3]}")
    print(f"  Question ID: {d2[4]}")
    print(f"  Dataset length: {len(dataset)}") 