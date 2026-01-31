#!/usr/bin/env python3
"""
测试Multi-MoE VQA模型的实现
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_vqa_multi_moe import MUMC_VQA_Multi_MoE, multi_level_distillation_loss
from models.tokenization_bert import BertTokenizer
from dataset.multi_feature_dataset import multi_feature_dataset
import ruamel.yaml as yaml

def test_model():
    print("=== 测试Multi-MoE VQA模型 ===")
    
    # 加载配置
    try:
        from ruamel.yaml import YAML
        yaml_loader = YAML(typ='rt')
        with open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
            config = yaml_loader.load(f)
    except ImportError:
        config = yaml.load(open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r'), Loader=yaml.Loader)

    # 加载tokenizer
    print("加载BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model')
    
    # 加载5个模型的特征文件
    print("加载5个模型的特征文件...")
    model_names = ['virchow2', 'uni_v2', 'phikon_v2', 'hoptimus1', 'gigapath']
    base_path = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/'
    
    # 加载训练集特征
    train_features = []
    for model_name in model_names:
        feature_file = os.path.join(base_path, f'train_{model_name}_features.pt')
        print(f"加载特征文件: {feature_file}")
        feature_data = torch.load(feature_file, map_location='cpu')
        train_features.append(feature_data)
    
    # 获取特征维度
    dim_list_low = []
    dim_list_mid = []
    dim_list_high = []
    
    for feature_data in train_features:
        low_features = feature_data[0]  # low features
        mid_features = feature_data[1]  # mid features
        high_features = feature_data[2] # high features
        
        dim_list_low.append(low_features.shape[1])
        dim_list_mid.append(mid_features.shape[1])
        dim_list_high.append(high_features.shape[1])
    
    print(f"各模型特征维度:")
    for i, model_name in enumerate(model_names):
        print(f"  {model_name}: Low={dim_list_low[i]}, Mid={dim_list_mid[i]}, High={dim_list_high[i]}")
    
    # 创建模型
    print("创建Multi-MoE VQA模型...")
    model = MUMC_VQA_Multi_MoE(
        text_encoder='bert-base-uncased',
        text_decoder='bert-base-uncased', 
        tokenizer=tokenizer,
        config=config,
        dim_list_low=dim_list_low,
        dim_list_mid=dim_list_mid,
        dim_list_high=dim_list_high,
        d_model=128,
        num_layers=12
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 测试前向传播
    print("测试前向传播...")
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 使用真实的特征数据
    batch = []
    for i, feature_data in enumerate(train_features):
        # 取前batch_size个样本
        low_feat = feature_data[0][:batch_size].to(device)   # low_i
        mid_feat = feature_data[1][:batch_size].to(device)   # mid_i
        high_feat = feature_data[2][:batch_size].to(device)  # high_i
        
        batch.append(low_feat)
        batch.append(mid_feat)
        batch.append(high_feat)
    
    # 获取问题文本
    questions = train_features[0][3][:batch_size]  # 使用第一个模型的问题
    question = [q for q in questions]
    
    # 获取答案
    answers = train_features[0][4][:batch_size]  # 使用第一个模型的答案
    answer = [a[0] if isinstance(a, (list, tuple)) else a for a in answers]
    
    print(f"特征形状:")
    for i, model_name in enumerate(model_names):
        print(f"  {model_name}: Low={batch[i*3].shape}, Mid={batch[i*3+1].shape}, High={batch[i*3+2].shape}")
    
    # 训练模式
    print("测试训练模式...")
    model.train()
    loss, out1, out2, out3 = model(batch, question, answer, train=True)
    print(f"训练损失: {loss.item():.4f}")
    
    # 计算蒸馏损失
    print("计算蒸馏损失...")
    lambda_distill = 0.01
    distill_loss = multi_level_distillation_loss(
        out1, out2, out3, batch, 12, model
    )
    print(f"蒸馏损失: {distill_loss.item():.4f}")
    
    # 总损失
    total_loss = loss + lambda_distill * distill_loss
    print(f"总损失: {total_loss.item():.4f}")
    
    # 测试模式
    print("测试推理模式...")
    model.eval()
    answer_list = ["blue", "red", "green", "yellow", "purple"]
    answer_list = [ans + '[SEP]' for ans in answer_list]
    
    with torch.no_grad():
        topk_ids, topk_probs, out1, out2, out3 = model(batch, question, answer_list, train=False, k=3)
        print(f"Top-k IDs shape: {topk_ids.shape}")
        print(topk_ids)
        print(f"Top-k probabilities shape: {topk_probs.shape}")
        print(topk_probs)
    
    print("=== 模型测试完成 ===")

def test_dataset():
    print("\n=== 测试Multi-Feature Dataset ===")
    
    # 测试数据集路径
    train_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/train_virchow2_features.pt'
    test_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/test_virchow2_features.pt'
    answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    
    try:
        # 测试训练集
        print("测试训练集...")
        dataset = multi_feature_dataset(train_feature_file, split='train')
        print(f"训练集大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"样本结构: {len(sample)} 个元素")
        print(f"前15个是特征，第16个是question，第17个是answer/question_id")
        print(f"Question: {sample[15]}")
        print(f"Answer: {sample[16]}")
        
        # 测试测试集
        print("\n测试测试集...")
        dataset = multi_feature_dataset(test_feature_file, answer_list=answer_list, split='test')
        print(f"测试集大小: {len(dataset)}")
        
        sample = dataset[0]
        print(f"样本结构: {len(sample)} 个元素")
        print(f"前15个是特征，第16个是question，第17个是question_id")
        print(f"Question: {sample[15]}")
        print(f"Question ID: {sample[16]}")
        
        # 获取特征维度
        feature_dims = dataset.get_feature_dims()
        print(f"特征维度: {feature_dims}")
        
    except Exception as e:
        print(f"数据集测试失败: {e}")
    
    print("=== 数据集测试完成 ===")

if __name__ == '__main__':
    test_dataset() 
    test_model()
    