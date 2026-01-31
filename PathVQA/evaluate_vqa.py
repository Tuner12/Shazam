import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score
import re
from collections import defaultdict

class VQAModel(nn.Module):
    """VQA模型，结合图像和文本特征进行答案预测"""
    
    def __init__(self, image_feature_dim, text_feature_dim, num_answers, fusion_dim=1024, dropout=0.1):
        super(VQAModel, self).__init__()
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 答案预测层
        self.answer_predictor = nn.Linear(fusion_dim // 2, num_answers)
        
    def forward(self, image_features, text_features):
        # 连接图像和文本特征
        combined_features = torch.cat([image_features, text_features], dim=-1)
        
        # 特征融合
        fused_features = self.fusion(combined_features)
        
        # 答案预测
        answer_logits = self.answer_predictor(fused_features)
        
        return answer_logits

class VQADataset(Dataset):
    """VQA数据集，加载预提取的特征"""
    
    def __init__(self, image_features_path, text_features_path, answer_vocab_path=None, split='test'):
        """
        Args:
            image_features_path: 图像特征文件路径
            text_features_path: 文本特征文件路径
            answer_vocab_path: 答案词汇表路径（closed-acc需要）
            split: 数据集分割（train/val/test）
        """
        self.split = split
        
        # 加载图像特征
        print(f"Loading image features from {image_features_path}")
        image_data = torch.load(image_features_path)
        if isinstance(image_data, tuple):
            # 多尺度特征格式 (low_features, mid_features, high_features, image_ids)
            self.image_features = image_data[2]  # 使用high_features
            self.image_ids = image_data[3]
        else:
            # 单尺度特征格式
            self.image_features = image_data['features']
            self.image_ids = image_data['image_ids']
        
        # 加载文本特征
        print(f"Loading text features from {text_features_path}")
        text_data = torch.load(text_features_path)
        self.question_features = text_data['question_features']
        self.answer_features = text_data['answer_features']
        self.questions = text_data['questions']
        self.answers = text_data['answers']
        self.text_image_ids = text_data['image_ids']
        
        # 创建image_id到索引的映射
        self.image_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}
        
        # 过滤有图像特征的样本
        self.valid_indices = []
        for idx, img_id in enumerate(self.text_image_ids):
            if img_id in self.image_id_to_idx:
                self.valid_indices.append(idx)
        
        print(f"Valid samples: {len(self.valid_indices)} / {len(self.text_image_ids)}")
        
        # 加载答案词汇表（closed-acc）
        self.answer_vocab = None
        if answer_vocab_path and os.path.exists(answer_vocab_path):
            print(f"Loading answer vocabulary from {answer_vocab_path}")
            vocab_data = torch.load(answer_vocab_path)
            self.answer_to_idx = vocab_data['answer_to_idx']
            self.idx_to_answer = vocab_data['idx_to_answer']
            self.answer_vocab = vocab_data['filtered_answers']
            self.num_answers = len(self.answer_to_idx)
            print(f"Answer vocabulary size: {self.num_answers}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        text_idx = self.valid_indices[idx]
        img_id = self.text_image_ids[text_idx]
        img_idx = self.image_id_to_idx[img_id]
        
        # 获取特征
        image_feature = self.image_features[img_idx]
        question_feature = self.question_features[text_idx]
        answer = self.answers[text_idx]
        
        # 准备标签（closed-acc）
        if self.answer_vocab is not None and answer.strip():
            answer_clean = answer.lower().strip()
            if answer_clean in self.answer_to_idx:
                label = self.answer_to_idx[answer_clean]
            else:
                label = -1  # 未知答案
        else:
            label = -1
        
        return {
            'image_feature': image_feature,
            'question_feature': question_feature,
            'label': label,
            'image_id': img_id,
            'question': self.questions[text_idx],
            'answer': answer
        }

def normalize_answer(answer):
    """标准化答案，用于open-acc计算"""
    # 转换为小写
    answer = answer.lower()
    
    # 移除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # 移除多余空格
    answer = ' '.join(answer.split())
    
    return answer

def compute_open_accuracy(predictions, ground_truths):
    """计算open-acc（开放词汇准确率）"""
    correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_normalized = normalize_answer(pred)
        gt_normalized = normalize_answer(gt)
        
        if pred_normalized == gt_normalized:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def compute_closed_accuracy(predicted_indices, ground_truth_indices):
    """计算closed-acc（封闭词汇准确率）"""
    # 过滤掉标签为-1的样本（未知答案）
    valid_mask = ground_truth_indices != -1
    if valid_mask.sum() == 0:
        return 0.0
    
    valid_preds = predicted_indices[valid_mask]
    valid_gts = ground_truth_indices[valid_mask]
    
    return accuracy_score(valid_gts, valid_preds)

def evaluate_model(model, data_loader, device, mode='test'):
    """评估模型，计算open-acc和closed-acc"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_predicted_answers = []
    all_gt_answers = []
    
    # 详细结果记录
    detailed_results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            image_features = batch['image_feature'].to(device)
            question_features = batch['question_feature'].to(device)
            labels = batch['label']
            answers = batch['answer']
            questions = batch['question']
            image_ids = batch['image_id']
            
            outputs = model(image_features, question_features)
            predicted_indices = torch.argmax(outputs, dim=1).cpu()
            
            # 收集closed-acc数据
            all_predictions.extend(predicted_indices.numpy())
            all_ground_truths.extend(labels.numpy())
            
            # 收集open-acc数据
            for i, (pred_idx, gt_answer) in enumerate(zip(predicted_indices, answers)):
                if hasattr(data_loader.dataset, 'idx_to_answer') and pred_idx.item() in data_loader.dataset.idx_to_answer:
                    pred_answer = data_loader.dataset.idx_to_answer[pred_idx.item()]
                else:
                    pred_answer = "unknown"
                
                all_predicted_answers.append(pred_answer)
                all_gt_answers.append(gt_answer)
                
                # 记录详细结果
                detailed_results.append({
                    'image_id': image_ids[i],
                    'question': questions[i],
                    'predicted_answer': pred_answer,
                    'ground_truth': gt_answer,
                    'predicted_idx': pred_idx.item(),
                    'ground_truth_idx': labels[i].item()
                })
    
    # 计算closed-acc
    closed_acc = compute_closed_accuracy(
        np.array(all_predictions), 
        np.array(all_ground_truths)
    )
    
    # 计算open-acc
    open_acc = compute_open_accuracy(all_predicted_answers, all_gt_answers)
    
    print(f"\n=== {mode.upper()} 评估结果 ===")
    print(f"Closed-ACC: {closed_acc:.4f}")
    print(f"Open-ACC: {open_acc:.4f}")
    
    # 分析错误案例
    analyze_errors(detailed_results, all_predicted_answers, all_gt_answers)
    
    return closed_acc, open_acc, detailed_results

def analyze_errors(detailed_results, predicted_answers, gt_answers):
    """分析错误案例"""
    print(f"\n=== 错误分析 ===")
    
    # 统计错误类型
    error_types = defaultdict(int)
    open_correct = 0
    closed_correct = 0
    total = len(detailed_results)
    
    for i, result in enumerate(detailed_results):
        pred_norm = normalize_answer(result['predicted_answer'])
        gt_norm = normalize_answer(result['ground_truth'])
        
        # Open-acc检查
        if pred_norm == gt_norm:
            open_correct += 1
        
        # Closed-acc检查
        if result['predicted_idx'] == result['ground_truth_idx'] and result['ground_truth_idx'] != -1:
            closed_correct += 1
        
        # 错误类型分析
        if pred_norm != gt_norm:
            if result['ground_truth_idx'] == -1:
                error_types['unknown_answer'] += 1
            elif result['predicted_idx'] == result['ground_truth_idx']:
                error_types['normalization_error'] += 1
            else:
                error_types['wrong_prediction'] += 1
    
    print(f"总样本数: {total}")
    print(f"Open-ACC正确: {open_correct}")
    print(f"Closed-ACC正确: {closed_correct}")
    print(f"错误类型分布:")
    for error_type, count in error_types.items():
        print(f"  {error_type}: {count} ({count/total*100:.1f}%)")
    
    # 显示一些错误案例
    print(f"\n=== 错误案例示例 ===")
    error_count = 0
    for result in detailed_results:
        pred_norm = normalize_answer(result['predicted_answer'])
        gt_norm = normalize_answer(result['ground_truth'])
        
        if pred_norm != gt_norm and error_count < 5:
            print(f"问题: {result['question']}")
            print(f"预测: {result['predicted_answer']} (标准化: {pred_norm})")
            print(f"真实: {result['ground_truth']} (标准化: {gt_norm})")
            print(f"图像ID: {result['image_id']}")
            print("-" * 50)
            error_count += 1

if __name__ == "__main__":
    import sys
    sys.path.append('/nas/leiwenhui/tys/PathVQA')
    
    # 设置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    
    # 模型参数（需要与训练时保持一致）
    image_feature_dim = 1280  # 根据实际图像特征维度调整
    text_feature_dim = 768   # BERT特征维度
    
    print("=== VQAMed 模型评估 ===")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    
    try:
        # 加载数据集
        print("正在加载测试数据集...")
        
        test_dataset = VQADataset(
            image_features_path="features/images/test_hoptimus1_features.pt",
            text_features_path="features/text/test_bert_features.pt",
            answer_vocab_path="features/text/answer_vocabulary.pt",
            split='test'
        )
        
        print(f"测试集大小: {len(test_dataset)} 样本")
        
        # 创建数据加载器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        model = VQAModel(
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            num_answers=test_dataset.num_answers,
            fusion_dim=1024,
            dropout=0.1
        )
        
        # 加载训练好的模型
        model_path = 'best_vqa_model.pth'
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化的模型")
        
        # 评估模型
        print("\n=== 开始评估 ===")
        closed_acc, open_acc, detailed_results = evaluate_model(model, test_loader, device, mode='test')
        
        # 保存详细结果
        results = {
            'closed_acc': closed_acc,
            'open_acc': open_acc,
            'detailed_results': detailed_results[:100]  # 只保存前100个结果作为示例
        }
        
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 评估完成 ===")
        print(f"最终结果:")
        print(f"Closed-ACC: {closed_acc:.4f}")
        print(f"Open-ACC: {open_acc:.4f}")
        print(f"详细结果已保存到 evaluation_results.json")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 