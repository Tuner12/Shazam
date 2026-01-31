import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from Dataset.simple_data_loader import create_data_loaders

# 设置代理（如有需要）
os.environ['http_proxy'] = 'http://192.168.1.18:7890'
os.environ['https_proxy'] = 'http://192.168.1.18:7890'

def extract_pathvqa_text_features(model_name, data_loader, save_path, device='cuda'):
    """提取PathVQA文本特征，并保存qa_type等信息"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Loading BERT model: {model_name}")
    
    # 使用本地BERT模型路径
    bert_path = '/home/leiwenhui/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
    model = BertModel.from_pretrained(bert_path, local_files_only=True)
    model = model.to(device)
    model.eval()

    question_features = []
    answer_features = []
    qids = []
    questions = []
    answers = []
    qa_types = []

    print(f"Extracting text features using BERT on device: {device}")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Text Features", unit="batch"):
            batch_questions = batch['questions']
            batch_answers = batch['answers']
            batch_qids = batch['qids']
            batch_qa_types = batch['qa_types'] if 'qa_types' in batch else [None]*len(batch_questions)

            # 处理问题
            question_inputs = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding='longest',
                truncation=True,
                max_length=25
            ).to(device)
            question_outputs = model(**question_inputs)
            question_features.append(question_outputs.last_hidden_state[:, 0, :].cpu())  # [CLS] token

            # 处理答案
            batch_answer_features = []
            for answer in batch_answers:
                if answer.strip():
                    answer_inputs = tokenizer(
                        answer,
                        return_tensors="pt",
                        padding='longest',
                    ).to(device)
                    answer_outputs = model(**answer_inputs)
                    answer_feature = answer_outputs.last_hidden_state[:, 0, :].cpu()
                else:
                    answer_feature = torch.zeros(1, 768)
                batch_answer_features.append(answer_feature)
            answer_features.extend(batch_answer_features)
            qids.extend(batch_qids)
            questions.extend(batch_questions)
            answers.extend(batch_answers)
            qa_types.extend(batch_qa_types)

    question_features = torch.cat(question_features, dim=0)
    answer_features = torch.cat(answer_features, dim=0)

    torch.save({
        'question_features': question_features,
        'answer_features': answer_features,
        'qids': qids,
        'questions': questions,
        'answers': answers,
        'qa_types': qa_types
    }, save_path)
    print(f"Text features saved to {save_path}")
    print(f"Shape of question features: {question_features.shape}")
    print(f"Shape of answer features: {answer_features.shape}")
    print(f"Number of samples: {len(qids)}")

if __name__ == "__main__":
    # 参数设置
    image_size = 224
    batch_size = 32
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'bert-base-cased'
    cache_dir = "/nas/leiwenhui/tys/PathVQA/Dataset/cache"

    # 创建PathVQA数据加载器
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        images_only=False,
        cache_dir=cache_dir
    )

    os.makedirs("features4pathvqa/texts", exist_ok=True)

    # 提取训练集文本特征
    print("\n正在提取训练集文本特征...")
    train_save_path = "features4pathvqa/texts/train_bert_features.pt"
    extract_pathvqa_text_features(model_name, train_loader, train_save_path, device)

    # 提取验证集文本特征
    print("正在提取验证集文本特征...")
    val_save_path = "features4pathvqa/texts/val_bert_features.pt"
    extract_pathvqa_text_features(model_name, val_loader, val_save_path, device)

    # 提取测试集文本特征
    print("正在提取测试集文本特征...")
    test_save_path = "features4pathvqa/texts/test_bert_features.pt"
    extract_pathvqa_text_features(model_name, test_loader, test_save_path, device)

    print("\n=== PathVQA 文本特征提取完成 ===")
    print(f"所有特征已保存到 features4pathvqa/texts 目录") 