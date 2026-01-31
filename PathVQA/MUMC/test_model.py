import argparse
import os
import sys
import ruamel.yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
# from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from torch.utils.data import DataLoader
from utils import cosine_lr_schedule
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.vqa_feature_dataset import vqa_feature_dataset
from vqaEvaluate import compute_vqa_acc
# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'


if __name__ == '__main__':
    from ruamel.yaml import YAML
    yaml_loader = YAML(typ='rt')
    with open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
        config = yaml_loader.load(f)
    # 重新下载BERT模型
    print("读取下载BERT模型...")
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # from tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    # image = torch.randn(4, 3, 384, 384)
    # image_dim = image.shape[1]
    model = MUMC_VQA(config=config, text_encoder='bert-base-uncased', text_decoder='bert-base-uncased', tokenizer=tokenizer, image_embeds_dim=1280)
    print("BERT模型下载成功！")
    print(model)

    # 测试模型
    print("测试模型...")

    train_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/train_virchow2_features.pt'
    test_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/test_virchow2_features.pt'
    answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    
    # 测试训练集
    print("=== 训练集 ===")
    dataset = vqa_feature_dataset(train_feature_file, split='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for i, (image, question, answer) in enumerate(dataloader):
        print("image：", image.shape)
        print("question：", question)
        print("answer：", answer)
        print("answer type：", type(answer))
        print("answer length：", len(answer))
        if isinstance(answer, list) and len(answer) == 1:
            answer = list(answer[0])
        print("answer：", answer)
        loss = model(image, question, answer)
        print("损失值：", loss)
        print("测试模型成功！")
        break
    
    # d1 = dataset[0]
    # print(f"  Image feature shape: {d1[0].shape}")
    # print(f"  Question: {d1[1]}")
    # print(f"  Answer: {d1[2]}")
    # print(f"  Feature dimension: {dataset.get_feature_dim()}")
    # print(f"  Dataset length: {len(dataset)}")
    # image = d1[0]
    # question = d1[1]
    # answer = d1[2]
    # loss = model(image, question, answer)
    # print("损失值：", loss)
    # print("测试模型成功！")
    # 测试测试集
    # print("\n=== 测试测试集 ===")
    # dataset = vqa_feature_dataset(test_feature_file, answer_list=answer_list, split='test')
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    # for i, (image, question, answer) in enumerate(dataloader):
    #     topk_ids, topk_probs = model(image, question, answer, k=3, train=False)
    #     print("topk_ids：", topk_ids)
    #     print("topk_probs：", topk_probs)
    #     print("测试模型成功！")
    #     break