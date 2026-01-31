#!/usr/bin/env python3
"""
将PathVQA的pkl格式数据转换为MUMC兼容的JSON格式
"""

import pickle
import json
import os

def load_pkl_file(file_path):
    """加载pkl文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_pvqa_to_json(data_dir, output_dir, split='train'):
    """
    将PathVQA数据转换为MUMC兼容的JSON格式
    
    Args:
        data_dir: PathVQA数据目录
        output_dir: 输出目录
        split: 数据集分割 ('train', 'val', 'test')
    """
    
    # 加载数据
    print(f"正在加载 {split} 数据...")
    
    # 加载问答数据
    vqa_file = os.path.join(data_dir, 'qas', f'{split}_vqa.pkl')
    vqa_data = load_pkl_file(vqa_file)
    
    # 加载问题ID到答案的映射
    qid2a_file = os.path.join(data_dir, 'qas', 'qid2a.pkl')
    qid2a = load_pkl_file(qid2a_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换数据
    json_data = []
    
    print(f"正在转换 {len(vqa_data)} 条样本...")
    
    for i, item in enumerate(vqa_data):
        qid = item['question_id']
        img_id = item['img_id']
        question = item['sent']
        
        # 从qid2a获取答案文本
        answer = qid2a.get(qid, "")
        
        # 创建JSON格式的样本
        sample = {
            "qid": str(qid),
            "image_name": img_id,
            "question": question,
            "answer": answer
        }
        
        json_data.append(sample)
        
        if i < 3:  # 打印前3个样本作为示例
            print(f"样本 {i+1}: {sample}")
    
    # 保存JSON文件
    output_file = os.path.join(output_dir, f'{split}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(json_data)} 条样本到 {output_file}")
    
    return json_data

def create_answer_list(data_dir, output_dir):
    """创建答案列表文件"""
    print("正在创建答案列表...")
    
    # 加载答案到标签的映射
    ans2label_file = os.path.join(data_dir, 'qas', 'ans2label.pkl')
    ans2label = load_pkl_file(ans2label_file)
    
    # 创建答案列表（只包含答案文本）
    answer_list = []
    for answer, label_id in ans2label.items():
        answer_list.append(answer)
    
    # 保存答案列表
    answer_list_file = os.path.join(output_dir, 'answer_list.json')
    with open(answer_list_file, 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(answer_list)} 个答案到 {answer_list_file}")
    
    return answer_list

def main():
    """主函数"""
    # 配置路径
    pvqa_dir = "Pathvqa/pvqa"
    output_dir = "Dataset/pvqa_json"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换训练集
    print("=" * 50)
    print("转换训练集...")
    train_data = convert_pvqa_to_json(pvqa_dir, output_dir, 'train')
    
    # 转换验证集
    print("=" * 50)
    print("转换验证集...")
    val_data = convert_pvqa_to_json(pvqa_dir, output_dir, 'val')
    
    # 转换测试集
    print("=" * 50)
    print("转换测试集...")
    test_data = convert_pvqa_to_json(pvqa_dir, output_dir, 'test')
    
    # 创建答案列表
    print("=" * 50)
    create_answer_list(pvqa_dir, output_dir)
    
    print("=" * 50)
    print("转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"训练集: {len(train_data)} 条样本")
    print(f"验证集: {len(val_data)} 条样本")
    print(f"测试集: {len(test_data)} 条样本")

if __name__ == "__main__":
    main() 