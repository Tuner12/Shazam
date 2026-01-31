# PathVQA 数据集工具

这个目录包含了用于处理PathVQA医学视觉问答数据集的工具集。

## 数据集概述

PathVQA是一个医学病理学视觉问答数据集，包含：
- **训练集**: 19,755个样本
- **验证集**: 6,279个样本  
- **测试集**: 6,761个样本
- **图像**: 5,004张病理学图像
- **问题类型**: what, where, when, who, how, why, does, is等
- **答案类型**: other, yes/no, number等

## 文件结构

```
Dataset/
├── simple_hf_dataset.py         # 简化HF数据集类（推荐，无BERT）
├── simple_data_loader.py        # 简化HF数据加载器工具
├── example_usage.py             # 使用示例
├── pathvqa_dataset.py           # 原始数据集类（使用本地文件）
├── vqamed_dataset.py            # VQAMed数据集类
├── answer_vocab.json            # 答案词汇表（运行后生成）
└── README.md                   # 本文件
```

## 主要功能

### 简化Hugging Face数据集类 (SimpleHFPathVQADataset) - 推荐

使用Hugging Face datasets库的简化Dataset类，提供以下功能：

- **自动下载**: 自动从Hugging Face下载数据集
- **图像处理**: 自动处理图像格式转换
- **文本处理**: 简单的空格分词（不包含BERT）
- **答案处理**: 构建和管理答案词汇表（不过滤低频答案）
- **数据统计**: 提供详细的数据集统计信息
- **更简单**: 无需处理复杂的文件路径映射

## 使用方法

### 基本使用

```python
from simple_hf_dataset import create_simple_hf_datasets
from simple_data_loader import create_simple_data_loaders

# 创建数据集
train_dataset, val_dataset, test_dataset = create_simple_hf_datasets(
    image_size=224
)

# 创建数据加载器
train_loader, val_loader, test_loader, _ = create_simple_data_loaders(
    batch_size=32,
    num_workers=4,
    image_size=224
)
```

### 单个样本示例

```python
# 获取单个样本
sample = train_dataset[0]

print(f"问题ID: {sample['qid']}")
print(f"问题: {sample['question']}")
print(f"问题词列表: {sample['question_words']}")
print(f"答案: {sample['answer']}")
print(f"图像形状: {sample['image'].shape}")
```

### 批次数据示例

```python
# 获取批次数据
for batch in train_loader:
    print(f"批次大小: {len(batch['qids'])}")
    print(f"图像形状: {batch['images'].shape}")
    print(f"答案索引形状: {batch['answer_indices'].shape}")
    break
```

## 数据格式

### 输入数据格式

每个样本包含以下字段：

```python
{
    'qid': int,                    # 问题ID
    'image': torch.Tensor,         # 图像张量 (3, 224, 224)
    'question': str,               # 问题文本
    'question_words': List[str],   # 问题词列表（简单分词）
    'answer': str,                 # 原始答案
    'normalized_answer': str,      # 标准化答案
    'answer_type': str,            # 答案类型
    'answer_idx': int,             # 答案在词汇表中的索引
    'question_type': str           # 问题类型
}
```

### 批次数据格式

```python
{
    'qids': List[int],             # 问题ID列表
    'images': torch.Tensor,        # 图像批次 (batch_size, 3, 224, 224)
    'questions': List[str],        # 问题文本列表
    'question_words': List[List[str]], # 问题词列表的列表
    'answers': List[str],          # 答案列表
    'normalized_answers': List[str], # 标准化答案列表
    'answer_types': List[str],     # 答案类型列表
    'answer_indices': torch.Tensor, # 答案索引张量 (batch_size,)
    'question_types': List[str]    # 问题类型列表
}
```

## 配置参数

### 数据集参数

- `image_size`: 图像尺寸（默认224）
- `split`: 数据集分割 ('train', 'validation', 'test')

### 数据加载器参数

- `batch_size`: 批次大小（默认32）
- `num_workers`: 工作进程数（默认4）
- `shuffle_train`: 是否打乱训练集（默认True）
- `shuffle_val`: 是否打乱验证集（默认False）
- `shuffle_test`: 是否打乱测试集（默认False）

## 特点

1. **不过滤低频答案**: 保留所有答案，不进行频率过滤
2. **简单文本处理**: 使用空格分词，不包含BERT tokenizer
3. **自动下载**: 自动从Hugging Face下载数据集
4. **图像预处理**: 自动调整图像尺寸和标准化
5. **答案词汇表**: 自动构建和管理答案词汇表

## 测试

运行以下命令测试数据集功能：

```bash
# 测试简化数据集
python simple_hf_dataset.py

# 测试简化数据加载器
python simple_data_loader.py

# 运行完整使用示例
python example_usage.py
```

## 依赖库

- torch
- torchvision
- datasets (Hugging Face)
- PIL
- numpy
- json
- re
- collections

## 注意事项

1. **网络连接**: 需要网络连接来下载Hugging Face数据集
2. **代理设置**: 代码中已设置代理，如需修改请编辑simple_hf_dataset.py
3. **内存使用**: 建议根据GPU内存调整批次大小
4. **多进程**: 在Windows系统上可能需要设置`num_workers=0`
5. **答案词汇表**: 答案词汇表基于训练集构建，包含所有答案（不过滤） 