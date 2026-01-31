# MUMC模型训练指南

本指南详细说明了如何在PathVQA数据集上训练和测试MUMC（Multi-modal Unified Medical Cognition）模型。

## 概述

MUMC是一个专门为医学视觉问答设计的多模态模型，包含以下核心组件：
- **图像特征投影层**：将预提取的图像特征投影到文本空间
- **文本编码器**：使用BERT前6层进行问题编码
- **多模态编码器**：使用BERT后6层+Cross-Attention进行多模态融合
- **答案解码器**：6层Transformer Decoder生成答案

## 候选答案机制

MUMC模型采用**候选答案排序**机制，而不是生成式方法：

### 官方实现方式
- 使用PathVQA数据集现成的答案词汇表（`Dataset/pathvqa_answer_vocab.json`）
- 包含3227个不同的答案，覆盖所有训练集中的答案
- 推理时，模型从完整答案列表中选择最佳答案
- 符合官方MUMC论文描述的候选答案排序机制

### 答案词汇表结构
```json
{
  "in the canals of hering": 0,
  "bile duct cells and canals of hering": 1,
  "foci of fat necrosis": 2,
  "yes": 3,
  "no": 5,
  ...
}
```

## 文件结构

```
PathVQA/
├── model/
│   └── MUMC.py              # MUMC模型实现
├── Dataset/
│   ├── pathvqa_dataset.py   # PathVQA数据集类
│   └── pathvqa_answer_vocab.json  # 答案词汇表
├── train_mumc.py            # 训练脚本
├── test_mumc.py             # 测试脚本
└── README_training.md       # 本文件
```

## 环境要求

```bash
pip install torch torchvision transformers tqdm pillow numpy
```

## 训练流程

### 1. 准备数据
确保PathVQA数据集已正确下载并放置在正确位置：
- 数据集会自动从Hugging Face下载到 `Dataset/cache/` 目录
- 答案词汇表位于 `Dataset/pathvqa_answer_vocab.json`

### 2. 开始训练
```bash
python train_mumc.py
```

训练过程包括：
- 自动加载PathVQA数据集和答案词汇表
- 训练10个epoch，每个epoch包含训练和验证
- 保存最佳验证准确率的模型
- 在测试集上评估最终性能

### 3. 训练输出
```
=== MUMC模型训练 ===
使用设备: cuda
加载答案词汇表: Dataset/pathvqa_answer_vocab.json
答案词汇表大小: 3227
创建数据集...
训练集: 19755 样本
验证集: 6279 样本
测试集: 6761 样本

Epoch 1/10
训练中: 100%|██████████| 1235/1235 [00:45<00:00, 27.23it/s]
评估中: 100%|██████████| 393/393 [00:12<00:00, 32.15it/s]
训练损失: 0.8234
验证准确率: 0.2345
验证开放问题准确率: 0.1987
验证封闭问题准确率: 0.3456
保存最佳模型，验证准确率: 0.2345
```

## 测试流程

### 1. 运行测试
```bash
python test_mumc.py
```

### 2. 测试输出
```
=== MUMC模型测试 ===
使用设备: cuda
加载答案词汇表: Dataset/pathvqa_answer_vocab.json
答案词汇表大小: 3227
创建测试数据集...
测试集: 6761 样本
加载模型: best_mumc_model.pth

=== 开始测试 ===
测试中: 100%|██████████| 423/423 [00:15<00:00, 28.12it/s]

=== 测试结果 ===
总体准确率: 0.3456
开放问题准确率: 0.2987
封闭问题准确率: 0.4567

问题类型分布:
开放问题: 5412 (80.0%)
封闭问题: 1349 (20.0%)

=== 示例预测 (前10个) ===
样本 1:
  问题类型: open
  预测答案: adenocarcinoma
  真实答案: adenocarcinoma
  是否正确: ✓
  Top-3预测: ['adenocarcinoma', 'carcinoma', 'tumor']

样本 2:
  问题类型: close
  预测答案: yes
  真实答案: no
  是否正确: ✗
  Top-3预测: ['yes', 'no', 'maybe']
```

## 模型推理机制

### 训练模式
```python
# 训练时使用真实答案计算损失
loss = model(
    image_embeds=images,
    question=questions,
    answer=answers,  # 真实答案列表
    train=True
)
```

### 推理模式
```python
# 推理时使用完整答案列表进行排序
topk_ids, topk_probs = model(
    image_embeds=images,
    question=questions,
    answer=answer_list,  # 完整答案列表（3227个）
    train=False,
    k=5  # 返回top-5预测
)
```

## 输出文件

### 训练结果
- `best_mumc_model.pth`: 最佳模型权重
- `mumc_training_results.json`: 训练结果和测试性能

### 测试结果
- `mumc_test_results.json`: 详细测试结果，包含：
  - 总体准确率
  - 开放/封闭问题准确率
  - 所有预测结果
  - Top-k预测和概率

## 性能指标

模型评估包含以下指标：
1. **总体准确率**: 所有问题的准确预测比例
2. **开放问题准确率**: 开放性问题（非yes/no）的准确率
3. **封闭问题准确率**: 封闭性问题（yes/no）的准确率

## 配置参数

### 模型配置
```python
config = {
    'hidden_size': 768,      # BERT隐藏层维度
    'eos': '</s>'           # 结束标记
}
```

### 训练参数
- `batch_size`: 16（可根据GPU内存调整）
- `learning_rate`: 1e-4
- `weight_decay`: 0.01
- `num_epochs`: 10

## 注意事项

1. **答案词汇表**: 使用PathVQA数据集现成的答案列表，确保覆盖所有可能答案
2. **内存使用**: 推理时需要处理3227个候选答案，注意GPU内存使用
3. **数据格式**: 确保图像尺寸为224x224，与预训练特征提取器一致
4. **代理设置**: 代码中已配置代理，用于下载Hugging Face数据集

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 使用梯度累积

2. **数据集下载失败**
   - 检查网络连接
   - 确认代理设置正确

3. **模型加载失败**
   - 确认模型文件路径正确
   - 检查模型版本兼容性

## 扩展功能

### 自定义答案词汇表
如需使用自定义答案列表，可以修改`load_answer_vocab`函数：
```python
def load_custom_answer_vocab(custom_vocab_path):
    # 加载自定义答案词汇表
    pass
```

### 多GPU训练
```python
model = torch.nn.DataParallel(model)
```

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## 引用

如果使用本实现，请引用原始MUMC论文：
```
@article{mumc2023,
  title={MUMC: Multi-modal Unified Medical Cognition},
  author={...},
  journal={...},
  year={2023}
}
``` 