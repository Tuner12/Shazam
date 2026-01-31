# VQAMed 文本特征提取与模型训练

本项目提供了完整的VQAMed数据集文本特征提取和VQA模型训练流程，支持open-acc和closed-acc两种评估方式。

## 文件说明

### 1. `extract_text_feature.py` - 文本特征提取
- 使用BERT-base-cased模型提取问题和答案的文本特征
- 分析答案分布并创建答案词汇表
- 支持训练集、验证集和测试集的特征提取

### 2. `train_vqa_model.py` - VQA模型训练
- 结合图像和文本特征进行答案预测
- 支持open-acc和closed-acc两种评估方式
- 包含完整的训练、验证和测试流程

### 3. `extract_multi_image_feature.py` - 图像特征提取
- 提取多尺度图像特征（early, middle, high）
- 支持多种预训练模型（UNI, CONCH, GigaPath, H-Optimus等）

## 使用流程

### 第一步：提取图像特征
```bash
python extract_multi_image_feature.py
```

这将提取图像特征并保存到 `features/images/` 目录。

### 第二步：提取文本特征
```bash
python extract_text_feature.py
```

这将：
- 使用BERT提取问题和答案的文本特征
- 分析答案分布并创建答案词汇表
- 保存特征到 `features/text/` 目录

### 第三步：训练VQA模型
```bash
python train_vqa_model.py
```

这将：
- 加载预提取的图像和文本特征
- 训练VQA模型
- 计算open-acc和closed-acc结果

## 两种评估方式

### 1. Closed-ACC（封闭词汇准确率）
- 基于预定义的答案词汇表
- 模型从固定答案集合中选择答案
- 适用于答案类别有限的情况
- 计算方式：准确预测答案类别的比例

### 2. Open-ACC（开放词汇准确率）
- 不限制答案词汇表
- 模型可以生成任意答案
- 适用于答案多样的情况
- 计算方式：预测答案与真实答案的精确匹配

## 数据格式

### 图像特征格式
```python
# 多尺度特征
(low_features, mid_features, high_features, image_ids)

# 单尺度特征
{
    'features': tensor,
    'image_ids': list
}
```

### 文本特征格式
```python
{
    'question_features': tensor,  # 问题特征
    'answer_features': tensor,    # 答案特征
    'image_ids': list,           # 图像ID
    'questions': list,           # 原始问题
    'answers': list              # 原始答案
}
```

### 答案词汇表格式
```python
{
    'answer_to_idx': dict,       # 答案到索引的映射
    'idx_to_answer': dict,       # 索引到答案的映射
    'filtered_answers': list     # 过滤后的答案列表
}
```

## 模型架构

VQA模型包含以下组件：
1. **特征融合层**：将图像和文本特征连接并融合
2. **多层感知机**：处理融合后的特征
3. **答案预测层**：输出答案类别的概率分布

## 配置参数

### 图像特征提取
- `model_name`: 使用的预训练模型
- `target_img_size`: 图像尺寸
- `batch_size`: 批次大小

### 文本特征提取
- `max_length`: 文本最大长度
- `batch_size`: 批次大小

### 模型训练
- `image_feature_dim`: 图像特征维度
- `text_feature_dim`: 文本特征维度（BERT: 768）
- `fusion_dim`: 特征融合维度
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率

## 输出结果

训练完成后，将生成：
- `best_vqa_model.pth`: 最佳模型权重
- `training_results.json`: 训练结果和评估指标

## 注意事项

1. **数据路径**：确保数据路径正确，特别是图像和QA文件的路径
2. **GPU内存**：根据GPU内存调整batch_size
3. **特征维度**：根据实际使用的图像模型调整image_feature_dim
4. **答案词汇表**：closed-acc需要预定义的答案词汇表

## 依赖库

```bash
pip install torch torchvision transformers tqdm scikit-learn pillow numpy
```

## 示例结果

训练完成后，你将看到类似以下的输出：
```
=== 训练完成 ===
最终测试结果:
Closed-ACC: 0.7234
Open-ACC: 0.6891
结果已保存到 training_results.json
```

这表示模型在测试集上达到了72.34%的封闭词汇准确率和68.91%的开放词汇准确率。 