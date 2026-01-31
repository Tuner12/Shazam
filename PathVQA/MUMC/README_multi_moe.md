# Multi-MoE 蒸馏版本的 PathVQA 模型

本项目为 PathVQA 项目增加了 Multi-MoE 蒸馏版本，支持多教师模型的蒸馏训练。

## 新增文件

### 1. 数据集相关
- `dataset/multi_feature_dataset.py`: 支持返回浅中深三层特征的数据集类
- 修改了 `dataset/__init__.py`: 添加了 `multi_feature_dataset` 和 `multi_moe_collate_fn` 的支持

### 2. 模型相关
- `models/model_vqa_multi_moe.py`: Multi-MoE 蒸馏版本的 VQA 模型

### 3. 训练相关
- `train_vqa_multi_moe.py`: Multi-MoE 蒸馏版本的训练脚本
- `test_multi_moe.py`: 测试脚本

## 主要特性

### 1. Multi-MoE 架构
- **MoE 模块**: 每个层级（low/mid/high）使用一个 MoE 来处理 4 个教师的特征
- **Cross-Attention**: 三段 cross-attention，每段 5 层，总共 15 层
- **特征融合**: 通过 gating 网络动态融合不同教师的特征

### 2. 多层蒸馏
- **特征蒸馏**: 学生的不同层级特征与教师特征进行蒸馏
- **可配置层数**: 支持 5/10/15 层的不同配置
- **蒸馏损失**: 结合余弦相似度和 Huber 损失

### 3. 数据处理
- **三层特征**: 返回 low/mid/high 三层特征而不是单一深层特征
- **多教师支持**: 支持 4 个教师模型的特征融合

## 使用方法

### 1. 测试模型和数据集

```bash
cd PathVQA/MUMC
python test_multi_moe.py
```

### 2. 训练 Multi-MoE 模型

```bash
cd PathVQA/MUMC
python train_vqa_multi_moe.py \
    --config ./configs/VQA.yaml \
    --output_dir output/multi_moe_vqa \
    --dataset_use multi_moe_pathvqa \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased
```

### 3. 配置参数

在 `configs/VQA.yaml` 中可以配置以下参数：

```yaml
# Multi-MoE 相关配置
d_model: 128                    # 模型维度
num_layers: 15                  # 层数 (5/10/15)
lambda_distill: 0.01           # 蒸馏损失权重
distill: true                  # 是否启用蒸馏

# 训练相关配置
batch_size_train: 64
batch_size_test: 64
max_epoch: 50
alpha: 0.4                     # 蒸馏温度参数
```

## 模型架构详解

### 1. MoE 模块 (MoEOnePerLevel)
```python
class MoEOnePerLevel(nn.Module):
    def __init__(self, in_dims, d_model=128):
        # 4个教师的特征投影
        self.proj_list = nn.ModuleList([
            nn.Linear(in_dim, d_model) for in_dim in in_dims
        ])
        # Gating 网络
        self.gate = nn.Sequential(...)
```

### 2. Cross-Attention 模块
```python
class MultiCrossAttentionLayers(nn.Module):
    def __init__(self, d_model, num_layers=5):
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model) for _ in range(num_layers)
        ])
```

### 3. 主模型 (MUMC_VQA_Multi_MoE)
```python
class MUMC_VQA_Multi_MoE(nn.Module):
    def __init__(self, ...):
        # 三个层级的 MoE
        self.moe_low = MoEOnePerLevel(dim_list_low, d_model)
        self.moe_mid = MoEOnePerLevel(dim_list_mid, d_model)
        self.moe_high = MoEOnePerLevel(dim_list_high, d_model)
        
        # 三段 Cross-Attention
        self.segment1 = MultiCrossAttentionLayers(d_model, 5)
        self.segment2 = MultiCrossAttentionLayers(d_model, 5)
        self.segment3 = MultiCrossAttentionLayers(d_model, 5)
```

## 数据格式

### 输入数据格式
训练时，每个样本包含：
- `low_features`: 低级特征 [batch_size, low_dim]
- `mid_features`: 中级特征 [batch_size, mid_dim]  
- `high_features`: 高级特征 [batch_size, high_dim]
- `question`: 问题文本
- `answer`: 答案文本

### 特征文件格式
特征文件应包含以下结构：
```python
(low_features, mid_features, high_features, questions, answers/question_ids)
```

## 蒸馏策略

### 1. 特征蒸馏
- 学生的 out1 与教师的 low 特征对齐
- 学生的 out2 与教师的 mid 特征对齐  
- 学生的 out3 与教师的 high 特征对齐

### 2. 损失函数
```python
def distill_pair(student_feat, teacher_feat):
    cos_term = 1.0 - cosine_similarity(student_feat, teacher_feat)
    smooth_l1 = HuberLoss()(student_feat, teacher_feat)
    return cos_term + smooth_l1
```

## 注意事项

1. **特征维度**: 确保 4 个教师的特征维度配置正确
2. **内存使用**: Multi-MoE 模型比单模型占用更多内存
3. **训练时间**: 由于模型复杂度增加，训练时间会相应增加
4. **数据格式**: 确保特征文件包含三层特征数据

## 故障排除

### 常见问题

1. **维度不匹配**: 检查 `dim_list_low/mid/high` 配置
2. **内存不足**: 减小 batch_size 或使用梯度累积
3. **数据加载错误**: 检查特征文件路径和格式

### 调试建议

1. 使用 `test_multi_moe.py` 验证模型和数据
2. 检查配置文件中的参数设置
3. 监控训练过程中的损失变化

## 扩展功能

### 1. 添加更多教师
修改 `MoEOnePerLevel` 类以支持更多教师模型

### 2. 自定义蒸馏策略
在 `multi_level_distillation_loss` 函数中实现自定义蒸馏策略

### 3. 不同的融合策略
修改 gating 网络或添加其他特征融合方法 