import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(12)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------ 数据加载 ------------------
train_features1, train_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/virchow_train_features.pt", weights_only=True)
train_features2, train_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/uni_v1_train_features.pt", weights_only=True)
train_features3, train_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/phikon_v2_train_features.pt", weights_only=True)
train_features4, train_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/gigapath_train_features.pt", weights_only=True)

test_features1, test_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/virchow_test_features.pt", weights_only=True)
test_features2, test_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/uni_v1_test_features.pt", weights_only=True)
test_features3, test_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/phikon_v2_test_features.pt", weights_only=True)
test_features4, test_labels = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/Combined_Pathology_Models/features/gigapath_test_features.pt", weights_only=True)

train_features_list = [train_features1, train_features2, train_features3, train_features4]
test_features_list = [test_features1, test_features2, test_features3, test_features4]
# print(train_features1.shape)
# 特征维度
input_dims = [1280, 1024, 1024, 1536]
num_classes = len(torch.unique(train_labels))

# 组合训练数据集
train_dataset = TensorDataset(*train_features_list, train_labels)

# 组合测试数据集，并划分 50% 作为验证集，50% 作为最终测试集
test_dataset = TensorDataset(*test_features_list, test_labels)
val_size = int(0.5 * len(test_dataset))
test_size = len(test_dataset) - val_size
val_dataset, final_test_dataset = random_split(test_dataset, [val_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(final_test_dataset, batch_size=64, shuffle=False)

# ------------------ 计算 Top-K Accuracy ------------------
def calculate_topk_accuracy(predictions, labels, k):
    topk_indices = torch.topk(predictions, k, dim=1).indices
    correct = sum(labels[i].item() in topk_indices[i] for i in range(len(labels)))
    return correct / len(labels)

# ------------------ 早停机制 (基于 BA + Weighted F1 加权和) ------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, lambda_ba=0.8, lambda_f1=0.2):
        self.patience = patience
        self.min_delta = min_delta
        self.lambda_ba = lambda_ba
        self.lambda_f1 = lambda_f1
        self.best_score = 0
        self.counter = 0
        self.best_epoch = 0
        self.best_model_state = None  # 用于存储最佳模型参数

    def __call__(self, val_ba, val_f1, model, epoch):
        validation_score = self.lambda_ba * val_ba + self.lambda_f1 * val_f1
        if validation_score > self.best_score + self.min_delta:
            self.best_score = validation_score
            self.best_epoch = epoch
            self.best_model_state = model.state_dict()  # 记录最佳模型参数
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience  # 达到 patience 限制则返回 True

# 初始化早停对象
early_stopping = EarlyStopping(patience=10, min_delta=1e-4, lambda_ba=0.8, lambda_f1=0.2)

# ------------------ 模型定义 ------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, features):
        queries = self.query(features)
        keys = self.key(features)
        values = self.value(features)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(keys.size(-1))
        attention_weights = self.softmax(attention_scores)
        attended_features = torch.matmul(attention_weights, values)
        out = self.output_layer(attended_features)
        out = self.layernorm(features + out)
        return out, attention_weights

class MultiCrossAttention(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super(MultiCrossAttention, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model) for _ in range(num_layers)])

    def forward(self, features):
        attention_weights_list = []
        for layer in self.layers:
            features, attention_weights = layer(features)
            attention_weights_list.append(attention_weights)
        # print(features.shape)
        fused_features = features.mean(dim=1)
        return fused_features, attention_weights_list

class CrossAttentionClassifierWithDistillation(nn.Module):
    def __init__(self, input_dims, d_model=512, num_classes=10, num_layers=2):
        super(CrossAttentionClassifierWithDistillation, self).__init__()
        self.feature_mappers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
            for dim in input_dims
        ])
        self.cross_attention = MultiCrossAttention(d_model, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, feature_groups):
        transformed_features = [mapper(features) for mapper, features in zip(self.feature_mappers, feature_groups)]
        features_stacked = torch.stack(transformed_features, dim=1)
        # print(features_stacked.shape)
        fused_features, attention_weights_list = self.cross_attention(features_stacked)
        logits = self.classifier(fused_features)
        return fused_features, logits, attention_weights_list

# ------------------ 特征映射模块 ------------------
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.mapper = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        return self.mapper(features)

# 初始化特征映射模块
feature_mappers = [
    FeatureMapper(input_dim, 128).to(device) for input_dim in input_dims
]

# ------------------ 损失函数定义 ------------------
def feature_distillation_loss(student_features, expert_features_list, feature_mappers):
    # num_teachers = len(expert_features_list)
    # print(num_teachers)
    cosine_loss = 0
    smooth_l1_loss = 0
    for expert_feature, mapper in zip(expert_features_list, feature_mappers):
        mapped_expert_feature = mapper(expert_feature)
        cosine_loss += 1 - nn.functional.cosine_similarity(student_features, mapped_expert_feature, dim=-1).mean()
        smooth_l1_loss += nn.HuberLoss()(student_features, mapped_expert_feature)
        # delta = 1.0
        # error = student_features - mapped_expert_feature
        # abs_error = error.abs()

        # # 打印当前 batch 中的最大误差和是否超过 delta
        # max_error = abs_error.max().item()
        # print(f"Max absolute error: {max_error:.4f} (Delta: {delta})")
        # if max_error > delta:
        #     print("Some errors exceed delta.")
        # else:
        #     print("All errors are within delta.")
    return cosine_loss + smooth_l1_loss

# ------------------ 初始化模型 ------------------
model = CrossAttentionClassifierWithDistillation(
    input_dims, d_model=128, num_classes=num_classes, num_layers=5

).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

# ------------------ 训练流程 ------------------
num_epochs = 50
# alpha = 0.001
ls = 0.5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for *features, labels in train_loader:
        features = [f.to(device) for f in features]
        labels = labels.to(device)

        # 学生模型特征 & logits
        student_features, student_logits, _ = model(features)

        # 特征蒸馏损失
        distill_loss = feature_distillation_loss(student_features, features, feature_mappers)

        # print(student_logits.shape)
        # 任务损失
        task_loss = criterion(student_logits, labels)
        # print(student_logits)
        # 总损失
        loss = task_loss + ls * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证集
    model.eval()
    all_val_predictions = []
    all_val_labels = []

    with torch.no_grad():
        for *features, labels in val_loader:
            features = [f.to(device) for f in features]
            labels = labels.to(device)
            _, val_logits, _ = model(features)
            all_val_predictions.append(val_logits)
            all_val_labels.append(labels)

    all_val_predictions = torch.cat(all_val_predictions, dim=0)
    all_val_labels = torch.cat(all_val_labels, dim=0)

    # 计算验证集 Top-1 Accuracy, BA, F1
    val_top1_acc = calculate_topk_accuracy(all_val_predictions, all_val_labels, k=1)
    val_ba = balanced_accuracy_score(
        all_val_labels.cpu().numpy(),
        torch.argmax(all_val_predictions, dim=1).cpu().numpy()
    )
    val_f1 = f1_score(
        all_val_labels.cpu().numpy(),
        torch.argmax(all_val_predictions, dim=1).cpu().numpy(),
        average="weighted"
    )

    # 测试集
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for *features, labels in test_loader:
            features = [f.to(device) for f in features]
            labels = labels.to(device)
            _, test_logits, _ = model(features)
            all_test_predictions.append(test_logits)
            all_test_labels.append(labels)

    all_test_predictions = torch.cat(all_test_predictions, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    # 计算测试集 Top-1, Top-3, Top-5 Accuracy, BA, F1
    test_top1_acc = calculate_topk_accuracy(all_test_predictions, all_test_labels, k=1)
    test_top3_acc = calculate_topk_accuracy(all_test_predictions, all_test_labels, k=3)
    test_top5_acc = calculate_topk_accuracy(all_test_predictions, all_test_labels, k=5)
    test_ba = balanced_accuracy_score(
        all_test_labels.cpu().numpy(),
        torch.argmax(all_test_predictions, dim=1).cpu().numpy()
    )
    test_f1 = f1_score(
        all_test_labels.cpu().numpy(),
        torch.argmax(all_test_predictions, dim=1).cpu().numpy(),
        average="weighted"
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {total_loss:.4f}, "
        f"Val BA: {val_ba:.4f}, Val F1: {val_f1:.4f}, "
        f"Test Top-1: {test_top1_acc:.4f}, Test Top-3: {test_top3_acc:.4f}, Test Top-5: {test_top5_acc:.4f}, Test BA: {test_ba:.4f}, Test F1: {test_f1:.4f}"
    )

    # 早停机制
    if early_stopping(val_ba, val_f1, model, epoch + 1):
        print(f"Early stopping at epoch {epoch + 1}")
        break