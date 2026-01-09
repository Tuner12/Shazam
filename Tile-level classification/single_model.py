import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import random
from tqdm import tqdm
import json

# from multi_moe_distill_test import EarlyStopping, calculate_topk_accuracy
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ------------------ 数据加载与拼接 ------------------
_, _, train_features, train_labels = torch.load("/nas/share/Extracted_Feature/multi_features4CRCMSI/phikon_v2_train_features.pt")
_, _, val_features, val_labels = torch.load("/nas/share/Extracted_Feature/multi_features4CRCMSI/phikon_v2_test_features.pt") # for PCAM is _valid_
_, _, test_features, test_labels = torch.load("/nas/share/Extracted_Feature/multi_features4CRCMSI/phikon_v2_test_features.pt")

train_features = torch.tensor(train_features, dtype=torch.float32)
val_features = torch.tensor(val_features, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.float32)

train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

print(f"拼接后的训练特征形状: {train_features.shape}")
print(f"拼接后的验证特征形状: {val_features.shape}")
print(f"拼接后的测试特征形状: {test_features.shape}")

# ---- 模型和结果保存路径 ----
model_save_path = "/nas/leiwenhui/thr_shazam/shazam/single_model_test4CRCMSI/phikon_v2_mlp_model.pth"
result_save_path = "/nas/leiwenhui/thr_shazam/shazam/single_model_test4CRCMSI/phikon_v2_mlp_results.json"

# ------------------ 定义MLP模型 ------------------
class MLPFeatureClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super(MLPFeatureClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 加入 LayerNorm
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 加入 LayerNorm
            nn.GELU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.mlp(x)
        predictions = self.classifier(features)
        return predictions

def calculate_topk_accuracy(predictions, labels, k):
    topk_indices = torch.topk(predictions, k, dim=1).indices
    correct = sum(labels[i].item() in topk_indices[i] for i in range(len(labels)))
    return correct / len(labels)

class EarlyStopping:
    def __init__(
        self,
        patience=20,
        min_delta=1e-4,
        lambda_ba=0.8,
        lambda_f1=0.2,
        model_path=None,
        result_path=None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.lambda_ba = lambda_ba
        self.lambda_f1 = lambda_f1
        self.model_path = model_path          # 新增
        self.result_path = result_path        # 新增
        self.best_score = 0
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_ba, val_f1, metrics_dict, model, epoch):
        validation_score = self.lambda_ba * val_ba + self.lambda_f1 * val_f1
        if validation_score > self.best_score + self.min_delta:
            # 更新最佳
            self.best_score = validation_score
            self.best_epoch = epoch
            # ─── 保存最佳模型 ───
            if self.model_path is not None:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(model.state_dict(), self.model_path)
            # ─── 保存对应指标 ───
            if self.result_path is not None and metrics_dict is not None:
                with open(self.result_path, "w") as f:
                    json.dump(metrics_dict, f, indent=2)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# 初始化模型
input_dim = train_features.shape[1]
num_classes = len(np.unique(train_labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPFeatureClassifier(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
model.to(device)
print(f"模型初始化：输入维度 {input_dim}, 隐藏层维度 128, 类别数 {num_classes}")
print(f"The model is on device: {next(model.parameters()).device}")
# ------------------ 损失函数和优化器 ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------ 数据加载器 ------------------
batch_size = 64
train_dataset = TensorDataset(
    train_features,
    train_labels
)

val_dataset = TensorDataset(
    val_features,
    val_labels
)

test_dataset = TensorDataset(
    test_features,
    test_labels
)

# val_size = int(0.125 * len(train_dataset))
# train_size = len(train_dataset) - val_size
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# train_loader   = DataLoader(train_dataset,   batch_size=64, shuffle=True)
# val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_loader   = DataLoader(train_dataset,   batch_size=64, shuffle=True)
val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# =============== 9. 启动训练 ===============
num_epochs = 100
early_stopping = EarlyStopping(patience=10, min_delta=1e-4, lambda_ba=0.5, lambda_f1=0.5,
                            model_path=model_save_path,
                            result_path=result_save_path)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
    for features, labels in train_pbar:
        features = features.to(device)
        labels = labels.to(device)              # label

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # ========== 验证集评估 ==========
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        for features, labels in val_pbar:
            features = features.to(device)
            labels = labels.to(device)              # label

            predictions = model(features)
            val_preds.append(predictions)
            val_labels.append(labels)

    # 拼接测试集预测和标签
    val_preds = torch.cat(val_preds, dim=0)
    val_labels= torch.cat(val_labels, dim=0)
    val_top1_acc = calculate_topk_accuracy(val_preds, val_labels, k=1)
    val_ba = balanced_accuracy_score(
        val_labels.cpu().numpy(),
        torch.argmax(val_preds, dim=1).cpu().numpy()
    )
    val_f1 = f1_score(
        val_labels.cpu().numpy(),
        torch.argmax(val_preds, dim=1).cpu().numpy(),
        average="weighted"
    )

    # ========== 测试集评估 ==========
    test_preds, test_labels_all = [], []
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for features, labels in test_pbar:
            features = features.to(device)
            labels = labels.to(device)              # label

            predictions = model(features)

            test_preds.append(predictions)
            test_labels_all.append(labels)

    test_preds = torch.cat(test_preds, dim=0)
    test_labels_all = torch.cat(test_labels_all, dim=0)

    test_top1_acc = calculate_topk_accuracy(test_preds, test_labels_all, k=1)
    # test_top3_acc = calculate_topk_accuracy(test_preds, test_labels_all, k=3)
    # test_top5_acc = calculate_topk_accuracy(test_preds, test_labels_all, k=5)
    test_ba = balanced_accuracy_score(
        test_labels_all.cpu().numpy(),
        torch.argmax(test_preds, dim=1).cpu().numpy()
    )
    test_f1 = f1_score(
        test_labels_all.cpu().numpy(),
        torch.argmax(test_preds, dim=1).cpu().numpy(),
        average="weighted"
    )
    ave_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {ave_loss:.4f}, "
        f"Val Acc: {val_top1_acc:.4f}, Val BA: {val_ba:.4f}, Val F1: {val_f1:.4f}, "
        f"Test Top1: {test_top1_acc:.4f}, "
        f"Test BA: {test_ba:.4f}, Test F1: {test_f1:.4f}"
    )

    metrics_dict = {
        "epoch": epoch + 1,
        "train_loss": ave_loss,
        "val_top1_acc": val_top1_acc,
        "val_ba": val_ba,
        "val_f1": val_f1,
        "val_labels": val_labels.cpu().numpy().tolist(),
        "val_preds": val_preds.cpu().numpy().tolist(),
        "test_top1_acc": test_top1_acc,
        "test_ba": test_ba,
        "test_f1": test_f1,
        "test_labels": test_labels_all.cpu().numpy().tolist(),
        "test_preds": test_preds.cpu().numpy().tolist()
        }
    
    if early_stopping(val_ba, val_f1, metrics_dict, model, epoch + 1):
        print(f"Early stopping at epoch {epoch + 1}")
        break

# 如果需要恢复最佳模型:
best_epoch = early_stopping.best_epoch
# model.load_state_dict(early_stopping.best_model_state)
print(f"Best model saved from epoch {early_stopping.best_epoch} "
      f"→ {model_save_path}")
print(f"Corresponding metrics stored at {result_save_path}")

