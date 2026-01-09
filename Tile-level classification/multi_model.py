import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
# dist.init_process_group(backend='nccl')
# =============== 1. 固定随机种子 ===============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 模型和结果保存路径 ----
model_save_path = "/nas/leiwenhui/thr_shazam/shazam/multi_model_test4CCRCC/moe_mlp_model.pth"
result_save_path = "/nas/leiwenhui/thr_shazam/shazam/multi_model_test4CCRCC/moe_mlp_results.json"

# =============== 2. 数据加载 (示例路径) ===============
train_low1, train_mid1, train_high1, train_labels = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/virchow2_train_features.pt",
    weights_only=True
)
train_low2, train_mid2, train_high2, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/uni_v2_train_features.pt",
    weights_only=True
)
train_low3, train_mid3, train_high3, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/phikon_v2_train_features.pt",
    weights_only=True
)
train_low4, train_mid4, train_high4, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/gigapath_train_features.pt",
    weights_only=True
)
train_low5, train_mid5, train_high5, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/hoptimus1_train_features.pt",
    weights_only=True
)

# for PCAM is _valid_, for else just _test_
val_low1, val_mid1, val_high1, val_labels = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/virchow2_test_features.pt",
    weights_only=True
)
val_low2, val_mid2, val_high2, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/uni_v2_test_features.pt",
    weights_only=True
)
val_low3, val_mid3, val_high3, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/phikon_v2_test_features.pt",
    weights_only=True
)
val_low4, val_mid4, val_high4, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/gigapath_test_features.pt",
    weights_only=True
)
val_low5, val_mid5, val_high5, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/hoptimus1_test_features.pt",
    weights_only=True
)

test_low1, test_mid1, test_high1, test_labels = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/virchow2_test_features.pt",
    weights_only=True
)
test_low2, test_mid2, test_high2, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/uni_v2_test_features.pt",
    weights_only=True
)
test_low3, test_mid3, test_high3, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/phikon_v2_test_features.pt",
    weights_only=True
)
test_low4, test_mid4, test_high4, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/gigapath_test_features.pt",
    weights_only=True
)
test_low5, test_mid5, test_high5, _ = torch.load(
    "/nas/share/Extracted_Feature/multi_features4CCRCC/hoptimus1_test_features.pt",
    weights_only=True
)


num_classes = len(torch.unique(train_labels))
print(f"Number of classes: {num_classes}")
# =============== 3. 构建 Dataset + DataLoader ===============
train_dataset = TensorDataset(
    train_low1,  train_mid1,  train_high1,
    train_low2,  train_mid2,  train_high2,
    train_low3,  train_mid3,  train_high3,
    train_low4,  train_mid4,  train_high4,
    train_low5,  train_mid5,  train_high5,
    train_labels
)

val_dataset = TensorDataset(
    val_low1,  val_mid1,  val_high1,
    val_low2,  val_mid2,  val_high2,
    val_low3,  val_mid3,  val_high3,
    val_low4,  val_mid4,  val_high4,
    val_low5,  val_mid5,  val_high5,
    val_labels
)

test_dataset = TensorDataset(
    test_low1,  test_mid1,  test_high1,
    test_low2,  test_mid2,  test_high2,
    test_low3,  test_mid3,  test_high3,
    test_low4,  test_mid4,  test_high4,
    test_low5,  test_mid5,  test_high5,
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


# =============== 4. Top-K Accuracy 与 早停机制 ===============
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


# =============== 5. 定义单层 Cross-Attention 和 4 层封装 ===============
import numpy as np

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, features):
        """
        features: [batch_size, n_tokens, d_model]
        """
        queries = self.query(features)
        keys    = self.key(features)
        values  = self.value(features)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(keys.size(-1))
        attn_weights = self.softmax(attention_scores)    # [B, n_tokens, n_tokens]

        attended_features = torch.matmul(attn_weights, values)
        out = self.output_layer(attended_features)
        out = self.layernorm(features + out)
        return out, attn_weights

class MultiCrossAttentionLayers(nn.Module):
    def __init__(self, d_model,num_layers=5):
        super(MultiCrossAttentionLayers, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model) for _ in range(num_layers)])

    def forward(self, features):
        """
        features: [batch_size, n_tokens, d_model]
        返回 fused_features: [batch_size, d_model]
        """
        all_attn_weights = []
        for layer in self.layers:
            features, attn_w = layer(features)
            all_attn_weights.append(attn_w)
        # mean pooling over token dimension
        fused_features = features.mean(dim=1)
        return fused_features, all_attn_weights


# =============== 6. 定义一个 MoE（单层级）: 只用 1 个 gating 管理 5 个教师 ===============
class MoEOnePerLevel(nn.Module):
    """
    - 对同一层级 5 个教师的特征 (各自[batch, in_dim]) 做线性投影 => 5个 [B, d_model]
    - 再拼接 5 个投影为 [B, 5*d_model]，用一个 gating 网络得 [B,5] 的权重
    - 将权重乘回每个投影后做 LayerNorm
    - 最终输出 [B,5,d_model]
    """

    def __init__(self, in_dims, d_model=128):
        super().__init__()
        assert len(in_dims) == 5

        self.d_model = d_model

        # 1) 分别投影 5 位教师 => d_model
        self.proj_list = nn.ModuleList([
            nn.Linear(in_dim, d_model) for in_dim in in_dims
        ])

        # 2) gating 网络：输入 [B, 5*d_model] => 输出 [B,5]
        # self.gate = nn.Linear(5 * d_model, 5)
        self.gate = nn.Sequential(
            nn.Linear(5*d_model, 128),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 5)
        )

        # 3) LayerNorm (可根据需求再添加激活/Dropout等)
        self.ln = nn.LayerNorm(d_model)
        

    def forward(self, f1, f2, f3, f4, f5):
        """
        f1,f2,f3,f4,f5 : [B, in_dim_i]，5 位教师同层级特征
        返回: [B,5,d_model]
        """
        # 投影到 d_model
        p1 = self.proj_list[0](f1)
        p2 = self.proj_list[1](f2)
        p3 = self.proj_list[2](f3)
        p4 = self.proj_list[3](f4)
        p5 = self.proj_list[4](f5)
        cat_p = torch.cat([p1, p2, p3, p4, p5], dim=1)  # [B,5*d_model]
        gating_logits = self.gate(cat_p)           # => [B,5]
        gating_weights= torch.softmax(gating_logits, dim=1)  # => [B,5]

        # 每个教师的投影乘相应权重
        w1 = gating_weights[:, 0].unsqueeze(-1)  # [B,1]
        w2 = gating_weights[:, 1].unsqueeze(-1)
        w3 = gating_weights[:, 2].unsqueeze(-1)
        w4 = gating_weights[:, 3].unsqueeze(-1)
        w5 = gating_weights[:, 4].unsqueeze(-1)

        p1 = self.ln(p1 * w1)
        p2 = self.ln(p2 * w2)
        p3 = self.ln(p3 * w3)
        p4 = self.ln(p4 * w4)
        p5 = self.ln(p5 * w5)

        # 堆叠回 [B,5,d_model] 供后续 cross-attention
        out = torch.stack([p1, p2, p3, p4, p5], dim=1)
        return out

# ------------------ 特征映射模块 ------------------
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.mapper = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        return self.mapper(features)



# =============== 7. 学生模型 12 层 (3段 × 4层 Cross-Attention)，每层级 1 个 MoE ===============
class StudentModel15Layers(nn.Module):
    """
    low/mid/high 各层级用 1 个 MoEOnePerLevel 来处理 5 位教师的特征。
    """
    def __init__(self, 
                 dim_list_low,  # 5 个教师 low 特征维度
                 dim_list_mid,  # 5 个教师 mid 特征维度
                 dim_list_high, # 5 个教师 high 特征维度
                 d_model=128,
                 num_classes=10,
                 num_layers=5):
        super().__init__()
        assert len(dim_list_low)  == 5
        assert len(dim_list_mid)  == 5
        assert len(dim_list_high) == 5
        # assert num_layers in [5, 10, 15], "支持 5/10/15 层示例"

        self.num_layers = num_layers
        self.d_model = d_model

        # =========== 1) 每个层级只用 1 个 MoE ===========
        self.moe_low  = MoEOnePerLevel(dim_list_low,  d_model=d_model)
        self.moe_mid  = MoEOnePerLevel(dim_list_mid,  d_model=d_model)
        self.moe_high = MoEOnePerLevel(dim_list_high, d_model=d_model)
        # === Trainable Mapper for Distillation ===
        self.mapper_low  = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_low])
        self.mapper_mid  = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_mid])
        self.mapper_high = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_high])

        # =========== 2) 三段 cross-attention，每段 4 层 ===========
        self.segment1 = MultiCrossAttentionLayers(d_model,4)
        self.segment2 = MultiCrossAttentionLayers(d_model,4)
        self.segment3 = MultiCrossAttentionLayers(d_model,4)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

        # =========== LayerNorm for residual ===========
        self.res_ln2 = nn.LayerNorm(d_model)
        self.res_ln3 = nn.LayerNorm(d_model)
        # =========== 3) 最终分类器 ===========
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, batch):
        """
        batch: (15个特征 + 1 label), 但这里只取前15个特征
        对应：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        """
        low1, mid1, high1 = batch[0],  batch[1],  batch[2]
        low2, mid2, high2 = batch[3],  batch[4],  batch[5]
        low3, mid3, high3 = batch[6],  batch[7],  batch[8]
        low4, mid4, high4 = batch[9],  batch[10], batch[11]
        low5, mid5, high5 = batch[12],  batch[13], batch[14]

        # ---------------- Segment1 (low) ----------------
        seg1_input = self.moe_low(low1, low2, low3, low4, low5)   # => [B,4,d_model]
        out1, attn1 = self.segment1(seg1_input)             # => [B,d_model]
        # if self.num_layers == 5:
        #     logits = self.classifier(out1)
        #     return out1, logits, attn1

        # ---------------- Segment2 (mid) ----------------
        # 把 out1 当作一个 token，然后加上 [B,4,d_model] 的 mid token => [B,5,d_model]
        seg2_mid = self.moe_mid(mid1, mid2, mid3, mid4, mid5)   # => [B,4,d_model]
        
        out2, attn2 = self.segment2(seg2_mid)             # => [B,d_model]
        # if self.num_layers == 10:
        #     logits = self.classifier(out2)
        #     return out2, logits, (attn1 + attn2)
        out2 = self.res_ln2(out2 + self.alpha * out1)
        # ---------------- Segment3 (high) ----------------
        seg3_high = self.moe_high(high1, high2, high3, high4, high5)  # => [B,4,d_model]
        # seg3_input = torch.cat([out2.unsqueeze(1), seg3_high], dim=1)
        out3, attn3 = self.segment3(seg3_high)                  # => [B,d_model]
        out3 = self.res_ln3(out3 + self.beta * out2)
        logits = self.classifier(out3)
        
        return out1,out2,out3, logits, (attn1 + attn2 + attn3)


# # =============== 8. 多层蒸馏：若需要，可保留原逻辑 (可选) ===============
# def forward_all_segments(model, batch):
#     """
#     一次性运行 3 个 segment, 拿到 out1/out2/out3，方便多层蒸馏。
#     如果不需要多层蒸馏，可直接 model(batch) 拿最终输出。
#     """

#     low1, mid1, high1 = batch[0],  batch[1],  batch[2]
#     low2, mid2, high2 = batch[3],  batch[4],  batch[5]
#     low3, mid3, high3 = batch[6],  batch[7],  batch[8]
#     low4, mid4, high4 = batch[9],  batch[10], batch[11]

#     # ---------- Segment1 ----------
#     seg1_input = model.moe_low(low1, low2, low3, low4)  # => [B,4,d_model]
#     out1, _ = model.segment1(seg1_input)

#     # ---------- Segment2 ----------
#     seg2_mid_4 = model.moe_mid(mid1, mid2, mid3, mid4)  # => [B,4,d_model]
#     seg2_input = torch.cat([out1.unsqueeze(1), seg2_mid_4], dim=1)
#     out2, _ = model.segment2(seg2_input)

#     # ---------- Segment3 ----------
#     seg3_high_4 = model.moe_high(high1, high2, high3, high4)  # => [B,4,d_model]
#     seg3_input = torch.cat([out2.unsqueeze(1), seg3_high_4], dim=1)
#     out3, _ = model.segment3(seg3_input)

#     return out1, out2, out3


def distill_pair(student_feat, teacher_feat):
    """
    一个简单的示例蒸馏损失: (1 - cos_sim) + SmoothL1
    student_feat: [B, d_model]
    teacher_feat: [B, d_model]
    """
    cos_term = 1.0 - nn.functional.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
    smooth_l1= nn.HuberLoss()(student_feat, teacher_feat)
    return cos_term + smooth_l1

def multi_level_distillation_loss(
    out1, out2, out3,
    batch,  # 5 位教师 low/mid/high
    num_layers,
    model
):
    """
    如果你想让学生的 out1 对齐 5个教师的 low（投影后），
    out2 对齐 mid，out3 对齐 high，按需加和。
    这里只做示例，可以根据你的实验需求做更多设计。
    """
    low1, mid1, high1 = batch[0],  batch[1],  batch[2]
    low2, mid2, high2 = batch[3],  batch[4],  batch[5]
    low3, mid3, high3 = batch[6],  batch[7],  batch[8]
    low4, mid4, high4 = batch[9],  batch[10], batch[11]
    low5, mid5, high5 = batch[12],  batch[13], batch[14]

    total_loss = 0.0

    if num_layers >= 4:
        # 用 moe_low 处理 4个 low 后, 做 mean or sum 之类 => teacher 的“融合”
        for i,teacher_low in enumerate([low1, low2, low3, low4, low5]):
            teacher_low = model.mapper_low[i](teacher_low)          # => [B,d_model]
            total_loss += distill_pair(out1, teacher_low)

    if num_layers >= 8:
        for i,teacher_mid in enumerate([mid1, mid2, mid3, mid4, mid5]):
            teacher_mid = model.mapper_mid[i](teacher_mid)
            total_loss += distill_pair(out2, teacher_mid)
        
    if num_layers == 12:
        for i,teacher_high in enumerate([high1, high2, high3, high4, high5]):
            teacher_high = model.mapper_high[i](teacher_high)
            total_loss += distill_pair(out3, teacher_high)

    return total_loss


# =============== 9. 启动训练 ===============
# 获取各层级输入维度
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print(torch.cuda.is_available())
dim_low_list  = [
    train_low1.shape[1],
    train_low2.shape[1],
    train_low3.shape[1],
    train_low4.shape[1],
    train_low5.shape[1]
]
dim_mid_list  = [
    train_mid1.shape[1],
    train_mid2.shape[1],
    train_mid3.shape[1],
    train_mid4.shape[1],
    train_mid5.shape[1]
]
dim_high_list = [
    train_high1.shape[1],
    train_high2.shape[1],
    train_high3.shape[1],
    train_high4.shape[1],
    train_high5.shape[1]
]

num_layers_choice = 12 # 3 segmentations * 4 layers
model = StudentModel15Layers(
    dim_list_low  = dim_low_list,
    dim_list_mid  = dim_mid_list,
    dim_list_high = dim_high_list,
    d_model=128,
    num_classes=num_classes,
    num_layers=num_layers_choice
).to(device)
# model = DDP(model.to(device))
# model = nn.DataParallel(model)  
# model.to(device)
for name, param in model.named_parameters():
    print(name, param.device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
# early_stopping = EarlyStopping(patience=100, min_delta=1e-4, lambda_ba=0.5, lambda_f1=0.5)

lambda_distill = 0.2
num_epochs = 100
# lambda_list = [0.01,0.03,0.05,0.08,0.1,0.15 ,0.2, 0.3]
lambda_list = [0.01]

for lambda_distill in lambda_list:
    early_stopping = EarlyStopping(patience=30, min_delta=1e-4, lambda_ba=0.5, lambda_f1=0.5,
                                   model_path=model_save_path, result_path=result_save_path)
    print(f"\n=== Training with lambda_distill = {lambda_distill} ===\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
        for batch in train_pbar:
            features = [b.to(device) for b in batch[:-1]]  # 15个特征
            labels   = batch[-1].to(device)               # label

            # 获取 out1/out2/out3
            out1,out2,out3, logits, attn = model(features)

            task_loss = criterion(logits, labels)

            distill_l = multi_level_distillation_loss(
                out1, out2, out3, features, model.num_layers, model
            )

            # distill_weight = max(0.1, 0.5 - epoch * 0.01)
            # total_loss = classification_loss + distill_weight * distill_loss

            loss = task_loss + lambda_distill * distill_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
        # ========== 验证集评估 ==========
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_pbar:
                features = [b.to(device) for b in batch[:-1]]
                labels   = batch[-1].to(device)

                out1,out2,out3, logits, attn = model(features)

                # if model.num_layers == 5:
                #     student_out = out1
                # elif model.num_layers == 10:
                #     student_out = out2
                # else:
                #     student_out = out3

                # logits = model.classifier(student_out)
                val_preds.append(logits)
                val_labels.append(labels)

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
            for batch in test_pbar:
                features = [b.to(device) for b in batch[:-1]]
                labels   = batch[-1].to(device)

                out1,out2,out3, logits, attn = model(features)

                # if model.num_layers == 5:
                #     student_out = out1
                # elif model.num_layers == 10:
                #     student_out = out2
                # else:
                #     student_out = out3

                # logits = model.classifier(student_out)
                test_preds.append(logits)
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
            # f"Val Acc: {val_top1_acc:.4f}, Val BA: {val_ba:.4f}, Val F1: {val_f1:.4f}, "
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
        # if early_stopping(valid_ba, valid_f1, metrics_dict, model, epoch + 1):
        if early_stopping(test_ba, test_f1, metrics_dict, model, epoch + 1):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    # 如果需要恢复最佳模型:
    best_epoch = early_stopping.best_epoch
    # model.load_state_dict(early_stopping.best_model_state)
    print(f"Loaded best model from epoch {best_epoch}")
