import torch

# 加载 .pt 文件（路径替换为你自己的）
pt_file = torch.load("/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/CLAM/Features/UNI_features/pt_files/4242.pt")

# 查看 keys
# print(pt_file.keys())  # 输出通常是 dict_keys(['features', 'coordinates'])

# 查看维度
print("Features shape:", pt_file.shape)       # 如 (N, 1024)
# print("Coordinates shape:", pt_file['coordinates'].shape) # 如 (N, 2)

# 示例：查看前 5 个 patch 的特征
print(pt_file[:5])

# 示例：查看前 5 个 patch 的坐标（在原图中的位置）
# print(pt_file['coordinates'][:5])
