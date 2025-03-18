import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

class CRCDataset(Dataset):
    def __init__(self, root_dir, csv_dir, transform=None):
        """
        Args:
            root_dir (string): 图像文件的根目录路径
            csv_dir (string): CSV 文件的根目录路径
            transform (callable, optional): 可选的图像变换
        """
        self.root_dir = root_dir
        self.csv_dir = csv_dir
        self.transform = transform
        self.data = []

        # 遍历所有 CSV 文件
        for folder_id in range(1, 201):  # 文件夹从 001 到 200
            folder_name = f"{folder_id:03d}"  # 格式化为三位数字
            csv_path = os.path.join(csv_dir, f"{folder_name}_labels.csv")
            
            if os.path.exists(csv_path):
                # 加载 CSV 文件
                csv_data = pd.read_csv(csv_path, sep=",")  # 根据文件分隔符修改
                
                csv_data['folder'] = folder_name  # 添加文件夹名称作为元数据
                self.data.append(csv_data)
                

        # 合并所有数据
        if self.data:
            self.data = pd.concat(self.data, ignore_index=True)
        else:
            raise ValueError("No valid CSV files found to concatenate.")
        label_columns = [
            "highgrade_dysplasia", "adenocarcinoma", "suspicious_for_invasion",
            "inflammation", "resection_edge", "tumor_necrosis",
            "artifact", "normal", "lowgrade_dysplasia"
        ]

        # 剔除无效标签的样本（多个类别为 1 或没有任何类别为 1）
        self.data = self.data[self.data[label_columns].sum(axis=1) == 1]
        self.label_columns = label_columns 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取当前行的数据
        row = self.data.iloc[idx]
        folder_name = row['folder']  # 文件夹名，例如 '001'
        fname = os.path.basename(row['fname'])  # 提取文件名，例如 '0.jpg'
        img_path = os.path.join(self.root_dir, folder_name, fname)  # 拼接完整路径
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} does not exist. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # 尝试加载下一个样本

        label = row[self.label_columns].tolist()
        label = label.index(1)

        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: {img_path} not found.")
            return None

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, label
# if __name__ == "__main__":
#     # 定义 transform
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),                # 调整图像大小为 224x224
#         transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
#         transforms.RandomRotation(10),               # 随机旋转 10 度
#         transforms.ToTensor(),                        # 转换为张量，并归一化到 [0, 1]
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # 数据路径
#     root_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"
#     csv_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"

#     # 检查每个 CSV 文件
#     invalid_csv_files = []  # 用于记录包含全零标签的文件
#     invalid_rows = []  # 用于记录具体问题行的 fname 和文件名

#     for folder_id in range(1, 201):  # 文件夹从 001 到 200
#         folder_name = f"{folder_id:03d}"  # 格式化为三位数字
#         csv_path = os.path.join(csv_dir, f"{folder_name}_labels.csv")

#         if os.path.exists(csv_path):
#             # 加载 CSV 文件
#             csv_data = pd.read_csv(csv_path, sep=",")
            
#             for idx, row in csv_data.iterrows():
#                 # 提取标签列并检查是否全为 0
#                 label = [
#                     int(row["highgrade_dysplasia"]),
#                     int(row["adenocarcinoma"]),
#                     int(row["suspicious_for_invasion"]),
#                     int(row["inflammation"]),
#                     int(row["resection_edge"]),
#                     int(row["tumor_necrosis"]),
#                     int(row["artifact"]),
#                     int(row["normal"]),
#                     int(row["lowgrade_dysplasia"])
#                 ]
#                 if sum(label) == 0:  # 如果标签全为 0
#                     invalid_csv_files.append(csv_path)
#                     invalid_rows.append({
#                         "csv_file": csv_path,
#                         "fname": row["fname"],
#                         "label": label
#                     })
#         else:
#             print(f"Warning: {csv_path} does not exist.")



if __name__ == "__main__":
    # 定义 transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                # 调整图像大小为 224x224
        transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
        transforms.RandomRotation(10),               # 随机旋转 10 度
        transforms.ToTensor(),                        # 转换为张量，并归一化到 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                             std=[0.229, 0.224, 0.225])
    ])

    # 数据路径
    root_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"
    csv_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"
    print(1)
    # 创建数据集
    dataset = CRCDataset(root_dir=root_dir, csv_dir=csv_dir, transform=transform)
   
    print(1)
    # 打印检查结果
    print(f"Total samples: {len(dataset)}")
    

    

    # 数据集划分
    # train_ratio = 151 / 200  # 按 151:49 划分
    # print(len(dataset))
    # train_size = int(len(dataset) * train_ratio)
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_folders = [f"{i:03d}" for i in range(1, 152)]  # 前151个病例
    test_folders = [f"{i:03d}" for i in range(152, 201)]  # 后49个病例

    train_data = dataset.data[dataset.data['folder'].isin(train_folders)]
    test_data = dataset.data[dataset.data['folder'].isin(test_folders)]

    print(f"Number of training cases: {len(train_folders)}, training samples: {len(train_data)}")
    print(f"Number of testing cases: {len(test_folders)}, testing samples: {len(test_data)}")

    # 创建训练集和测试集
    train_dataset = CRCDataset(root_dir=root_dir, csv_dir=csv_dir, transform=transform)
    train_dataset.data = train_data.reset_index(drop=True)

    test_dataset = CRCDataset(root_dir=root_dir, csv_dir=csv_dir, transform=transform)
    test_dataset.data = test_data.reset_index(drop=True)
    # Test dataset length and a sample
    print(f"Number of training samples: {len(train_dataset)}")
    

    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    print(f"Number of test samples: {len(test_dataset)}")
    img, label = test_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")

    # 遍历测试集并打印每个样本的索引、图像形状和标签
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        print(f"Sample {idx}: Image shape: {img.shape}, Label: {label}")
    # # 验证测试集中的每个索引
    # invalid_samples = []  # 用于记录无效样本的索引
    # print("Validating test dataset...")
    # for idx in range(len(test_dataset)):
    #     try:
    #         img, label = test_dataset[idx]
    #         if img is None or label is None:
    #             print(f"Invalid sample at index {idx}: Missing image or label.")
    #             invalid_samples.append(idx)
    #         else:
    #             print(f"Sample {idx} loaded successfully: Image shape: {img.shape}, Label: {label}")
    #     except Exception as e:
    #         print(f"Error at index {idx}: {e}")
    #         invalid_samples.append(idx)

    # if invalid_samples:
    #     print(f"Found {len(invalid_samples)} invalid samples in the test dataset: {invalid_samples}")
    # else:
    #     print("All test samples are valid.")