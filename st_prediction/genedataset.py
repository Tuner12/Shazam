import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class GeneExpressionJSONLDataset(Dataset):
    def __init__(self, metadata_file, image_folder, transform=None):
        """
        Args:
            metadata_file (str): metadata.jsonl 文件路径，每行一个 JSON 对象，包含 "file_name" 和 "label"。
            image_folder (str): 图像存放目录。
            transform (callable, optional): 图像预处理转换。
        """
        self.image_folder = image_folder
        self.transform = transform or transforms.ToTensor()
        self.entries = []
        # 读取 metadata.jsonl
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.entries.append(data)

        # 从第一条 label 构建基因顺序（保证一致）
        first_label = self.entries[0]['label']
        # "VHL:0, PBRM1:0, FH:0, MET:1, ..."
        parts = [kv.strip() for kv in first_label.split(',')]
        self.gene_names = [kv.split(':', 1)[0].strip() for kv in parts]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # 1. 读取并处理图像
        img_path = os.path.join(self.image_folder, entry['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 2. 拆分 label 字符串为数值列表
        label_str = entry['label']
        parts = [kv.strip() for kv in label_str.split(',')]
        values = [float(kv.split(':', 1)[1]) for kv in parts]

        # 3. 转为 tensor
        label = torch.tensor(values, dtype=torch.float32)

        return image, label
