import os
from functools import partial
import timm
import torch
# from utils.constants import MODEL2CONSTANTS
# from utils.transform_utils import get_eval_transforms
from torch.utils.data import DataLoader,random_split
from transformers import AutoModel
from torch.nn import Module
from torchvision import transforms
from dataset import ImageDataset
from dataset4huncrc import CRCDataset
from tqdm import tqdm 
def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
class Virchow2Wrapper(torch.nn.Module):
    """
    A wrapper to process the output of the Virchow2 model.
    The original model output is (1, 261, 1280):
      - output[:, 0] is the class token (1 x 1280)
      - output[:, 1:5] are register tokens (ignored)
      - output[:, 5:] are patch tokens (256 tokens, each 1280-d)
    We will produce a final embedding by concatenating:
      class_token: (1 x 1280)
      mean pooled patch tokens: (1 x 1280)
    Final embedding size: (1 x 2560)
    """
    def __init__(self, model):
        super(Virchow2Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # model output: (B, 261, 1280)
        output = self.model(x)
        class_token = output[:, 0]       # (B, 1280)
        # patch_tokens = output[:, 5:]     # (B, 256, 1280)
        # patch_mean = patch_tokens.mean(dim=1)  # (B, 1280)
        # embedding = torch.cat([class_token, patch_mean], dim=-1)  # (B, 2560)
        return class_token
    
class PhikonWrapper(torch.nn.Module):
    """
    A wrapper for phikon-v2 model.
    The output of phikon-v2 (AutoModel) is a standard Transformer output:
    outputs.last_hidden_state: (B, seq_len, hidden_dim)
    We take the [CLS] token at index 0 for final representation, shape: (B, 1024).
    """
    def __init__(self, model):
        super(PhikonWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            features = outputs.last_hidden_state[:, 0, :]  # (B, 1024)
        return features
    

def get_encoder(model_name):
    print('loading model checkpoint')
    if model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
        # assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name =='virchow2':
        from timm.layers import SwiGLUPacked
        base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = Virchow2Wrapper(base_model)
        model.eval()
    elif model_name =='virchow':
        from timm.layers import SwiGLUPacked
        base_model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = Virchow2Wrapper(base_model)
        model.eval()
    elif model_name =='virchow2':
        from timm.layers import SwiGLUPacked
        base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = Virchow2Wrapper(base_model)
        model.eval()
    elif model_name == 'phikon_v2':
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained("owkin/phikon-v2")
        base_model.eval()
        model = PhikonWrapper(base_model)
    
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    return model


# Feature extraction function
def extract_features(model_name, data_loader, save_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_encoder(model_name).to(device)
    all_features, all_labels = [], []
    print(f"Extracting features using {model_name} on device: {device}")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            features = model(images)  # Extract features
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all extracted features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    torch.save((all_features, all_labels), save_path)
    print(f"Features saved to {save_path}")
    loaded_features, loaded_labels = torch.load(save_path)
    print(f"Shape of features tensor: {loaded_features.shape}")
    print(f"Shape of labels tensor: {loaded_labels.shape}")

# Main: Define DataLoader and Extract Features
if __name__ == "__main__":
    # Paths and configurations
    # root_dir = '/ailab/public/pjlab-smarthealth03/leiwenhui/Data/unitho/800'
    # train_csv = os.path.join(root_dir, 'train.csv')
    # test_csv = os.path.join(root_dir, 'test.csv')
    batch_size = 16

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    root_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"
    csv_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/Data/huncrc"

    # 创建数据集
    dataset = CRCDataset(root_dir=root_dir, csv_dir=csv_dir, transform=transform)

    # 数据集划分
    # train_ratio = 151 / 200  # 按 151:49 划分
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
    # DataLoader for train and test sets
    # train_dataset = ImageDataset(train_csv, root_dir, transform=transform)
    # test_dataset = ImageDataset(test_csv, root_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define the folder to store extracted features
    output_dir = "features4HunCRC"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Extract features for all modelsprint(f"Extracting features using {model_name}")
    
    for model_name in ['uni_v1', 'virchow','virchow2', 'phikon_v2', 'gigapath']:
        print(f"Extracting features using {model_name}")
        train_save_path = os.path.join(output_dir, f"{model_name}_train_features.pt")
        test_save_path = os.path.join(output_dir, f"{model_name}_test_features.pt")

        extract_features(model_name, train_loader, train_save_path, device='cuda')
        extract_features(model_name, test_loader, test_save_path, device='cuda')
        # Data type: <class 'tuple'>
        # Number of elements: 2
        # Element 0: Tensor shape: torch.Size([6270, 1024])
        # Element 1: Tensor shape: torch.Size([6270])
