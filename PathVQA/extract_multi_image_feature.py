import os
from functools import partial
import timm
# from .timm_wrapper import TimmCNNEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from huggingface_hub import login
import os
# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'

def hf_login_if_needed():
    """Login to Hugging Face Hub using token from environment variable if needed."""
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

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
    

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    elif model_name == 'gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
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
    elif model_name == 'hoptimus0':
        model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    elif model_name == 'chief':
        from ctran import ctranspath
        import torch.nn as nn
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'./checkpoints/Chief/CHIEF_CTransPath.pth',weights_only=False)
        model.load_state_dict(td['model'], strict=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    # print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms



def get_multi_encoder(model_name, target_img_size=224,extract_layers=['early', 'middle']):
    print(f'Loading model checkpoint: {model_name}')
    # target_img_size = 224
    def hook_fn(module, input, output):
        """ Hook function to store intermediate features """
        if isinstance(output, tuple):  
            output = output[0]
        extracted_features.append(output[:, 0, :])
    
    extracted_features = []

    if model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                                  init_values=1e-5, 
                                  num_classes=0, 
                                  dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

        # 选择合适的 Transformer 层
        #24
        layer_map = {
            'early': model.blocks[7],   # 浅层：第 3 层
            'middle': model.blocks[15], # 中层：第 13 层
        }

        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    elif model_name == 'uni_v2':
        # login('REMOVED_TOKEN')  # login with your User Access Token, found at https://huggingface.co/settings/tokens
        # login('REMOVED_TOKEN')
        # login("REMOVED_TOKEN")
        # login('REMOVED_TOKEN')
        # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
        timm_kwargs = {
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        layer_map = {
            'early': model.blocks[7],   # 浅层：第 3 层
            'middle': model.blocks[15], # 中层：第 13 层
        }

        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    # elif model_name == 'musk':
    #     # login('REMOVED_TOKEN')
    #     from musk import utils, modeling
    #     from timm.models import create_model
    #     base_model = create_model("musk_large_patch16_384")
    #     utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", base_model, 'model|module', '')
    #     model = Muskwrapper(base_model)
    #     layer_map = {
    #         'early': base_model.beit3.encoder.layers[7],   # 浅层：第 7 层
    #         'middle': base_model.beit3.encoder.layers[15], # 中层：第 15 层
    #     }

    #     # 注册 hooks
    #     for layer_name, layer in layer_map.items():
    #         if layer_name in extract_layers:
    #             layer.register_forward_hook(hook_fn)
    elif model_name == 'gigapath':
        # login("REMOVED_TOKEN")
        # login('REMOVED_TOKEN')
        # login("REMOVED_TOKEN")
        # login('REMOVED_TOKEN')
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        layer_map = {
            'early': model.blocks[13],   # 浅层：第 3 层
            'middle': model.blocks[26], # 中层：第 13 层
        }
        # 40
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    elif model_name == 'hoptimus0':
        model = timm.create_model( "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        layer_map = {
            'early': model.blocks[13],   # 浅层：第 3 层
            'middle': model.blocks[26], # 中层：第 13 层
        }
        # 40
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    elif model_name == 'hoptimus1':
        # login('REMOVED_TOKEN')
        # login('REMOVED_TOKEN')
        model = timm.create_model( "hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
        layer_map = {
            'early': model.blocks[13],   # 浅层：第 3 层
            'middle': model.blocks[26], # 中层：第 13 层
        }
        # 40
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    # elif model_name =='virchow2':
    #     from timm.layers import SwiGLUPacked
    #     base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    #     model = Virchow2Wrapper(base_model)
    #     layer_map = {
    #         'early': base_model.blocks[5],   # 浅层：第 5 层
    #         'middle': base_model.blocks[15], # 中层：第 15 层
    #     }

    #     # 注册 hooks
    #     for layer_name, layer in layer_map.items():
    #         if layer_name in extract_layers:
    #             layer.register_forward_hook(hook_fn)
    #     model.eval()
    elif model_name =='virchow':
        hf_login_if_needed()
        from timm.layers import SwiGLUPacked
        base_model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = Virchow2Wrapper(base_model)
        layer_map = {
            'early': base_model.blocks[10],   
            'middle': base_model.blocks[20], 
        }
        # 32
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
        model.eval()
    elif model_name =='virchow2':
        # login('REMOVED_TOKEN')
        from timm.layers import SwiGLUPacked
        # login('REMOVED_TOKEN')
        # login('REMOVED_TOKEN')
        base_model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model = Virchow2Wrapper(base_model)
        layer_map = {
            'early': base_model.blocks[10],   
            'middle': base_model.blocks[20], 
        }
        # 32
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
        model.eval()
    elif model_name == 'phikon_v2':
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained("owkin/phikon-v2")
        base_model.eval()
        model = PhikonWrapper(base_model)
        layer_map = {
            'early': base_model.encoder.layer[7],   
            'middle': base_model.encoder.layer[15], 
        }
        # 24
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    elif model_name == 'gigapath':
        # login("REMOVED_TOKEN")
        # login('REMOVED_TOKEN')
        # login("REMOVED_TOKEN")
        hf_login_if_needed()
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        layer_map = {
            'early': model.blocks[13],   # 浅层：第 3 层
            'middle': model.blocks[26], # 中层：第 13 层
        }
        # 40
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    elif model_name == 'chief':
        from ctran import ctranspath
        import torch.nn as nn
        base_model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        base_model.head = nn.Identity()
        td = torch.load(r'/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/CLAM/checkpoints/Chief/CHIEF_CTransPath_v09.pth',weights_only=False)
        base_model.load_state_dict(td['model'], strict=False)
        model = chiefWrapper(base_model)

        layer_map = {
            'early': base_model.layers[1].blocks[1],
            'middle': base_model.layers[2].blocks[3],
        }

        # 24
        # 注册 hooks
        for layer_name, layer in layer_map.items():
            if layer_name in extract_layers:
                layer.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    def forward_with_extraction(x):
        """ Custom forward function to extract multi-layer features """
        nonlocal extracted_features
        extracted_features = []  # 清空特征存储
        _ = model(x)  # 前向传播
        
        # 仅返回选定的层
        feature_dict = {}
        layer_names = ['early', 'middle']
        for i, layer_name in enumerate(layer_names):
            if layer_name in extract_layers:
                feature_dict[layer_name] = extracted_features[i]

        return feature_dict
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    return model, forward_with_extraction, img_transforms


def extract_features(model_name, data_loader, save_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, forward_fn, img_transforms = get_multi_encoder(model_name, target_img_size=224, extract_layers=['early', 'middle'])
    model = model.to(device)
    model.eval()
    
    low_features = []
    mid_features = []
    high_features = []
    question_ids = []
    questions = []
    answers = []
    
    print(f"Extracting features using {model_name} on device: {device}")
    if data_loader.dataset.split == 'train':
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting Features", unit="batch"):
                images = batch[0].to(device)
                # batch_image_ids = batch['image_idx']
                question = batch[1]
                answer = batch[2]
                features_low_mid = forward_fn(images)
                features_high = model(images)  # Extract features
                
                low_features.append(features_low_mid['early'].cpu())
                mid_features.append(features_low_mid['middle'].cpu())
                high_features.append(features_high.cpu())
                questions.extend(question)  # 使用extend而不是append
                # 保持原始dataset的答案格式：包含单个答案的列表
                answers.extend(answer)      # 直接使用原始格式

        # Concatenate all extracted features
        low_features = torch.cat(low_features, dim=0)
        mid_features = torch.cat(mid_features, dim=0)
        high_features = torch.cat(high_features, dim=0)
        # 保存为元组格式：(low_features, mid_features, high_features, questions, answers)
        torch.save((low_features, mid_features, high_features, questions, answers), save_path)
        
        print(f"Features saved to {save_path}")
        print(f"Shape of low features tensor: {low_features.shape}")
        print(f"Shape of mid features tensor: {mid_features.shape}")
        print(f"Shape of high features tensor: {high_features.shape}")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of answers: {len(answers)}")
    elif data_loader.dataset.split == 'val':
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting Features", unit="batch"):
                images = batch[0].to(device)
                # batch_image_ids = batch['image_idx']
                question = batch[1]
                answer = batch[2]
                features_low_mid = forward_fn(images)
                features_high = model(images)  # Extract features
                
                low_features.append(features_low_mid['early'].cpu())
                mid_features.append(features_low_mid['middle'].cpu())
                high_features.append(features_high.cpu())
                questions.extend(question)  # 使用extend而不是append
                # 保持原始dataset的答案格式：包含单个答案的列表
                answers.extend(answer)      # 直接使用原始格式

        # Concatenate all extracted features
        low_features = torch.cat(low_features, dim=0)
        mid_features = torch.cat(mid_features, dim=0)
        high_features = torch.cat(high_features, dim=0)
        # 保存为元组格式：(low_features, mid_features, high_features, questions, answers)
        torch.save((low_features, mid_features, high_features, questions, answers), save_path)
        
        print(f"Features saved to {save_path}")
        print(f"Shape of low features tensor: {low_features.shape}")
        print(f"Shape of mid features tensor: {mid_features.shape}")
        print(f"Shape of high features tensor: {high_features.shape}")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of answers: {len(answers)}")
    elif data_loader.dataset.split == 'test':
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting Features", unit="batch"):
                images = batch[0].to(device)
                # batch_image_ids = batch['image_idx']
                question = batch[1]
                question_id = batch[2]
                features_low_mid = forward_fn(images)
                features_high = model(images)  # Extract features
                
                low_features.append(features_low_mid['early'].cpu())
                mid_features.append(features_low_mid['middle'].cpu())
                high_features.append(features_high.cpu())
                questions.extend(question)      # 使用extend而不是append
                question_ids.extend(question_id) # 使用extend而不是append

        # Concatenate all extracted features
        low_features = torch.cat(low_features, dim=0)
        mid_features = torch.cat(mid_features, dim=0)
        high_features = torch.cat(high_features, dim=0)
        # 保存为元组格式：(low_features, mid_features, high_features, questions, question_ids)
        torch.save((low_features, mid_features, high_features, questions, question_ids), save_path)
        
        print(f"Features saved to {save_path}")
        print(f"Shape of low features tensor: {low_features.shape}")
        print(f"Shape of mid features tensor: {mid_features.shape}")
        print(f"Shape of high features tensor: {high_features.shape}")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of question_ids: {len(question_ids)}")


if __name__ == "__main__":
    # os.environ['http_proxy'] = 'http://192.168.1.18:7890'
    # os.environ['https_proxy'] = 'http://192.168.1.18:7890'
    import sys
    sys.path.append('/nas/leiwenhui/tys/PathVQA/MUMC/')
    from dataset.vqa_dataset import vqa_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import numpy as np
    # from pathvqa_dataset import create_datasets
    # 设置参数
    # data_root = "/nas/leiwenhui/tys/PathVQA/VQAMed2019"
    model_name = 'gigapath'  # 可以修改为其他模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    print("=== VQAMed 图片特征提取 ===")
    print(f"使用模型: {model_name}")
    print(f"设备: {device}")
    print(f"批次大小: {batch_size}")
    
    try:
        # 创建专门用于特征提取的数据集（只处理唯一图片）
        print("正在创建图片数据集...")
        # train_dataset = ImageOnlyDataset(
        #     root_dir=os.path.join(data_root, 'ImageClef-2019-VQA-Med-Training'),
        #     image_dir='Train_images'
        # )
        # val_dataset = ImageOnlyDataset(
        #     root_dir=os.path.join(data_root, 'ImageClef-2019-VQA-Med-Validation'),
        #     image_dir='Val_images'
        # )
        # test_dataset = ImageOnlyDataset(
        #     root_dir=os.path.join(data_root, 'VQAMed2019Test'),
        #     image_dir='VQAMed2019_Test_Images'
        # )
        from ruamel.yaml import YAML
        yaml_loader = YAML(typ='rt')
        with open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
            config = yaml_loader.load(f)

        _, _, img_transforms = get_multi_encoder(model_name, target_img_size=224, extract_layers=['early', 'middle'])
        # train_dataset, test_dataset = create_dataset('pathvqa', config)
        train_dataset = vqa_dataset(config['pathvqa']['train_file'], img_transforms, config['pathvqa']['vqa_root'], split='train')
        val_dataset = vqa_dataset(config['pathvqa']['val_file'], img_transforms, config['pathvqa']['vqa_root'], split='val')
        test_dataset = vqa_dataset(config['pathvqa']['test_file'], img_transforms, config['pathvqa']['vqa_root'], split='test',
                                   answer_list=config['pathvqa']['answer_list'])
        print(f"图片数量:")
        print(f"训练集: {len(train_dataset)} 张图片")
        print(f"验证集: {len(val_dataset)} 张图片")
        print(f"测试集: {len(test_dataset)} 张图片")
        print(f"总计: {len(train_dataset) + len(val_dataset) + len(test_dataset)} 张图片")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建features目录
        os.makedirs("features4pathvqa/images", exist_ok=True)
        
        # 提取训练集特征
        # print("正在提取训练集特征...")
        # train_save_path = f"features4pathvqa/images/train_{model_name}_features.pt"
        # extract_features(model_name, train_loader, train_save_path, device)
        
        # 提取验证集特征
        print("正在提取验证集特征...")
        val_save_path = f"features4pathvqa/images/val_{model_name}_features.pt"
        extract_features(model_name, val_loader, val_save_path, device)
        
        # 提取测试集特征
        # print("正在提取测试集特征...")
        # test_save_path = f"features4pathvqa/images/test_{model_name}_features.pt"
        # extract_features(model_name, test_loader, test_save_path, device)
        
        print("\n=== 特征提取完成 ===")
        print(f"所有特征已保存到 features4pathvqa/images 目录")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

