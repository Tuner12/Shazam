import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from huggingface_hub import login
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
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
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
    elif model_name == 'gigapath':
       
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
   
    elif model_name =='virchow':
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

        from timm.layers import SwiGLUPacked
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
