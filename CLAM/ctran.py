from timm.models.layers.helpers import to_2tuple
import timm
import torch.nn as nn
import torch, torchvision
from torchvision import transforms
class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        
        # self.strict_img_size = strict_img_size
        # self.output_fmt = output_fmt
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model

if __name__ == "__main__":
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'./checkpoints/Chief/CHIEF_CTransPath.pth',weights_only=False)
    model.load_state_dict(td['model'], strict=False)
    model.eval()
    def print_model_parameters(model, model_name):
        print(f"\n{'='*20} {model_name} Parameters {'='*20}")
        for name, param in model.named_parameters():
            print(f"{name:70} {tuple(param.shape)}")


    print_model_parameters(model, "chief")
    image = torch.randn(1,3,224,224)
    with torch.no_grad():
        patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
        print(patch_feature_emb.size())