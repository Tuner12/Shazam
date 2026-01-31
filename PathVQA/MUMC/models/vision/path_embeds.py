import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

class PatchEmbeds(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, drop_rate=0., norm_layer=None):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    model = PatchEmbeds(img_size=256)
    x = model(x)
    print(model)
    print(x.shape)