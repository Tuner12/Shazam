import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ctran import ctranspath

# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# trnsfrms_val = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = mean, std = std)
#     ]
# )


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./checkpoints/Chief/CHIEF_CTransPath.pth')
model.load_state_dict(td['model'], strict=True)
# model.eval()
image = torch.randn(1,3,224,224)
with torch.no_grad():
    patch_feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,768]
    print(patch_feature_emb.size())
    