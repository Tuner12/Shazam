from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel



# Load phikon-v2
# processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model = AutoModel.from_pretrained("owkin/phikon-v2")
model.eval()
image = torch.randn(1,3,224,224)
# Process the image
# inputs = processor(image, return_tensors="pt")

# Get the features
with torch.inference_mode():
    outputs = model(image)
    features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
print(features.shape)
# assert features.shape == (1, 1024)
