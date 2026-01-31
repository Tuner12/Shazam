import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption