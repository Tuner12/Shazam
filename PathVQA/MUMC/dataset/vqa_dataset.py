import os
import json
from PIL import Image
from torch.utils.data import Dataset
#   File "dataset/vqa_dataset.py", line 5, in <module>
#     from .utils import pre_question, pre_answer
# ImportError: attempted relative import with no known parent package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from .utils import pre_question, pre_answer
import torchvision.transforms as transforms

class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.vqa_root, ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # prompt_info = ', its organ is {}, the type of answer is {}, the type of question is {}'\
        #               .format(ann['image_organ'], ann['answer_type'], ann['question_type'])
        # ann['question'] = ann['question'] + prompt_info


        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['qid']
            return image, question, question_id

        elif self.split in ['train', 'val']:
            question = pre_question(ann['question'], self.max_ques_words)
            # answer = pre_answer(ann['answer']) + self.eos
            answer = ann['answer']
            # answers = [pre_answer(answers)]
            # answers = [answer + self.eos for answer in answers]

            return image, question, answer



if __name__ == '__main__':
    ann_file = [ '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/train.json',
                '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/val.json' ]
    ann_file1 = [ '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/test.json' ]
    vqa_root = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/images'
    answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = vqa_dataset(ann_file, transform, vqa_root, split='train')
    print(len(dataset))
    d1 = dataset[0]
    print(d1[0].shape)
    print(d1[1])
    print(d1[2])

    dataset = vqa_dataset(ann_file1, transform, vqa_root, answer_list=answer_list, split='test')
    print(len(dataset))
    d2 = dataset[0]
    print(d2[0].shape)
    print(d2[1])
    print(d2[2])
    