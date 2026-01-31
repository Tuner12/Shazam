from functools import partial
import ruamel.yaml as yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# (vqa) leiwenhui@sjtu54:/nas/leiwenhui/tys/PathVQA/MUMC$ python train_vqa.py 
# Traceback (most recent call last):
#   File "train_vqa.py", line 13, in <module>
#     from models.model_vqa import MUMC_VQA
#   File "/nas/leiwenhui/tys/PathVQA/MUMC/models/model_vqa.py", line 6, in <module>
#     from vision.vit import VisionTransformer
# ModuleNotFoundError: No module named 'vision'
from vision.vit import VisionTransformer
from xbert import BertConfig, BertModel, BertLMHeadModel
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from transformers import BertTokenizer
from tokenization_bert import BertTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from dataset.vqa_feature_dataset import vqa_feature_dataset
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# os.environ['http_proxy'] = 'http://192.168.1.18:7890'
# os.environ['https_proxy'] = 'http://192.168.1.18:7890'
# # os.environ.pop('http_proxy', None)
# # os.environ.pop('https_proxy', None)
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

class MUMC_VQA(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 image_embeds_dim=1536,
                 ):
        super().__init__()

        self.tokenizer = tokenizer

        self.distill = config['distill']

        # self.visual_encoder = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.image_embeds_proj = nn.Linear(image_embeds_dim, 768)
        config_encoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        # text_encoder = '/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model'
        # text_decoder = '/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model'
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)
        
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder,
                                                            add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, image, question, answer=None, alpha=0, k=None, train=True):
      
        # image_embeds = self.visual_encoder(image)
        # # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_embeds = self.image_embeds_proj(image)
        image_embeds = image_embeds.unsqueeze(1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # print("image_embeds：", image_embeds.shape)
        # print("image_atts：", image_atts.shape)
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(image.device)
        answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device).to(image.device)

        # train
        if train:
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    question_output_m = self.text_encoder_m(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                    logits_m = self.text_decoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=question_output_m.last_hidden_state,
                                                   encoder_attention_mask=question.attention_mask,
                                                   return_logits=True,
                                                   )

                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  )
            else:
                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  reduction='none',
                                                  )
            loss = answer_output.loss
            loss = loss.sum() / image.size(0)   # image.size(0) = batch_size
            return loss

        # test
        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k)

            return topk_ids, topk_probs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)  # num_ques = batch_size_test
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id,
                                            -100)

        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)


        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        # topk() 会自动按概率从高到低排序，返回的 topk_probs 和 rerank_id 都是排序后的
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        # 根据重新排序的索引 rerank_id，重新排列 topk_ids，使其与 topk_probs 的顺序一致
        # 此时 topk_ids 和 topk_probs 都是按照最终概率从高到低排序的
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

if __name__ == '__main__':
    # config_path = '/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml'
    # config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    from ruamel.yaml import YAML
    yaml_loader = YAML(typ='rt')
    with open('/data2/tanyusheng/Code/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
        config = yaml_loader.load(f)
    # 重新下载BERT模型
    print("读取下载BERT模型...")
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # from tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    # image = torch.randn(4, 3, 384, 384)
    # image_dim = image.shape[1]
    text_encoder = '/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model'
    text_decoder = '/data2/tanyusheng/Code/PathVQA/MUMC/models/bert_model'
    model = MUMC_VQA(config=config, text_encoder=text_encoder, text_decoder=text_decoder, tokenizer=tokenizer)
    print("BERT模型下载成功！")
    # print(model)

    # 测试模型
    print("测试模型...")

    # image = torch.randn(4, 768)
    # question = ["What is the color of the sky?"] * 4
    # answer = ["blue"] * 4
    # loss = model(image, question, answer)
    # print("损失值：", loss)
    # print("测试模型成功！")
    
    # train_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/train_virchow2_features.pt'
    # test_feature_file = '/nas/leiwenhui/tys/PathVQA/features4pathvqa/images/test_virchow2_features.pt'
    # answer_list = '/nas/leiwenhui/tys/PathVQA/Pathvqa/pvqa/pvqa_json/answer_list.json'
    
    # # 测试训练集
    # print("=== 训练集 ===")
    # dataset = vqa_feature_dataset(train_feature_file, split='train')
    # d1 = dataset[0]
    # print(f"  Image feature shape: {d1[0].shape}")
    # print(f"  Question: {d1[1]}")
    # print(f"  Answer: {d1[2]}")
    # print(f"  Feature dimension: {dataset.get_feature_dim()}")
    # print(f"  Dataset length: {len(dataset)}")
    # image = d1[0]
    # question = d1[1]
    # answer = d1[2]
    # loss = model(image, question, answer)
    # print("损失值：", loss)
    # print("测试模型成功！")
    # # 测试测试集
    # print("\n=== 测试测试集 ===")
    # dataset = vqa_feature_dataset(test_feature_file, answer_list=answer_list, split='test')
    # d2 = dataset[0]
    # image = d2[0]
    # question = d2[1]
    # answer = d2[2]
    # topk_ids, topk_probs = model(image, question, answer, k=3, train=False)
    # print("topk_ids：", topk_ids)
    # print("topk_probs：", topk_probs)
    # print("测试模型成功！")
    # image = torch.randn(4, 3, 384, 384)
    # question = ["What is the color of the sky?"] * 4    
    # answer = ["blue"] * 4
    # loss = model(image, question, answer)
    # print("损失值：", loss)
    # print("测试模型成功！")

    # # answerlist
    # 注意：image应该是预提取的图像特征，形状为 (batch_size, 1536)
    # 而不是原始图像张量 (batch_size, 3, 384, 384)
    image = torch.randn(4, 1536)  # 4个样本，每个样本1536维特征
    question = ["What is the color of the sky?" , "Waht is the color of the cat?", "What is the color of the dog?", "What is the color of the bird?"] 
    answerlist = ["blue", "red", "green", "yellow", "purple", "orange", "brown", "gray", "black", "white"]
    print(f"答案列表长度: {len(answerlist)}")
    print(f"答案列表: {answerlist}")
    
    # 验证：在传入模型前，先看看tokenize后的顺序
    answer_tokenized = model.tokenizer(answerlist, padding='longest', return_tensors="pt")
    print(f"\n验证：tokenize后的答案顺序（通过第一个token确认）")
    for i, ans in enumerate(answerlist):
        first_token = answer_tokenized.input_ids[i, 1].item()  # 第1个位置是第一个token（第0个是CLS）
        token_text = model.tokenizer.decode([first_token])
        print(f"  索引 {i}: {ans} -> 第一个token: {token_text} (token_id: {first_token})")
    
    topk_ids, topk_probs = model(image, question, answerlist, k=10, train=False)
    print(f"\n实际返回的topk数量: {topk_ids.shape[1]}")
    print("topk_ids：", topk_ids)
    print("topk_probs：", topk_probs)
    
    # 打印每个问题对应的top-k答案
    print("\n=== Top-K 答案结果 ===")
    for i, (q, ids, probs) in enumerate(zip(question, topk_ids, topk_probs)):
        print(f"\n问题 {i+1}: {q}")
        ids_list = ids.cpu().tolist()
        probs_list = probs.cpu().tolist()
        for rank, (idx, prob) in enumerate(zip(ids_list, probs_list), 1):
            if 0 <= idx < len(answerlist):
                answer_text = answerlist[idx]
                print(f"  Top-{rank}: {answer_text} (索引: {idx}, 概率: {prob:.4f})")
            else:
                print(f"  Top-{rank}: 索引超出范围! (索引: {idx}, 概率: {prob:.4f})")
    
    print("\n测试模型成功！")