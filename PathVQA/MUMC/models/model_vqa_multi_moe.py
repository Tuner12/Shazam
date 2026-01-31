from functools import partial
import ruamel.yaml as yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .xbert import BertConfig, BertModel, BertLMHeadModel
from .tokenization_bert import BertTokenizer
import copy

# =============== Cross-Attention 模块 ===============
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, features):
        """
        features: [batch_size, n_tokens, d_model]
        """
        queries = self.query(features)
        keys    = self.key(features)
        values  = self.value(features)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(keys.size(-1))
        attn_weights = self.softmax(attention_scores)    # [B, n_tokens, n_tokens]

        attended_features = torch.matmul(attn_weights, values)
        out = self.output_layer(attended_features)
        out = self.layernorm(features + out)
        return out, attn_weights

class MultiCrossAttentionLayers(nn.Module):
    def __init__(self, d_model, num_layers=5):
        super(MultiCrossAttentionLayers, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model) for _ in range(num_layers)])

    def forward(self, features):
        """
        features: [batch_size, n_tokens, d_model]
        返回 fused_features: [batch_size, d_model]
        """
        all_attn_weights = []
        for layer in self.layers:
            features, attn_w = layer(features)
            all_attn_weights.append(attn_w)
        # mean pooling over token dimension
        fused_features = features.mean(dim=1)
        return fused_features, all_attn_weights

# =============== MoE 模块 ===============
class MoEOnePerLevel(nn.Module):
    """
    - 对同一层级 5 个教师的特征 (各自[batch, in_dim]) 做线性投影 => 5个 [B, d_model]
    - 再拼接 5 个投影为 [B, 5*d_model]，用一个 gating 网络得 [B,5] 的权重
    - 将权重乘回每个投影后做 LayerNorm
    - 最终输出 [B,5,d_model]
    """
    def __init__(self, in_dims, d_model=128):
        super().__init__()
        assert len(in_dims) == 5  # 支持5个模型

        self.d_model = d_model

        # 1) 分别投影 5 位教师 => d_model
        self.proj_list = nn.ModuleList([
            nn.Linear(in_dim, d_model) for in_dim in in_dims
        ])

        # 2) gating 网络：输入 [B, 5*d_model] => 输出 [B,5]
        self.gate = nn.Sequential(
            nn.Linear(5*d_model, 128),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 5)
        )

        # 3) LayerNorm
        self.ln = nn.LayerNorm(d_model)

    def forward(self, f1, f2, f3, f4, f5):
        """
        f1,f2,f3,f4,f5 : [B, in_dim_i]，5 位教师同层级特征
        返回: [B,5,d_model]
        """
        # 投影到 d_model
        p1 = self.proj_list[0](f1)
        p2 = self.proj_list[1](f2)
        p3 = self.proj_list[2](f3)
        p4 = self.proj_list[3](f4)
        p5 = self.proj_list[4](f5)
        cat_p = torch.cat([p1, p2, p3, p4, p5], dim=1)  # [B,5*d_model]
        gating_logits = self.gate(cat_p)           # => [B,5]
        gating_weights= torch.softmax(gating_logits, dim=1)  # => [B,5]

        # 每个教师的投影乘相应权重
        w1 = gating_weights[:, 0].unsqueeze(-1)  # [B,1]
        w2 = gating_weights[:, 1].unsqueeze(-1)
        w3 = gating_weights[:, 2].unsqueeze(-1)
        w4 = gating_weights[:, 3].unsqueeze(-1)
        w5 = gating_weights[:, 4].unsqueeze(-1)

        p1 = self.ln(p1 * w1)
        p2 = self.ln(p2 * w2)
        p3 = self.ln(p3 * w3)
        p4 = self.ln(p4 * w4)
        p5 = self.ln(p5 * w5)

        # 堆叠回 [B,5,d_model] 供后续 cross-attention
        out = torch.stack([p1, p2, p3, p4, p5], dim=1)
        return out

# =============== 特征映射模块 ===============
class FeatureMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.mapper = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        return self.mapper(features)

# =============== Multi-MoE VQA 模型 ===============
class MUMC_VQA_Multi_MoE(nn.Module):
    """
    Multi-MoE VQA模型，支持5个模型的特征融合
    - 每个层级（low/mid/high）使用一个MoE来处理5个模型的特征
    - 三段cross-attention，每段5层，总共15层
    - 支持特征蒸馏和知识蒸馏
    """
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 dim_list_low=None,   # 5个教师low特征维度
                 dim_list_mid=None,   # 5个教师mid特征维度
                 dim_list_high=None,  # 5个教师high特征维度
                 d_model=128,
                 num_layers=12):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.distill = config['distill']
        self.d_model = d_model
        self.num_layers = num_layers
        config_encoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder = BertConfig.from_json_file(config['bert_config'])
  
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)
        
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)
        # 蒸馏相关参数
       
        
        # =========== 1) 每个层级使用一个MoE来处理5个模型的特征 ===========
        self.moe_low = MoEOnePerLevel(dim_list_low, d_model=d_model)
        self.moe_mid = MoEOnePerLevel(dim_list_mid, d_model=d_model)
        self.moe_high = MoEOnePerLevel(dim_list_high, d_model=d_model)
        self.mapper_low  = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_low])
        self.mapper_mid  = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_mid])
        self.mapper_high = nn.ModuleList([FeatureMapper(d_in, d_model) for d_in in dim_list_high])
        # =========== 2) 三段cross-attention，每段4层 ===========
        self.segment1 = MultiCrossAttentionLayers(d_model, 4)
        self.segment2 = MultiCrossAttentionLayers(d_model, 4)
        self.segment3 = MultiCrossAttentionLayers(d_model, 4)
        
        # 残差连接的权重参数
        self.alpha_res = nn.Parameter(torch.tensor(1.0))
        self.beta_res = nn.Parameter(torch.tensor(1.0))
        
        # LayerNorm for residual connections
        self.res_ln2 = nn.LayerNorm(d_model)
        self.res_ln3 = nn.LayerNorm(d_model)
        
        # 特征投影层，将d_model映射到BERT期望的768维度
        self.feature_projection = nn.Linear(d_model, 768)
        
     
            
      
        
        # 蒸馏相关
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

    def forward(self, batch, question, answer=None, alpha=0, k=None, train=True):
        """
        batch: (15个特征 + 1 label), 但这里只取前15个特征
        对应：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
        """
        low1, mid1, high1 = batch[0],  batch[1],  batch[2]
        low2, mid2, high2 = batch[3],  batch[4],  batch[5]
        low3, mid3, high3 = batch[6],  batch[7],  batch[8]
        low4, mid4, high4 = batch[9],  batch[10], batch[11]
        low5, mid5, high5 = batch[12], batch[13], batch[14]

        # ---------------- Segment1 (low) ----------------
        # print("low_features：", low1.shape)
        # print("low_features：", low2.shape)
        # print("low_features：", low3.shape)
        # print("low_features：", low4.shape)
        # print("low_features：", low5.shape)
        seg1_input = self.moe_low(low1, low2, low3, low4, low5)   # => [B,5,d_model]
        out1, attn1 = self.segment1(seg1_input)                   # => [B,d_model]
        # print("out1：", out1.shape)

        # ---------------- Segment2 (mid) ----------------
        seg2_mid = self.moe_mid(mid1, mid2, mid3, mid4, mid5)     # => [B,5,d_model]
        out2, attn2 = self.segment2(seg2_mid)                     # => [B,d_model]
        out2 = self.res_ln2(out2 + self.alpha_res * out1)
        # print("out2：", out2.shape)

        # ---------------- Segment3 (high) ----------------
        seg3_high = self.moe_high(high1, high2, high3, high4, high5)  # => [B,5,d_model]
        out3, attn3 = self.segment3(seg3_high)                         # => [B,d_model]
        out3 = self.res_ln3(out3 + self.beta_res * out2)
        # print("out3：", out3.shape)

        # 使用最终特征进行VQA任务
        fused_features = out3.unsqueeze(1)  # [B,1,d_model] 作为图像特征
        
        # 投影到BERT期望的768维度
        fused_features = self.feature_projection(fused_features)  # [B,1,768]
        # print("fused_features：", fused_features.shape)
        image_atts = torch.ones(fused_features.size()[:-1], dtype=torch.long).to(fused_features.device)

        # 处理文本输入
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(fused_features.device)
        
        if train:
            answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(fused_features.device)
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=fused_features,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    question_output_m = self.text_encoder_m(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=fused_features,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                    logits_m = self.text_decoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=question_output_m.last_hidden_state,
                                                   encoder_attention_mask=question.attention_mask,
                                                   return_logits=True)

                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none')
            else:
                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  reduction='none')
            
            loss = answer_output.loss
            loss = loss.sum() / fused_features.size(0)
            return loss, out1, out2, out3

        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=fused_features,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            
            # 在推理模式下，answer是答案列表，需要先tokenize
            if isinstance(answer, list):
                answer_tokens = self.tokenizer(answer, padding='longest', return_tensors="pt").to(fused_features.device)
                answer_ids = answer_tokens.input_ids
                answer_atts = answer_tokens.attention_mask
            else:
                answer_ids = answer.input_ids
                answer_atts = answer.attention_mask
                
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                                    answer_ids, answer_atts, k)
            return topk_ids, topk_probs, out1, out2, out3

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)

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

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

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

        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))

def distill_pair(student_feat, teacher_feat):
    """
    一个简单的示例蒸馏损失: (1 - cos_sim) + SmoothL1
    student_feat: [B, d_model]
    teacher_feat: [B, d_model]
    """
    cos_term = 1.0 - nn.functional.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()
    smooth_l1= nn.HuberLoss()(student_feat, teacher_feat)
    return cos_term + smooth_l1

def multi_level_distillation_loss(
    out1, out2, out3,
    batch,  # 15个特征：[low1, mid1, high1, low2, mid2, high2, low3, mid3, high3, low4, mid4, high4, low5, mid5, high5]
    num_layers,
    model
):
    """
    多层蒸馏损失 - 支持5个模型
    """
    # 解包15个特征
    low1, mid1, high1 = batch[0],  batch[1],  batch[2]
    low2, mid2, high2 = batch[3],  batch[4],  batch[5]
    low3, mid3, high3 = batch[6],  batch[7],  batch[8]
    low4, mid4, high4 = batch[9],  batch[10], batch[11]
    low5, mid5, high5 = batch[12], batch[13], batch[14]

    total_loss = 0.0

    # 如果 num_layers >= 4
    if num_layers >= 4:
        for i, teacher_low in enumerate([low1, low2, low3, low4, low5]):
            teacher_low = model.mapper_low[i](teacher_low)          # => [B,d_model]
            total_loss += distill_pair(out1, teacher_low)

    # 如果 num_layers >= 8
    if num_layers >= 8:
        for i, teacher_mid in enumerate([mid1, mid2, mid3, mid4, mid5]):
            teacher_mid = model.mapper_mid[i](teacher_mid)
            total_loss += distill_pair(out2, teacher_mid)

    # 如果 num_layers == 12
    if num_layers == 12:
        for i, teacher_high in enumerate([high1, high2, high3, high4, high5]):
            teacher_high = model.mapper_high[i](teacher_high)
            total_loss += distill_pair(out3, teacher_high)

    return total_loss


if __name__ == '__main__':
    from ruamel.yaml import YAML
    yaml_loader = YAML(typ='rt')
    with open('/nas/leiwenhui/tys/PathVQA/MUMC/configs/VQA.yaml', 'r') as f:
        config = yaml_loader.load(f)
    
    print("读取BERT模型...")
    tokenizer = BertTokenizer.from_pretrained('/nas/leiwenhui/tys/PathVQA/MUMC/models/bert_model')
    print("读取BERT模型成功！")
    
    # 示例维度
    dim_list_low = [256, 256, 256, 256]
    dim_list_mid = [512, 512, 512, 512]
    dim_list_high = [768, 768, 768, 768]
    
    model = MUMC_VQA_Multi_MoE(
        config=config, 
        text_encoder='bert-base-uncased', 
        text_decoder='bert-base-uncased', 
        tokenizer=tokenizer,
        dim_list_low=dim_list_low,
        dim_list_mid=dim_list_mid,
        dim_list_high=dim_list_high
    )
    print("Multi-MoE VQA模型创建成功！")
    print(model)

    # 测试模型
    print("测试模型...")
    batch_size = 4
    batch = [
        torch.randn(batch_size, dim_list_low[0]),   # low1
        torch.randn(batch_size, dim_list_mid[0]),   # mid1
        torch.randn(batch_size, dim_list_high[0]),  # high1
        torch.randn(batch_size, dim_list_low[1]),   # low2
        torch.randn(batch_size, dim_list_mid[1]),   # mid2
        torch.randn(batch_size, dim_list_high[1]),  # high2
        torch.randn(batch_size, dim_list_low[2]),   # low3
        torch.randn(batch_size, dim_list_mid[2]),   # mid3
        torch.randn(batch_size, dim_list_high[2]),  # high3
        torch.randn(batch_size, dim_list_low[3]),   # low4
        torch.randn(batch_size, dim_list_mid[3]),   # mid4
        torch.randn(batch_size, dim_list_high[3]),  # high4
        torch.randn(batch_size, dim_list_low[0]),   # low5
        torch.randn(batch_size, dim_list_mid[0]),   # mid5
        torch.randn(batch_size, dim_list_high[0]),  # high5
    ]
    question = ["What is the color of the sky?"] * batch_size
    answer = ["blue"] * batch_size
    
    loss = model(batch, question, answer)
    print("损失值：", loss)
    print("测试模型成功！") 