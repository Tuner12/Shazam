from .model import MUMCBASE
import torch
import torch.nn.functional as F

class MUMC_Pretrain(MUMCBASE):
    def __init__(self, config=None):
        super().__init__(config=config, use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True)

    def forward(self, image, text, alpha=0.):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds, image_atts, image_feat = self.get_vision_embeds(image, mask_ratio=0.15)
        text = self.tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(image.device)
        text_embeds, text_feat = self.get_text_embeds(text)

        image_embeds_m, image_feat_m = self.get_momentum_vision_embeds(image, mask_ratio=0.15)
        text_feat_m = self.get_momentum_text_embeds(text)

        loss_ita = self.get_contrastive_loss(image_feat, text_feat, image_feat_m, text_feat_m, alpha)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text.attention_mask, text_feat)
        loss_mlm = self.get_mlm_loss(text, image_embeds, image_atts, image_embeds_m, alpha)

        return loss_mlm, loss_ita, loss_itm



