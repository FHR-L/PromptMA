import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from einops import einops

from model.clip.model import LayerNorm, Transformer, QuickGELU
from model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

"""

"""

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class Part_Attention(nn.Module):
    def __init__(self, ratio=0.5, token_select_type='', alpha=0.5):
        super(Part_Attention, self).__init__()
        self.token_select_type = token_select_type
        self.ratio = ratio
        self.alpha = alpha

    def forward(self, x, modal):
        if modal == 0:
            position_mix = [-1, -2]
            position_special = -3
        elif modal == 1:
            position_mix = [-1, -3]
            position_special = -2
        else:
            position_mix = [-2, -3]
            position_special = -1
        length = len(x)
        N = x[0].shape[2] - 4
        B = x[0].shape[0]
        k = 0
        attn_norn = torch.eye(132, dtype=x[0].dtype, device=x[0].device).unsqueeze(0).expand(B, -1, -1)
        last_map = x[k] * self.alpha + attn_norn * (1. - self.alpha)
        # print(last_map.shape)  #[32, 132, 132]
        for i in range(k + 1, length):
            last_map = torch.matmul(x[i] * self.alpha + attn_norn * (1. - self.alpha), last_map)

        last_map_key = last_map[:, 0, 1:129]
        _, topk_indices = torch.topk(last_map_key, int(N * self.ratio), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((B, N), dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        max_index_set_key = selected_tokens_mask

        last_map_prompt = last_map[:, position_mix[0], 1:129] + last_map[:, position_mix[1], 1:129]
        _, topk_indices = torch.topk(last_map_prompt, int(N * self.ratio), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((B, N), dtype=torch.bool).cuda()
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        max_index_set_prompt = selected_tokens_mask

        if self.token_select_type == 'all':
            max_index_set = max_index_set_prompt | max_index_set_key
        elif self.token_select_type == 'inner':
            max_index_set = max_index_set_key
        elif self.token_select_type == 'cross':
            max_index_set = max_index_set_prompt
        else:
            assert 1 < 0
        return _, max_index_set

class ResidualAttentionBlockModal(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, layer_index=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.layer_index = layer_index

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, feat):
        q = feat[0].unsqueeze(0)
        kv = feat[1:]
        q = q + self.attention(self.ln_1(q), self.ln_1(kv), self.ln_1(kv))
        q = q + self.mlp(self.ln_2(q))
        return torch.cat([q, kv], dim=0)

class Token_fusion(nn.Module):
    def __init__(self, visual, layers, ratio, token_select_type, alpha):
        super().__init__()
        print("Token fusion ration:", ratio)
        width = 768
        self.layers = layers
        heads = 12
        self.ln_pre = torch.nn.InstanceNorm1d(width)
        self.ratio = ratio

        self.resblocks = copy.deepcopy(visual.transformer.resblocks[-layers:])

        self.class_embedding = copy.deepcopy(visual.class_embedding)
        self.part_attention_shared = Part_Attention(ratio=self.ratio, token_select_type=token_select_type, alpha=alpha)
        self.ln_post = LayerNorm(width)

    def forward(self,
                rgb_feature_list, rgb_atten_list,
                ni_feature_list, ni_atten_list,
                ti_feature_list, ti_atten_list,
                prompt):
        cls = prompt
        l = prompt.shape[0]
        _, R = self.part_attention_shared(rgb_atten_list, 0)
        _, T = self.part_attention_shared(ti_atten_list, 1)
        _, N = self.part_attention_shared(ni_atten_list, 2)
        for blk, rgb, ni, ti in zip(self.resblocks, rgb_feature_list, ni_feature_list, ti_feature_list):
            index = blk.layer_index - 1
            rgb_feature = rgb_feature_list[index][1:] * R.unsqueeze(-1).permute(1, 0, 2)
            ti_feature = ti_feature_list[index][1:] * T.unsqueeze(-1).permute(1, 0, 2)
            ni_feature = ni_feature_list[index][1:] * N.unsqueeze(-1).permute(1, 0, 2)
            feat = torch.cat([cls, rgb_feature, ti_feature, ni_feature], dim=0)
            cls = blk(feat, cls_len=l)[:l]
        cls = cls.permute(1, 0, 2)  # LND -> NLD
        cls = self.ln_post(cls)
        return cls


class LOCAL_FEATURE(nn.Module):
    def __init__(self, visual, layers, ratio):
        super().__init__()
        print("Token fusion ration:", ratio)
        width = 768
        self.layers = layers
        heads = 12
        self.ln_pre = torch.nn.InstanceNorm1d(width)
        self.ratio = ratio

        self.resblocks = copy.deepcopy(visual.transformer.resblocks[-layers:])

        self.class_embedding = copy.deepcopy(visual.class_embedding)
        self.part_attention_shared = Part_Attention(ratio=self.ratio)
        self.ln_post = LayerNorm(width)

    def forward(self,
                feature_list, atten_list,
                prompt):
        cls = prompt
        l = prompt.shape[0]
        for blk in self.resblocks:
            index = blk.layer_index - 1
            _, mask = self.part_attention_shared(atten_list[0: index + 1])
            feature = feature_list[index][1:] * mask.unsqueeze(-1).permute(1, 0, 2)
            feat = torch.cat([cls, feature], dim=0)
            cls = blk(feat)[:l]
        cls = cls.permute(1, 0, 2)  # LND -> NLD
        cls = self.ln_post(cls)
        return cls

class LocalFeature(nn.Module):
    def __init__(self, visual, layers, ratio):
        super().__init__()
        print("Token fusion ration:", ratio)
        width = 768
        self.layers = layers
        heads = 12
        self.ln_pre = torch.nn.InstanceNorm1d(width)
        self.ratio = ratio

        self.resblocks = copy.deepcopy(visual.transformer.resblocks[-layers:])

        self.class_embedding = copy.deepcopy(visual.class_embedding)
        self.part_attention_shared = Part_Attention(ratio=self.ratio)
        self.ln_post = LayerNorm(width)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.AL = cfg.MODEL.AL
        self.prompt_layer = cfg.MODEL.PROMPT_LAYER_NUM
        self.fusion_layer = 12 - self.prompt_layer
        self.special_learner = cfg.MODEL.SPECIAL_LEARNER
        self.fusion = cfg.MODEL.FUSION_LAYER

        self.rgb_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.rgb_classifier.apply(weights_init_classifier)
        self.ni_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.ni_classifier.apply(weights_init_classifier)
        self.ti_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.ti_classifier.apply(weights_init_classifier)
        self.fusion_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.fusion_classifier.apply(weights_init_classifier)

        self.rgb_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.rgb_bottleneck.bias.requires_grad_(False)
        self.rgb_bottleneck.apply(weights_init_kaiming)
        self.ni_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.ni_bottleneck.bias.requires_grad_(False)
        self.ni_bottleneck.apply(weights_init_kaiming)
        self.ti_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.ti_bottleneck.bias.requires_grad_(False)
        self.ti_bottleneck.apply(weights_init_kaiming)
        self.fusion_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.fusion_bottleneck.bias.requires_grad_(False)
        self.fusion_bottleneck.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.shared_class_token = cfg.MODEL.SHARED_CLASS_TOKEN

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size,
                                      self.shared_class_token)
        self.image_encoder = clip_model.visual

        clip_model = load_clip_to_cpu_qkv(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size,
                                      self.shared_class_token)
        self.SB = Token_fusion(clip_model.visual, self.fusion_layer, cfg.MODEL.RATIO, cfg.MODEL.TOKEN_SELECT_TYPE, cfg.MODEL.TOKEN_SELECT_ALPHA)
        # scale = self.in_planes ** -0.5
        # self.sb_learn_cls_token = nn.Parameter(scale * torch.randn(self.in_planes))

        self.prompt_embedding = nn.Parameter(torch.randn(3, 3, self.in_planes))
        self.prompt_mask = torch.ones(3, 3) - torch.eye(3, 3)
        self.prompt_change_machine = cfg.MODEL.PROMPT_CHANGE_MACHINE
        self.sb_cls_type = cfg.MODEL.FUSION_CLASS_TOKEN
        self.sb_learn_cls_token = copy.deepcopy(clip_model.visual.class_embedding)

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed_rgb = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed_rgb, std=.02)
            self.cv_embed_ni = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed_ni, std=.02)
            self.cv_embed_ti = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed_ti, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed_rgb = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed_rgb, std=.02)
            self.cv_embed_ni = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed_ni, std=.02)
            self.cv_embed_ti = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed_ti, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed_rgb = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed_rgb, std=.02)
            self.cv_embed_ni = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed_ni, std=.02)
            self.cv_embed_ti = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed_ti, std=.02)
            print('camera number is : {}'.format(view_num))

    def forward(self, x=None, cam_label=None, view_label=None):
        rgb = x['RGB']  # torch.Size([128, 3, 256, 128])
        ni = x['NI']  # torch.Size([128, 3, 256, 128])
        ti = x['TI']  # torch.Size([128, 3, 256, 128])

        if cam_label != None and view_label!=None:
            cv_embed_rgb = self.sie_coe * self.cv_embed_rgb[cam_label * self.view_num + view_label]
            cv_embed_ni = self.sie_coe * self.cv_embed_ni[cam_label * self.view_num + view_label]
            cv_embed_ti = self.sie_coe * self.cv_embed_ti[cam_label * self.view_num + view_label]
        elif cam_label != None:
            cv_embed_rgb = self.sie_coe * self.cv_embed_rgb[cam_label]
            cv_embed_ni = self.sie_coe * self.cv_embed_ni[cam_label]
            cv_embed_ti = self.sie_coe * self.cv_embed_ti[cam_label]
        elif view_label!=None:
            cv_embed_rgb = self.sie_coe * self.cv_embed_rgb[view_label]
            cv_embed_ni = self.sie_coe * self.cv_embed_ni[view_label]
            cv_embed_ti = self.sie_coe * self.cv_embed_ti[view_label]
        else:
            cv_embed_rgb = None
            cv_embed_ni = None
            cv_embed_ti = None

        rgb_atten = []
        ni_atten = []
        ti_atten = []
        rgb_layer_features = []
        ni_layer_features = []
        ti_layer_features = []

        rgb_features = self.image_encoder(rgb, cv_embed_rgb, get_embedding=True)
        ni_features = self.image_encoder(ni, cv_embed_ni, get_embedding=True)
        ti_features = self.image_encoder(ti, cv_embed_ti, get_embedding=True)

        l, batch, d = rgb_features.shape
        prompt = self.prompt_embedding.unsqueeze(2).expand(-1, -1, batch, -1)
        for blk, layer in zip(self.image_encoder.transformer.resblocks,
                                           range(12)):
            if layer < self.prompt_layer:
                if self.prompt_change_machine == 'transpose':
                    # print("transpose")
                    rgb_features, rgb_attn, prompt_rgb = blk(rgb_features, get_att=True, prompt=prompt[0])
                    ni_features, ni_attn, prompt_ni = blk(ni_features, get_att=True, prompt=prompt[1])
                    ti_features, ti_attn, prompt_ti = blk(ti_features, get_att=True, prompt=prompt[2])
                    prompt = torch.cat([
                        prompt_rgb.unsqueeze(0),
                        prompt_ni.unsqueeze(0),
                        prompt_ti.unsqueeze(0)], dim=0)
                    prompt = prompt.transpose(0, 1)
                elif self.prompt_change_machine == 'permute':
                    rgb_features, rgb_attn, prompt_rgb = blk(rgb_features, get_att=True, prompt=prompt[(0 + layer) % 3])
                    ni_features, ni_attn, prompt_ni = blk(ni_features, get_att=True, prompt=prompt[(1 + layer) % 3])
                    ti_features, ti_attn, prompt_ti = blk(ti_features, get_att=True, prompt=prompt[(2 + layer) % 3])
                    prompt = torch.cat([
                        prompt_rgb.unsqueeze(0),
                        prompt_ni.unsqueeze(0),
                        prompt_ti.unsqueeze(0)], dim=0)
                elif self.prompt_change_machine == 'rote':
                    rgb_features, rgb_attn, prompt_rgb = blk(rgb_features, get_att=True, prompt=prompt[0])
                    ni_features, ni_attn, prompt_ni = blk(ni_features, get_att=True, prompt=prompt[1])
                    ti_features, ti_attn, prompt_ti = blk(ti_features, get_att=True, prompt=prompt[2])
                    prompt = torch.cat([
                        prompt_rgb.unsqueeze(0),
                        prompt_ni.unsqueeze(0),
                        prompt_ti.unsqueeze(0)], dim=0)
                    prompt = torch.rot90(prompt, 1, [0, 1])
                elif self.prompt_change_machine == 'frozen':
                    rgb_features, rgb_attn, prompt_rgb = blk(rgb_features, get_att=True, prompt=prompt[0])
                    ni_features, ni_attn, prompt_ni = blk(ni_features, get_att=True, prompt=prompt[1])
                    ti_features, ti_attn, prompt_ti = blk(ti_features, get_att=True, prompt=prompt[2])
                    prompt = torch.cat([
                        prompt_rgb.unsqueeze(0),
                        prompt_ni.unsqueeze(0),
                        prompt_ti.unsqueeze(0)], dim=0)
                else:
                    assert 1 < 0
                rgb_atten.append(rgb_attn)
                ni_atten.append(ni_attn)
                ti_atten.append(ti_attn)

            else:
                rgb_features = blk(rgb_features)
                ni_features = blk(ni_features)
                ti_features = blk(ti_features)
            rgb_layer_features.append(rgb_features)
            ni_layer_features.append(ni_features)
            ti_layer_features.append(ti_features)


        if self.sb_cls_type == 'un-diagonal':
            shared_prompt = einops.rearrange(prompt, 'mn1 mn2 b d -> (mn1 mn2) b d')[:-1]  # flatten
            shared_prompt = einops.rearrange(shared_prompt, '(mn1 mn2) b d -> mn1 mn2 b d', mn1=2)[:, 1:]
            shared_prompt = einops.rearrange(shared_prompt, 'mn1 mn2 b d -> (mn1 mn2) b d')  # flatten
            sb_cls = torch.mean(shared_prompt, dim=0, keepdim=True)
        elif self.sb_cls_type == 'diagonal':
            sb_cls = (prompt[0][0] + prompt[1][1] + prompt[2][2])/3.
            sb_cls = sb_cls.unsqueeze(0)
        elif self.sb_cls_type == 'all':
            prompt = einops.rearrange(prompt, 'mn1 mn2 b d -> (mn1 mn2) b d') # flatten
            sb_cls = torch.mean(prompt, dim=0, keepdim=True)
        elif self.sb_cls_type == 'learnable':
            sb_cls = self.sb_learn_cls_token.to(rgb_features.dtype) + torch.zeros(1, batch, d, dtype=rgb_features.dtype, device=rgb_features.device)
        else:
            print(self.sb_cls_type)
            assert 1 < 0
        # shared_prompt = einops.rearrange(prompt, 'mn1 mn2 b d -> (mn1 mn2) b d')[:-1]  # flatten
        # shared_prompt = einops.rearrange(shared_prompt, '(mn1 mn2) b d -> mn1 mn2 b d', mn1=2)[:, 1:]
        # shared_prompt = einops.rearrange(shared_prompt, 'mn1 mn2 b d -> (mn1 mn2) b d')  # flatten
        # sb_cls = torch.mean(shared_prompt, dim=0, keepdim=True)

        rgb_features = rgb_features.permute(1, 0, 2)
        ni_features = ni_features.permute(1, 0, 2)
        ti_features = ti_features.permute(1, 0, 2)

        rgb_features = self.image_encoder.ln_post(rgb_features)
        ni_features = self.image_encoder.ln_post(ni_features)
        ti_features = self.image_encoder.ln_post(ti_features)

        rgb_feature_ = rgb_features[:, 0]
        ni_feature_ = ni_features[:, 0]
        ti_feature_ = ti_features[:, 0]

        if self.fusion:
            fusion_feature = self.SB(rgb_layer_features, rgb_atten,
                                     ti_layer_features, ti_atten,
                                     ni_layer_features, ni_atten,
                                     sb_cls)[:, 0]

        rgb_feat = self.rgb_bottleneck(rgb_feature_)
        ni_feat = self.ni_bottleneck(ni_feature_)
        ti_feat = self.ti_bottleneck(ti_feature_)
        if self.fusion:
            fusion_feat = self.fusion_bottleneck(fusion_feature)

        if self.training:
            rgb_cls_score = self.rgb_classifier(rgb_feat)
            ni_cls_score = self.ni_classifier(ni_feat)
            ti_cls_score = self.ti_classifier(ti_feat)
            if self.fusion:
                fusion_cls_score = self.fusion_classifier(fusion_feat)
                return ([rgb_cls_score, ni_cls_score, ti_cls_score, fusion_cls_score],
                        [rgb_feature_, ni_feature_, ti_feature_, fusion_feature],
                        [prompt, prompt.transpose(0, 1)])
            else:
                return ([rgb_cls_score, ni_cls_score, ti_cls_score],
                        [rgb_feature_, ni_feature_, ti_feature_],
                        [prompt, prompt.transpose(0, 1)])
        else:

            if self.AL:
                assert 1 < 0
            if self.neck_feat == 'after':
                if self.fusion:
                    return [rgb_feat, ni_feat, ti_feat, fusion_feat]  # rgb_feat, ni_feat, ti_feat
                else:
                    return [rgb_feat, ni_feat, ti_feat]
            else:
                assert 1 < 0


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from model.clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, shared_class_token):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size,
                             shared_class_token=shared_class_token,
                             vision_coder='VisionTransformerTokenFusion')

    return model

def load_clip_to_cpu_qkv(backbone_name, h_resolution, w_resolution, vision_stride_size, shared_class_token):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size,
                             shared_class_token=shared_class_token,
                             vision_coder='qkv')

    return model
