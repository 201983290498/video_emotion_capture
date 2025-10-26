# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import time
import os
from abc import ABC, abstractmethod
import math
import re
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from .projector import load_mm_projector, build_vision_projector, build_audio_projector
from .encoder import build_vision_tower, build_audio_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP, IMAGE_TOKEN_PATCH, MODAL_INDEX_REMAP
from humanomni.mm_utils import frame_sample
from transformers import BertModel, BertTokenizer
import h5py
import torch.distributed as dist
import ipdb

# 特征压缩器
class SFDynamicCompressor(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.out_channels = vision_tower.hidden_size  # 输出特征维度（与视觉塔隐藏层一致）
        self.mid_channel = 256   # 中间层维度（特征压缩过渡）

        self.vlm_query_projector = nn.Linear(self.out_channels, self.mid_channel)
        self.vlm_key_projector = nn.Linear(self.out_channels, self.mid_channel)

    def downsample(self, x):
        return F.avg_pool2d(x, 2, 2)    # 2倍下采样（步长=2，核大小=2）
    
    def downsample_4(self, x):
        return F.avg_pool2d(x, 4, 4)
        
    def forward(self, image_features, image_size=None):
        if image_size is None:
            W = int(math.sqrt(image_features.shape[1]))
            H = int(W)
        else:
            H, W = image_size
        image_features = einops.rearrange(image_features, 't (r w) h -> t r w h', r = H)
        T, H, W, C = image_features.shape
        image_features = image_features.unsqueeze(0)
        B = 1

        # Fast特征：2倍下采样（快速通道，低计算量）
        fast_feature = F.avg_pool2d(image_features.permute(0, 1, 4, 2, 3).view(B*T, C, H, W), 2, 2) # B * T, C, H // 2, W //2 
        fast_feature = fast_feature.view(B*T, C, -1)
        fast_feature = fast_feature.permute(0, 2, 1).view(B, T, -1, C).view(B, -1, C)
        # Slow特征：间隔采样（稀疏通道，高精度）
        index = torch.arange(1, T, 4)
        if len(index) == 0:
            index = torch.tensor([0])
        slow_feature = image_features[:, index, :, :, :].view(B, -1, C)
        # 多尺度特征融合：拼接Fast和Slow特征
        final_feature = torch.cat([fast_feature, slow_feature], dim=1)  
        return final_feature

# 多模态元模型
class HumanOmniMetaModel:

    def __init__(self, config):
        super(HumanOmniMetaModel, self).__init__(config)
        # 初始化视觉塔与视觉投影器
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)


        # BERT门控网络：用于动态生成多尺度视觉特征的融合权重（训练时可注释避免重复初始化）
        num_branches = 3
        bert_model = "/data/testmllm/models/bert-base-uncased"   # 本地BERT模型路径
        self.bert_model =  BertModel.from_pretrained(bert_model)   # 加载BERT模型（用于文本指令编码）
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)   # BERT分词器
        # BERT门控：将BERT输出（768维）映射为3个分支的权重
        modules = [nn.Linear(self.bert_model.config.hidden_size, 3584)]
        modules.append(nn.GELU())    # 激活函数
        modules.append(nn.Linear(3584, num_branches))
        self.bert_gate = nn.Sequential(*modules)
        self.bert_softmax = nn.Softmax(dim=1)   # 归一化权重为概率
       # self.feature_compressor = SFDynamicCompressor(config, self.vision_tower)
        #####
        # 初始化音频塔与音频投影器
        if hasattr(config, "mm_audio_tower"):
            self.audio_tower = build_audio_tower(config, delay_load=True)
            self.config.audio_hidden_size = getattr(self.audio_tower, "hidden_size", 1280)    # 音频特征维度
            self.audio_projector = build_audio_projector(config, vision_cfg=self.audio_tower.config)   # 音频投影器

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:   # 分布式训练（FSDP）时视觉塔以列表存储
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, "audio_tower", None)
        return audio_tower

    # 视觉模块初始化
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter    # 预训练视觉投影器权重路径

        self.config.mm_vision_tower = vision_tower

        # 加载视觉塔权重
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)   # 新建视觉塔

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:   # 已有视觉塔，直接加载权重
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        # 更新视觉相关配置
        self.config.use_mm_proj = True   # 启用多模态投影
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')   # 投影器类型（如线性）
        self.config.mm_hidden_size = vision_tower.hidden_size   # 视觉特征维度
        self.config.mm_vision_select_layer = mm_vision_select_layer   # 特征选择层
        self.config.mm_vision_select_feature = mm_vision_select_feature  # 特征类型

        # 初始化/加载视觉投影器
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # 初始化音频模块
        if model_args.audio_tower:
            self.initialize_audio_modules(model_args, fsdp)
        # 加载预训练投影器权重
        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                # 加载本地权重（文件夹或单文件）
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            # 提取投影器权重（过滤前缀"mm_projector."）
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
          
            # 加载权重到投影器
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

   
     #  self.feature_compressor = SFDynamicCompressor(model_args, vision_tower)
        # 重新初始化BERT门控网络
        num_branches = 3
        bert_model = "bert-base-uncased"
        self.bert_model =  BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        # self.bert_gate = nn.Linear(self.bert_model.config.hidden_size, num_branches)
        modules = [nn.Linear(self.bert_model.config.hidden_size, 3584)]
        modules.append(nn.GELU())
        modules.append(nn.Linear(3584, num_branches))
        self.bert_gate = nn.Sequential(*modules)
        self.bert_softmax = nn.Softmax(dim=1)


    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_audio_mlp_adapter = model_args.pretrain_audio_mlp_adapter
        self.config.mm_audio_tower = audio_tower
        self.config.mm_audio_projector_type = getattr(model_args, "mm_audio_projector_type", "mlp2x_gelu")
        if self.get_audio_tower() is None:
            audio_tower = build_audio_tower(model_args)
            
            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
            else:
                audio_tower = self.audio_tower
            audio_tower.load_model()
        self.config.audio_hidden_size = getattr(audio_tower, "hidden_size", 1280)
        if getattr(self, "audio_projector", None) is None:
            self.audio_projector = build_audio_projector(self.config, vision_cfg=audio_tower.config)
        else:
            # In case it is frozen by LoRA
            for p in self.audio_projector.parameters():
                p.requires_grad = True
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        if pretrain_audio_mlp_adapter is not None:
            audio_projector_weights = torch.load(pretrain_audio_mlp_adapter, map_location="cpu")
        #     # import pdb; pdb.set_trace()
            incompatible_keys = self.audio_projector.load_state_dict(get_w(audio_projector_weights, "audio_projector"))
            print(f"load audio projector: {incompatible_keys}")
        num_trainable_parameters = sum(p.numel() for p in self.audio_projector.parameters() if p.requires_grad) / 1e6
        print(f"Number of trainable parameters in audio projector: {num_trainable_parameters}M")


class HumanOmniMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        # 获取视频采样帧数（优先从配置读取，默认NUM_FRAMES=32）
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()   # 代理获取视觉塔

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()   # 代理获取音频塔

    # 对视觉特征进行空间下采样，减少 token 数量
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

        height, weight = image_feature.shape[2:]
        # 计算下采样后的形状
        scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
        # 双线性插值下采样
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images_or_videos(self, images, device=None, inputs_bert_dict=None):
        """修正后的批量图像/视频编码函数"""
        
        # 统一输入格式处理
        data_batch = []
        modal_types = []
        valid_indices = []
        
        for i, item in enumerate(images):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                # 标准格式: (data, modal)
                data = item[0]
                modal = item[1] if len(item) > 1 else 'video'
            elif isinstance(item, torch.Tensor):
                # 直接传入张量
                data = item
                modal = 'video'
            else:
                print(f"警告: 跳过无效的输入格式 {type(item)}")
                continue
                
            if data is not None and isinstance(data, torch.Tensor):
                data_batch.append(data)
                modal_types.append(modal)
                valid_indices.append(i)
        
        if not data_batch:
            return []
        
        batch_size = len(data_batch)
        current_device = torch.cuda.current_device()
        
        try:
            # 批量处理: 拼接所有帧
            frames = torch.cat(data_batch, dim=0)
            
            # 视觉塔批量提取特征
            frames_features = self.get_model().get_vision_tower()(frames)
            
            # 重新排列为批次格式
            video_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b=batch_size)
            body_features = video_features       
            face_features = frames_features
            
            # 投影器批量处理
            video_features, body_features, face_features = self.get_model().mm_projector(
                video_features, body_features, face_features
            )
            
            # 面部特征形状调整
            face_features = einops.rearrange(face_features, '(b t) n h -> b t n h', b=batch_size)
            
            # BERT门控处理
            # 统一门控权重的设备与数据类型到视觉特征
            feature_device = video_features.device
            feature_dtype = video_features.dtype
            if inputs_bert_dict is not None:
                outputs_bert = self.get_model().bert_model(**inputs_bert_dict)
                last_hidden_state_bert = outputs_bert.last_hidden_state
                cls_token_embedding_bert = last_hidden_state_bert[:, 0, :]
                logits = self.get_model().bert_gate(cls_token_embedding_bert)
                branch_probs = self.get_model().bert_softmax(logits).to(device=feature_device, dtype=feature_dtype)
            else:
                # 默认权重，保持与视觉特征一致的设备/类型
                branch_probs = torch.ones(batch_size, 3, device=feature_device, dtype=feature_dtype) / 3
            
            # 多尺度特征融合
            final_features = []
            for i, (video_feat, body_feat, face_feat) in enumerate(zip(video_features, body_features, face_features)):
                # 面部特征下采样
                if modal_types[i] == 'video':
                    face_feat = self.get_2dPool(face_feat)
                # 统一面部特征的token维度，与video/body一致：展平为 (t*n, h)
                if face_feat.ndim == 3:
                    face_feat = einops.rearrange(face_feat, 't n h -> (t n) h')
                
                # 加权融合（各分支形状均为 [L, H]）
                fused_feature = (
                    video_feat * branch_probs[i][0] + 
                    body_feat * branch_probs[i][1] + 
                    face_feat * branch_probs[i][2]
                )
                
                final_features.append(fused_feature)

            return final_features
            
        except Exception as e:
            print(f"批量编码图像/视频时出错: {str(e)}")
            # 回退到逐个处理
            individual_features = []
            for i, (data, modal) in enumerate(zip(data_batch, modal_types)):
                try:
                    single_input = [(data, modal)]
                    single_features = self.encode_images_or_videos(single_input, device, inputs_bert_dict)
                    if single_features:
                        individual_features.append(single_features[0])
                    else:
                        individual_features.append(None)
                except Exception as single_e:
                    print(f"单个视频编码失败: {str(single_e)}")
                    individual_features.append(None)
            
            return individual_features

    # 塔提取→池化降维→投影适配
    def encode_audios(self, audios):
        audio_features = self.get_model().get_audio_tower()(audios).permute(0, 2, 1).contiguous() #b, t, c -> b, c, t   # torch.Size([1, 1280, 1500])
        audio_features = torch.nn.functional.avg_pool1d(audio_features, kernel_size=3, stride=3).permute(0, 2, 1).contiguous() # torch.Size([1, 1280, 500])
        audio_features = self.get_model().audio_projector(audio_features)
        # print("222")
        return audio_features

    # 多模态输入与标签准备
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None,audios=None
    ):

        if audios is not None:
            if len(audios.shape) == 4 and audios.shape[1] == 1:
                audios = audios.squeeze(1)  # 移除第一维
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        # NOTE: text-only situation
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, past_key_values, None, labels
        device_ = input_ids.device
        mm_features = self.encode_images_or_videos(images ,device_,prompts)

        if audios is not None and audio_tower is not None:
            audio_features = self.encode_audios(audios)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # pure text input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1 
                continue

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            while mm_token_indices.numel() > 0:
                mm_token_start = mm_token_indices[0]
                cur_modal = MODAL_INDEX_REMAP[cur_input_ids[mm_token_start].item()]
                if cur_modal in ["<image>", "<video>"]:
                    cur_mm_idx += 1
                    cur_mm_features = mm_features[batch_idx]
                    if len(cur_mm_features.size())==3:
                        cur_mm_features=cur_mm_features.flatten(0,1)
                elif cur_modal in  ["<audio>"] and audio_tower is not None:
                    cur_mm_features = audio_features[batch_idx]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start])) 
                cur_new_input_embeds.append(cur_mm_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                cur_input_ids = cur_input_ids[mm_token_start+1:] 
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels
