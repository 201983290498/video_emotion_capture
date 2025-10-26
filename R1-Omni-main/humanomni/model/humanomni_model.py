# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .humanomni_arch import HumanOmniMetaModel, HumanOmniMetaForCausalLM
from torch.nn import CrossEntropyLoss

class HumanOmniQwen2Config(Qwen2Config):
    model_type = "HumanOmni_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "HumanOmni_qwen2"


class HumanOmniQwen2Model(HumanOmniMetaModel, Qwen2Model):
    config_class = HumanOmniQwen2Config

    def __init__(self, config: HumanOmniQwen2Config):
        super(HumanOmniQwen2Model, self).__init__(config)


class HumanOmniQwen2ForCausalLM(Qwen2ForCausalLM,HumanOmniMetaForCausalLM):
    config_class = HumanOmniQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = HumanOmniQwen2Model(config)
        self.vocab_size = config.vocab_size
        # 语言模型输出头：将隐藏层特征映射到词汇表维度（用于预测下一个 Token）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[int] = None,
        prompts: Optional[List[str]] = None,
        audios: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
       # 多模态输入预处理：整合图像、音频与文本
        if inputs_embeds is None:

            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                prompts=prompts,
                audios=audios
            )

         # 调用父类（Qwen2ForCausalLM）的 forward 方法，执行语言模型前向传播
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        outputs.labels = labels
        return outputs




    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 提取额外参数（位置编码、注意力掩码、提示文本等）
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        prompts = kwargs.pop("prompts", None)
        face_videos = kwargs.pop("face_videos", None)
        body_videos = kwargs.pop("body_videos", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            if face_videos is None:   # 无细分视频模态
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=images,
                    prompts=prompts,
                    audios=audios
                )
            else:   # 有细分视频模态（面部/身体）
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=images,
                    prompts=prompts,
                    face_videos=face_videos,
                    body_videos=body_videos,
                    audios=audios
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    # 生成过程输入准备
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("HumanOmni_qwen2", HumanOmniQwen2Config)
AutoModelForCausalLM.register(HumanOmniQwen2Config, HumanOmniQwen2ForCausalLM)
