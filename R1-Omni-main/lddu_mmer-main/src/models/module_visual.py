# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'visual_config.json'
WEIGHTS_NAME = 'visual_pytorch_model.bin'


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None 
        self.relu = nn.ReLU()
        self.init_weights()
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) 
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) // 2 + dilation_size - 1, dropout=dropout)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer(x)
            else:
                out = layer(out)
        return out

class VisualConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `VisualModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file=4096,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02):
        """Constructs VisualConfig.

        Args:
            vocab_size_or_config_json_file: Size of the encoder layers and the pooler layer.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class VisualEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(VisualEmbeddings, self).__init__()

        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeddings):
        seq_length = input_embeddings.size(1)
        # 确保位置ID不超过max_position_embeddings
        max_pos = self.position_embeddings.num_embeddings
        position_ids = torch.arange(min(seq_length, max_pos), dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)
        
        # 如果序列长度超过max_position_embeddings，需要截断输入特征
        if seq_length > max_pos:
            input_embeddings = input_embeddings[:, :max_pos, :]
        
        words_embeddings = self.word_embeddings(input_embeddings)
        # words_embeddings = self.transform_act_fn(words_embeddings)
        
        position_embeddings = self.position_embeddings(position_ids)
      #  print("!!!INFO: VISUAL", words_embeddings.shape, position_embeddings.shape)
        embeddings = words_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VisualSelfAttention(nn.Module):
    def __init__(self, config):
        super(VisualSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # 使用更高效的方式计算注意力以减少内存占用
        with torch.cuda.amp.autocast(enabled=False):  # 确保计算稳定性
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            # 直接计算所需形状，避免多次permute和view操作
            batch_size = hidden_states.size(0)
            seq_length = hidden_states.size(1)
            
            # 重写transpose_for_scores逻辑以减少中间张量
            query_layer = mixed_query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            query_layer = query_layer.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_size]
            
            # 对于key，直接转置为适合matmul的形状
            key_layer = mixed_key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            key_layer = key_layer.permute(0, 2, 3, 1)  # [B, num_heads, head_size, seq_len]
            
            value_layer = mixed_value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
            value_layer = value_layer.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_size]

            # 计算注意力分数，使用scaled dot-product
            attention_scores = torch.matmul(query_layer, key_layer)  # [B, num_heads, seq_len, seq_len]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # 彻底修复注意力掩码处理，确保形状完全匹配
            
            # 计算目标形状 [B, num_heads, seq_len, seq_len]
            batch_size = attention_scores.size(0)
            num_heads = attention_scores.size(1)
            seq_len_k = attention_scores.size(3)
            
            # 重塑掩码以匹配注意力分数形状
            if attention_mask.dim() == 2:  # [B, seq_len_k]
                # 扩展到 [B, 1, 1, seq_len_k]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:  # [B, 1, seq_len_k]
                # 扩展到 [B, 1, 1, seq_len_k]
                attention_mask = attention_mask.unsqueeze(1)
            
            # 如果掩码长度与键序列长度不匹配，需要调整
            if attention_mask.size(-1) != seq_len_k:
                # 创建新的掩码，保持原始掩码的有效部分
                new_mask = torch.zeros(batch_size, 1, 1, seq_len_k, device=attention_mask.device, dtype=attention_mask.dtype)
                valid_len = min(attention_mask.size(-1), seq_len_k)
                new_mask[:, :, :, :valid_len] = attention_mask[:, :, :, :valid_len]
                attention_mask = new_mask
            
            # 广播掩码以匹配注意力分数的完整形状 [B, num_heads, seq_len, seq_len]
            attention_mask = attention_mask.expand(-1, num_heads, -1, -1)  # [B, num_heads, 1, seq_len_k]
            
            # 应用注意力掩码
            attention_scores = attention_scores + attention_mask

            # 归一化注意力分数
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            
            # 应用dropout
            attention_probs = self.dropout(attention_probs)

            # 计算上下文向量
            context_layer = torch.matmul(attention_probs, value_layer)  # [B, num_heads, seq_len, head_size]
            
            # 重塑回原始形状
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class VisualSelfOutput(nn.Module):
    def __init__(self, config):
        super(VisualSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualAttention(nn.Module):
    def __init__(self, config):
        super(VisualAttention, self).__init__()
        self.self = VisualSelfAttention(config)
        self.output = VisualSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class VisualIntermediate(nn.Module):
    def __init__(self, config):
        super(VisualIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VisualOutput(nn.Module):
    def __init__(self, config):
        super(VisualOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualLayer(nn.Module):
    def __init__(self, config):
        super(VisualLayer, self).__init__()
        self.attention = VisualAttention(config)
        self.intermediate = VisualIntermediate(config)
        self.output = VisualOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super(VisualEncoder, self).__init__()
        layer = VisualLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            # 每处理一个编码器层后清理中间变量，释放内存
            hidden_states = layer_module(hidden_states, attention_mask)
            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                # 对于除最后一层外的所有层，删除中间结果以节省内存
                if i < len(self.layer) - 1 and len(all_encoder_layers) > 1:
                    del all_encoder_layers[-2]  # 删除前一层的输出
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 释放未使用的GPU内存
            elif i == len(self.layer) - 1:
                all_encoder_layers.append(hidden_states)
                
        return all_encoder_layers


class VisualPooler(nn.Module):
    def __init__(self, config):
        super(VisualPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(VisualPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualLMPredictionHead, self).__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.weight = visual_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(visual_model_embedding_weights.size(1)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(self.weight) + self.bias
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualOnlyMLMHead, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class VisualOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(VisualOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class VisualPreTrainingHeads(nn.Module):
    def __init__(self, config, visual_model_embedding_weights):
        super(VisualPreTrainingHeads, self).__init__()
        self.predictions = VisualLMPredictionHead(config, visual_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config):
        super(VisualModel, self).__init__(config)
        self.embeddings = VisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))
        
        # 确保attention_mask与video具有相同的序列长度
        if attention_mask.size(1) != video.size(1):
            # 如果长度不匹配，截断或填充attention_mask
            if attention_mask.size(1) > video.size(1):
                attention_mask = attention_mask[:, :video.size(1)]
            else:
                # 如果需要填充，添加0
                pad_size = video.size(1) - attention_mask.size(1)
                attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask[:, :pad_size])], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(video)
        
        # 检查embedding_output的长度，并相应地调整attention_mask
        output_seq_length = embedding_output.size(1)
        if extended_attention_mask.size(3) != output_seq_length:
            extended_attention_mask = extended_attention_mask[:, :, :, :output_seq_length]
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
    
class TCNVisualEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TCNVisualEmbeddings, self).__init__()

        self.word_embeddings = TemporalConvNet(config.vocab_size, [config.vocab_size, config.hidden_size], 3, 0.3)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeddings):
        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)

        words_embeddings = self.word_embeddings(input_embeddings.transpose(1,2)).transpose(1,2)
        # words_embeddings = self.transform_act_fn(words_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
      #  print("!!!INFO: VISUAL", words_embeddings.shape, position_embeddings.shape)
        embeddings = words_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TCNVisualModel(PreTrainedModel):
    """Visual model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a VisualConfig class instance with the configuration to build a new model

    Inputs:
        `type`: a str, indicates which masking will be used in the attention, choice from [`bi`, `seq`, `gen`]
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see  paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for Visual-base, 24 for Visual-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see 's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])

    config = modeling.VisualConfig(vocab_size_or_config_json_file=4096, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.VisualModel(config=config)
    all_encoder_layers, pooled_output = model(video, video_mask)
    ```
    """
    def __init__(self, config):
        super(TCNVisualModel, self).__init__(config)
        self.embeddings = TCNVisualEmbeddings(config)
        self.encoder = VisualEncoder(config)
        self.pooler = VisualPooler(config)
        self.apply(self.init_weights)

    def forward(self, video, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(video.size(0), video.size(1))
        
        # 确保attention_mask与video具有相同的序列长度
        if attention_mask.size(1) != video.size(1):
            # 如果长度不匹配，截断或填充attention_mask
            if attention_mask.size(1) > video.size(1):
                attention_mask = attention_mask[:, :video.size(1)]
            else:
                # 如果需要填充，添加0
                pad_size = video.size(1) - attention_mask.size(1)
                attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask[:, :pad_size])], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(video)
        
        # 检查embedding_output的长度，并相应地调整attention_mask
        output_seq_length = embedding_output.size(1)
        if extended_attention_mask.size(3) != output_seq_length:
            extended_attention_mask = extended_attention_mask[:, :, :, :output_seq_length]
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output