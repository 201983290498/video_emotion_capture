from importlib.metadata import entry_points
from .models import *
import math
import torch
import numpy as np
import torch.nn as nn
from .module_cross import CrossModel
from .module_decoder import DecoderModel, Decoder
from .module_bert import BertModel
from .module_audio import AudioModel
from .module_visual import VisualModel
from .until_module import CTCModule
from ..utils.eval import get_metrics
from ..utils.eval import search_binary as eval_search_binary
import copy



def getBinaryTensor(imgTensor, boundary = 0.21):
    # 添加数值检查
    if torch.isnan(imgTensor).any():
        print("警告: getBinaryTensor 输入包含 NaN，使用零值替换")
        imgTensor = torch.nan_to_num(imgTensor, nan=0.0)
    
    if torch.isinf(imgTensor).any():
        print("警告: getBinaryTensor 输入包含 Inf，使用有限值替换")
        imgTensor = torch.nan_to_num(imgTensor, posinf=1.0, neginf=0.0)
    one = torch.ones_like(imgTensor).fill_(1)
    zero = torch.zeros_like(imgTensor).fill_(0)
    return torch.where(imgTensor > boundary, one, zero)

# 代理到 utils.eval.search_binary，避免重复逻辑与混乱的日志输出
# 保持统一的 (y_true, y_pred_scores) 调用习惯
# 注意：eval_search_binary 会进行全局阈值搜索并返回 float 阈值
# 如需调试日志，请在调用处打印

def search_binary(pred_score, total_label, eval_threshold=None):
    # 如果显式传入 eval_threshold，则直接用该阈值计算一次指标（保持与旧实现兼容）
    if eval_threshold is not None:
        total_pred = getBinaryTensor(pred_score, eval_threshold)
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_label, total_pred)
        # 仅用于兼容早期调用，不在此处打印日志
        # 调用方可选择打印这些指标以做对比
    # 委托给 utils.eval.search_binary
    return eval_search_binary(pred_score, total_label)

def search_binary(pred_score, total_label, eval_threshold=None):
    # 统一代理到 utils.eval.search_binary，避免重复实现造成行为不一致
    return eval_search_binary(pred_score, total_label, eval_threshold)

class LabelWiseModalModel(nn.Module):
    
    def __init__(self, label_dim, hidden_dim, pro_dim, dropout=0.1):
        super(LabelWiseModalModel, self).__init__()
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.pro_dim = pro_dim
        self.seq_projection = nn.ModuleList([nn.Linear(hidden_dim, pro_dim) for _ in range(3)]) # 3 modalities v,a,t
        self.label_projection = nn.ModuleList([nn.Linear(label_dim, pro_dim) for _ in range(3)]) # 3 modalities v,a,t

        self.out_projection = nn.ModuleList([nn.Linear(pro_dim, hidden_dim) for _ in range(3)]) # 3 modalities
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, label_features, visual_features, audio_features, text_features):
        # label_features: [batch_size, labels, label_dim]
        # modal_features: [batch_size, seq, hidden_dim]
        label_proj_features = [] # v, a, t
        for i in range(len(self.label_projection)):
            label_proj_features.append(self.label_projection[i](label_features)) # [batch_size, labels, pro_dim]
        
        modal_proj_features = [] # v, a, t
        modal_proj_features.append(self.seq_projection[0](visual_features))
        modal_proj_features.append(self.seq_projection[1](audio_features))
        modal_proj_features.append(self.seq_projection[2](text_features)) # [batch_size, seq, pro_dim]


        Vij = []
        for i in range(3):
            attention_scores = torch.matmul(label_proj_features[i], modal_proj_features[i].transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.pro_dim) # b,l,s
            attention_scores = self.softmax(attention_scores) # b,l,s
            feat = torch.matmul(attention_scores, modal_proj_features[i]) # b,l,d
            label_proj_features[i] = self.dropout(feat)
            # label_proj_features[i] = feat
            Vij.append(self.out_projection[i](label_proj_features[i]))

        return Vij # [[batch_size, labels, hidden_dim], [batch_size, labels, hidden_dim], [batch_size, labels, hidden_dim]]

class LatentEncoder(nn.Module):
    def __init__(self, state_list, output_size, latten_size, dropout=0.5):
        super(LatentEncoder, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        self.fc_mu = nn.Linear(output_size, latten_size)
        self.fc_var = nn.Linear(output_size, latten_size)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_var.weight)
        self.dropout = nn.Dropout(dropout)
        
        
    def encode(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        output = F.dropout(F.relu(self.output(linear_out)), p=0.1) # [B, labels, latent_size]
        output = F.relu(self.output(linear_out))
        mu = self.fc_mu(output) 
        logvar = self.fc_var(output) # [B, labels, latent_size]
        return [mu, logvar]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # return mu + eps*std
        return mu 

class LatentDecoder(nn.Module):
    def __init__(self, state_list, output_size, dropout=0.5):
        super(LatentDecoder, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s) for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)
        self.dropout = nn.Dropout(dropout)
    
    def decode(self, latent_feature):
        linear_out = latent_feature
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        linear_out = F.dropout(linear_out, p=0.1)
        return self.output(linear_out)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, neg_mask=None, batch_size=-1, device=None, other_features=None):
        if mask is not None:
            mask = mask.float().detach()
            if other_features is None:
                similarity = torch.matmul(features[:batch_size], features.T) / 2.0  # 计算相似度
                anchor_dot_contrast = torch.div(
                    similarity,
                    self.temperature)
            else:
                similarity = torch.matmul(features[:batch_size], other_features.T) / 2.0  # 计算相似度
                anchor_dot_contrast = torch.div(
                    similarity,
                    self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            if neg_mask is None:
                logits_mask = torch.ones_like(mask)
            else:
                logits_mask = torch.scatter(neg_mask, 1, torch.arange(batch_size).view(-1, 1).to(neg_mask.device), 0)
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)
        return loss, ((similarity * mask).sum() / mask.sum(),(similarity * logits_mask).sum() / logits_mask.sum())

class MyModel(TAILORPreTrainedModel):
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config, task_config):
        super(MyModel, self).__init__(bert_config, visual_config, audio_config, cross_config, decoder_config)
        
        self.task_config = task_config 
        self.ignore_video_index = -1 
        self.num_classes = task_config.num_classes 
        self.aligned = task_config.aligned 
        
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_sequence <= audio_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "bert_num_hidden_layers")
        bert_config = update_attr("bert_config", bert_config, "vocab_size",
                                   self.task_config, "text_dim")

        self.bert = BertModel(bert_config) 
        self.embedding_layer = nn.Embedding(self.num_classes, task_config.hidden_size)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        visual_config = update_attr("visual_config", visual_config, "vocab_size",
                                    self.task_config, "video_dim")

        self.visual = VisualModel(visual_config)
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                    self.task_config, "audio_num_hidden_layers")
        audio_config = update_attr("audio_config", audio_config, "vocab_size",
                                    self.task_config, "audio_dim")

        self.audio = AudioModel(audio_config)
        
        self.text_norm = NormalizeText(task_config) 
        self.visual_norm = NormalizeVideo(task_config)
        self.audio_norm = NormalizeAudio(task_config)
        
        self.label_wise_attention = LabelWiseModalModel(task_config.label_dim, task_config.hidden_size, task_config.pro_dim)
        
        self.common_classfier = nn.Sequential(
            nn.Linear(task_config.hidden_size * 3, task_config.hidden_size // 2) ,
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 1)
        )

        self.final_classifier = nn.ModuleList([nn.Sequential( 
                nn.Linear(task_config.latent_size * 3, task_config.hidden_size // 2),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(task_config.hidden_size // 2, 1)
            ) for _ in range(6)])
        
        self.final_classifer2 = nn.Sequential(
            nn.Linear(task_config.latent_size * 6, task_config.hidden_size), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size, 6),
            nn.Sigmoid()
        )

        self.var_classifier1 = nn.ModuleList([nn.Sequential( 
            nn.Linear(task_config.latent_size * 3, task_config.hidden_size // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 1)
        ) for _ in range(6)])

        self.var_classifier2 = nn.Sequential(
            nn.Linear(task_config.latent_size * 6, task_config.hidden_size ), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size, 6),
            nn.Sigmoid()
        )

        # --- 新增：注册类先验与pos_weight缓冲，用于稳定训练与提升召回 ---
        self.register_buffer("class_priors", torch.full((self.num_classes,), 0.5))
        self.register_buffer("pos_weight", torch.ones(self.num_classes))
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss(task_config.temperature, task_config.temperature)
        self.encoder = nn.ModuleList([LatentEncoder([task_config.hidden_size, task_config.hidden_size // 4 * 3, task_config.hidden_size // 2], task_config.hidden_size //2, task_config.latent_size) for _ in range(3)])
        
        # 添加缺失的特征提取器属性初始化
        self.private_feature_extractor = nn.ModuleList([nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.1),
            nn.Tanh()
        ) for _ in range(3)])
        
        # 修正CTC模块初始化 - 移除hidden_size参数
        # 视频到文本的CTC对齐模块
        self.v2t_ctc = CTCModule(
            in_dim=task_config.video_dim,  # 视频特征维度
            out_seq_len=task_config.max_words  # 目标文本序列长度
        )
        
        # 音频到文本的CTC对齐模块
        self.a2t_ctc = CTCModule(
            in_dim=task_config.audio_dim,  # 音频特征维度
            out_seq_len=task_config.max_words  # 目标文本序列长度，与文本保持一致
        )
        
        self.common_feature_extractor= nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )
        
        self.cross_classifier = EmotionClassifier(cross_config.hidden_size, 1) 
        
        self.decoder = DecoderModel(decoder_config)

        self.va_cross = CrossModel(cross_config)
        self.vat_cross = CrossModel(cross_config)
        self.pc_cross = CrossModel(cross_config)

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.latent_size*2))
        # self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.latent_size))     
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)
        self.numCounts = {}
        self.correctCounts = {}
    @torch.cuda.amp.autocast()  # 添加自动混合精度训练
    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=None, training=False, samples_index=None): 
        """
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        groundTruth_labels: [B, num_classes]

        """

        label_input = label_input.unsqueeze(0).repeat(text.shape[0], 1) # L=> b,L
        label_mask = label_mask.unsqueeze(0).repeat(text.shape[0], 1) # l => b,l

        # print(f"嵌入层调用前 - label_input 数据类型: {label_input.dtype}")

        label_features = self.embedding_layer(label_input) # B, L, D

        # 添加输入检查
        def check_and_fix(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"前向传播输入 {name} 包含异常值，进行修复")
                return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            return tensor

        text = self.text_norm(text)
        visual = self.visual_norm(visual)  
        audio = self.audio_norm(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask) #[B, L, D]
        
        # [batch_size, label_nums, hidden_size]
        label_features = self.label_wise_attention(label_features, visual_output, audio_output, text_output)

        mus, stds = [], []
        for idx in range(len(label_features)):
            label_feature = label_features[idx]
            mu, log_var = self.encoder[idx].encode(label_feature)
            mus.append(mu)
            stds.append(torch.exp(0.5 * log_var))

        common_features = torch.cat((label_features[0], label_features[1],label_features[2]), axis=-1)# [batch_size, label_nums, 3 * hidden_size] 
        common_result = self.common_classfier(common_features).squeeze(-1)  
        # 修复双重sigmoid：若分类器已包含Sigmoid，则直接作为概率；否则进行一次sigmoid
        common_prob = common_result
        if torch.min(common_prob) < 0 or torch.max(common_prob) > 1:
            common_prob = torch.sigmoid(common_prob)
        
        mus_features = torch.cat(mus, axis=-1)
        stds_features = torch.cat(stds, axis=-1)
        mus_result, stds_result = [], []
        for i in range(len(self.final_classifier)):
            mus_result.append(self.final_classifier[i](mus_features[:,i,:].unsqueeze(1)).squeeze(-1))
        mus_result = torch.cat(mus_result, dim=-1)
        for i in range(len(self.var_classifier1)):
            stds_result.append(self.var_classifier1[i](stds_features[:,i,:].unsqueeze(1)).squeeze(-1))
        stds_result = torch.cat(stds_result, dim=-1)


        # 在关键计算处添加保护
        try:
            # 使用概率域融合，避免直接加权logits带来的失真
            mus_prob = torch.sigmoid(mus_result)
            stds_prob = torch.sigmoid(stds_result)
            final_prob = mus_prob * common_prob + stds_prob * (1 - common_prob)
        except Exception as e:
            print(f"最终概率计算出错: {e}")
            final_prob = torch.sigmoid((mus_result + stds_result) / 2)
        final_prob = torch.nan_to_num(final_prob, nan=0.0, posinf=1.0, neginf=0.0)

        # 统一到概率域用于评估与阈值
        final_labels = getBinaryTensor(final_prob)
        l2_norms = torch.norm(stds_features, p=2, dim=2).data.clone().cpu()
        if training:
            # Compute losses in FP32 to avoid overflow under AMP
            with torch.cuda.amp.autocast(enabled=False):
                # 引入基于类先验的权重，提升召回与稳定性
                pos_w = getattr(self, 'pos_weight', torch.ones(self.num_classes, device=final_prob.device)).float()
                weights = groundTruth_labels.float() * pos_w.unsqueeze(0) + (1.0 - groundTruth_labels.float()) * 1.0
                common_loss = torch.nn.functional.binary_cross_entropy(common_prob.float(), groundTruth_labels.float(), weight=weights, reduction='mean')
                final_loss = torch.nn.functional.binary_cross_entropy(final_prob.float(), groundTruth_labels.float(), weight=weights, reduction='mean')
                cdl_loss, mean_similarity = self.SupContrasts(mus, stds, groundTruth_labels)
                crl_loss = self.CRLDis(common_prob , groundTruth_labels, stds_features, samples_index)
                total_loss = self.task_config.cml*common_loss + self.task_config.final_loss * final_loss + self.task_config.cdl*cdl_loss + self.task_config.crl*crl_loss
            return total_loss, final_labels, groundTruth_labels, final_prob, mean_similarity, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)
        else:
            return final_labels, groundTruth_labels, final_prob, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)

    def get_text_visual_audio_output(self, text, text_mask, visual, visual_mask, audio, audio_mask):
        """
        Uni-modal Extractor
        """
        text_layers, _ = self.bert(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1] 
        visual_layers, _ = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]  
        audio_layers, _ = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1] 

        return text_output, visual_output, audio_output

    def SupContrasts(self, mus, stds, groundTruth_labels):
        mus = torch.stack(mus, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        mus = torch.nn.functional.normalize(mus, dim=-1)
        stds = torch.stack(stds, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        stds = torch.nn.functional.normalize(stds, dim=-1)
        total_proj = torch.cat((mus, stds), dim=-1) # [batch_size, 3, 6 labels, 2 * latent_dim]
        total_proj = total_proj.view(-1, total_proj.shape[-1]).float()  # force FP32 for stability
        
        cl_labels = self.get_cl_labels(groundTruth_labels, times=1).view(-1).unsqueeze(-1)
        # Ensure queue tensors are FP32 to match
        cl_feats = torch.cat((total_proj, self.queue.clone().detach().float()), dim=0)
        total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
        batch_size = cl_feats.shape[0]
        cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
        # Compute contrastive loss in FP32 by passing FP32 logits
        cdl_loss, mean_similarity = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
        self.dequeue_and_enqueue(total_proj, cl_labels)
        return cdl_loss, mean_similarity

    def get_similar(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        term1 = 0.125 * (diff ** 2) / (0.5 * (sigma1 ** 2 + sigma2 ** 2))
        term2 = 0.5 * torch.log(0.5 * (sigma1 ** 2 + sigma2 ** 2) / (sigma1 * sigma2))
        distance = torch.sum(term1 + term2, dim=-1)
        coefficient = torch.exp(-distance)
        return coefficient.sum()

    def CRLDis(self, common_result , groundTruth_labels, stds_features, samples_index=None):
        var_norm2 = torch.norm(stds_features, p=2, dim=-1).view(-1)
        # Use log-softmax in FP32 to avoid underflow/overflow and log(0)
        var_scores_log = torch.nn.functional.log_softmax(var_norm2.float(), dim=0)
        common_scores_log = torch.nn.functional.log_softmax(common_result.float().view(-1), dim=0)
        var_scores = var_scores_log.exp()
        common_scores = common_scores_log.exp()
        # Symmetric KL in stable form
        kl_loss1 = torch.nn.functional.kl_div(var_scores_log, common_scores, reduction='batchmean') \
                   + torch.nn.functional.kl_div(common_scores_log, var_scores, reduction='batchmean')
        c_scores = []
        for idx, index in enumerate(samples_index): # 记录历史得分
            if index not in self.numCounts:
                self.numCounts[index] = 1
                self.correctCounts[index] = common_result[idx]
            else:
                self.numCounts[index] += 1
                self.correctCounts[index] += common_result[idx]
            c_scores.append(self.correctCounts[index] / self.numCounts[index])
        # Keep shape consistent and probabilities stable
        if len(c_scores) > 0:
            c_scores = torch.stack(c_scores).view(-1).float()
            _ = torch.nn.functional.log_softmax(c_scores, dim=0)  # precompute log-prob if needed later
        
        return kl_loss1 / 2

    @torch.cuda.amp.autocast()  # 添加自动混合精度训练
    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=None, training=False, samples_index=None): 
        """
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        groundTruth_labels: [B, num_classes]

        """

        label_input = label_input.unsqueeze(0).repeat(text.shape[0], 1) # L=> b,L
        label_mask = label_mask.unsqueeze(0).repeat(text.shape[0], 1) # l => b,l

        # print(f"嵌入层调用前 - label_input 数据类型: {label_input.dtype}")

        label_features = self.embedding_layer(label_input) # B, L, D

        # 添加输入检查
        def check_and_fix(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"前向传播输入 {name} 包含异常值，进行修复")
                return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            return tensor

        text = self.text_norm(text)
        visual = self.visual_norm(visual)  
        audio = self.audio_norm(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask) #[B, L, D]
        
        # [batch_size, label_nums, hidden_size]
        label_features = self.label_wise_attention(label_features, visual_output, audio_output, text_output)

        mus, stds = [], []
        for idx in range(len(label_features)):
            label_feature = label_features[idx]
            mu, log_var = self.encoder[idx].encode(label_feature)
            mus.append(mu)
            stds.append(torch.exp(0.5 * log_var))

        common_features = torch.cat((label_features[0], label_features[1],label_features[2]), axis=-1)# [batch_size, label_nums, 3 * hidden_size] 
        common_result = self.common_classfier(common_features).squeeze(-1)  
        common_prob = common_result
        if torch.min(common_prob) < 0 or torch.max(common_prob) > 1:
            common_prob = torch.sigmoid(common_prob)
        
        mus_features = torch.cat(mus, axis=-1)
        stds_features = torch.cat(stds, axis=-1)
        mus_result, stds_result = [], []
        for i in range(len(self.final_classifier)):
            mus_result.append(self.final_classifier[i](mus_features[:,i,:].unsqueeze(1)).squeeze(-1))
        mus_result = torch.cat(mus_result, dim=-1)
        for i in range(len(self.var_classifier1)):
            stds_result.append(self.var_classifier1[i](stds_features[:,i,:].unsqueeze(1)).squeeze(-1))
        stds_result = torch.cat(stds_result, dim=-1)


        # 在关键计算处添加保护
        try:
            # 使用概率域融合，避免直接加权logits带来的失真
            mus_prob = torch.sigmoid(mus_result)
            stds_prob = torch.sigmoid(stds_result)
            final_prob = mus_prob * common_prob + stds_prob * (1 - common_prob)
        except Exception as e:
            print(f"最终概率计算出错: {e}")
            final_prob = torch.sigmoid((mus_result + stds_result) / 2)
        final_prob = torch.nan_to_num(final_prob, nan=0.0, posinf=1.0, neginf=0.0)

        # 统一到概率域用于评估与阈值
        final_labels = getBinaryTensor(final_prob)
        l2_norms = torch.norm(stds_features, p=2, dim=2).data.clone().cpu()
        if training:
            # Compute losses in FP32 to avoid overflow under AMP
            with torch.cuda.amp.autocast(enabled=False):
                # 引入基于类先验的权重，提升召回与稳定性
                pos_w = getattr(self, 'pos_weight', torch.ones(self.num_classes, device=final_prob.device)).float()
                weights = groundTruth_labels.float() * pos_w.unsqueeze(0) + (1.0 - groundTruth_labels.float()) * 1.0
                common_loss = torch.nn.functional.binary_cross_entropy(common_prob.float(), groundTruth_labels.float(), weight=weights, reduction='mean')
                final_loss = torch.nn.functional.binary_cross_entropy(final_prob.float(), groundTruth_labels.float(), weight=weights, reduction='mean')
                cdl_loss, mean_similarity = self.SupContrasts(mus, stds, groundTruth_labels)
                crl_loss = self.CRLDis(common_prob , groundTruth_labels, stds_features, samples_index)
                total_loss = self.task_config.cml*common_loss + self.task_config.final_loss * final_loss + self.task_config.cdl*cdl_loss + self.task_config.crl*crl_loss
            return total_loss, final_labels, groundTruth_labels, final_prob, mean_similarity, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)
        else:
            return final_labels, groundTruth_labels, final_prob, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)

    def SupContrasts(self, mus, stds, groundTruth_labels):
        mus = torch.stack(mus, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        mus = torch.nn.functional.normalize(mus, dim=-1)
        stds = torch.stack(stds, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        stds = torch.nn.functional.normalize(stds, dim=-1)
        total_proj = torch.cat((mus, stds), dim=-1) # [batch_size, 3, 6 labels, 2 * latent_dim]
        total_proj = total_proj.view(-1, total_proj.shape[-1]).float()  # force FP32 for stability
        
        cl_labels = self.get_cl_labels(groundTruth_labels, times=1).view(-1).unsqueeze(-1)
        # Ensure queue tensors are FP32 to match
        cl_feats = torch.cat((total_proj, self.queue.clone().detach().float()), dim=0)
        total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
        batch_size = cl_feats.shape[0]
        cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
        # Compute contrastive loss in FP32 by passing FP32 logits
        cdl_loss, mean_similarity = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
        self.dequeue_and_enqueue(total_proj, cl_labels)
        return cdl_loss, mean_similarity

    def _get_cross_output(self, sequence_output, visual_output, audio_output, common_feature, attention_mask, visual_mask, audio_mask, common_mask):
        # 找到所有模态的最小序列长度
        min_seq_length = min(
            sequence_output.size(1), 
            visual_output.size(1), 
            audio_output.size(1),
            common_feature.size(1)
        )
        
        # 对齐所有模态到最小序列长度
        def align_to_length(tensor, target_length):
            if tensor.size(1) > target_length:
                # 截断到目标长度
                return tensor[:, :target_length, :]
            elif tensor.size(1) < target_length:
                # 填充到目标长度
                pad_size = target_length - tensor.size(1)
                return F.pad(tensor, (0, 0, 0, pad_size))
            else:
                return tensor
        
        def align_mask_to_length(mask, target_length):
            if mask.size(1) > target_length:
                # 截断到目标长度
                return mask[:, :target_length]
            elif mask.size(1) < target_length:
                # 填充到目标长度
                pad_size = target_length - mask.size(1)
                return F.pad(mask, (0, pad_size))
            else:
                return mask
        
        # 对齐所有特征和掩码
        sequence_output = align_to_length(sequence_output, min_seq_length)
        visual_output = align_to_length(visual_output, min_seq_length)
        audio_output = align_to_length(audio_output, min_seq_length)
        common_feature = align_to_length(common_feature, min_seq_length)
        
        attention_mask = align_mask_to_length(attention_mask, min_seq_length)
        visual_mask = align_mask_to_length(visual_mask, min_seq_length)
        audio_mask = align_mask_to_length(audio_mask, min_seq_length)
        common_mask = align_mask_to_length(common_mask, min_seq_length)
        
        # 然后使用简化的融合逻辑
        # =============> visual audio fusion
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        va_concat_mask = torch.cat((audio_mask, visual_mask), dim=1)
        
        audio_type_ = torch.zeros(va_concat_features.size(0), audio_output.size(1), dtype=torch.long, device=va_concat_features.device)
        video_type_ = torch.ones(va_concat_features.size(0), visual_output.size(1), dtype=torch.long, device=va_concat_features.device)
        va_concat_type = torch.cat((audio_type_, video_type_), dim=1)
        
        va_cross_layers, va_pooled_output = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)
        va_cross_output = va_cross_layers[-1]
        # <============= visual audio fusion

        # =============> VisualAudio and text fusion
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        
        text_type_ = torch.zeros(vat_concat_features.size(0), sequence_output.size(1), dtype=torch.long, device=vat_concat_features.device)
        va_type_ = torch.ones(vat_concat_features.size(0), va_cross_output.size(1), dtype=torch.long, device=vat_concat_features.device)
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)
        
        vat_cross_layers, vat_pooled_output = self.vat_cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
        # <============= VisualAudio and text fusion

        # =============> private common fusion
        pc_concate_features = torch.cat((vat_cross_output, common_feature), dim=1)
        pc_concat_mask = torch.cat((vat_concat_mask, common_mask), dim=1)
        
        specific_type = torch.zeros(pc_concate_features.size(0), vat_cross_output.size(1), dtype=torch.long, device=pc_concate_features.device)
        common_type = torch.ones(pc_concate_features.size(0), common_feature.size(1), dtype=torch.long, device=pc_concate_features.device)
        pc_concate_type = torch.cat((specific_type, common_type), dim=1)
        
        pc_cross_layers, pc_pooled_output = self.pc_cross(pc_concate_features, pc_concate_type, pc_concat_mask)
        pc_cross_output = pc_cross_layers[-1]
        # <============= private common fusion

        return pc_pooled_output, pc_cross_output, pc_concat_mask

    def inference(self, text, text_mask, visual, visual_mask, audio, audio_mask, \
            label_input, label_mask, groundTruth_labels=None):
        # 确保输入特征有效
        if text is None or visual is None or audio is None:
            print("警告: 输入特征包含None值")
            # 返回默认值而不是使用可能不正确的特征
            batch = 1 if text is None else text.size(0)
            default_scores = torch.zeros(batch, self.num_classes, device=next(self.parameters()).device)
            return getBinaryTensor(default_scores), groundTruth_labels
        
        # 确保标签输入的正确性
        batch = text.size(0)
        if label_input.dim() == 1:
            label_input = label_input.unsqueeze(0).repeat(batch, 1)
        else:
            # 如果已经有批次维度，确保形状正确
            label_input = label_input[:batch]
            
        if label_mask.dim() == 1:
            label_mask = label_mask.unsqueeze(0).repeat(batch, 1)
        else:
            label_mask = label_mask[:batch]
        
        # 强制转换为 long 类型
        label_input = label_input.long()
        
        # 标签特征提取
        label_features = self.embedding_layer(label_input)
        
        # 特征归一化和数值检查
        text = self.text_norm(text)
        visual = self.visual_norm(visual)   
        audio = self.audio_norm(audio)
        
        # 检查特征是否包含NaN或Inf
        if torch.isnan(text).any() or torch.isinf(text).any():
            print("警告: 文本特征包含NaN或Inf值")
            text = torch.nan_to_num(text, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(visual).any() or torch.isinf(visual).any():
            print("警告: 视觉特征包含NaN或Inf值")
            visual = torch.nan_to_num(visual, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            print("警告: 音频特征包含NaN或Inf值")
            audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 将视觉和音频特征与文本对齐
        if self.aligned == False:
            try:
                visual, _ = self.v2t_ctc(visual)
                audio, _ = self.a2t_ctc(audio)
            except Exception as e:
                print(f"CTC对齐出错: {str(e)}")
                # 如果对齐失败，继续使用原始特征
        
        # 提取各模态输出特征
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask)
        
        # 添加特征差异检查
        if text_output.var().item() < 1e-8 or visual_output.var().item() < 1e-8 or audio_output.var().item() < 1e-8:
            print("警告: 提取的特征方差过小，可能存在梯度消失问题")
        
        # 私有特征提取
        private_text = self.private_feature_extractor[0](text_output)
        private_visual = self.private_feature_extractor[1](visual_output)
        private_audio = self.private_feature_extractor[2](audio_output)
        
        # 公共特征提取
        common_text = self.common_feature_extractor(text_output)
        common_visual = self.common_feature_extractor(visual_output)
        common_audio = self.common_feature_extractor(audio_output)
        
        # 聚合所有模态的公共/私有特征，添加残差连接以增强信息流
        common_feature = (common_text + common_visual + common_audio) / 3.0  # 平均而不是简单相加
        private_feature = private_text + private_visual + private_audio
        
        # 添加层标准化以稳定训练
        if hasattr(self, 'norm_layer'):
            common_feature = self.norm_layer(common_feature)
        
        # 准备掩码
        common_mask = torch.ones_like(text_mask)

        # 融合私有特征、公共特征及各模态掩码，生成跨模态融合特征
        pooled_output, cross_output, cross_mask = self._get_cross_output(private_text, private_visual, private_audio, common_feature, text_mask, visual_mask, audio_mask, common_mask)
        
        # 解码器处理
        decoder_output = self.decoder(label_input, cross_output, label_mask, cross_mask)
        
        # 最终预测分数计算
        cross_predict_scores = self.cross_classifier(decoder_output)
        cross_predict_scores = cross_predict_scores.view(-1, self.num_classes)
        
        predict_scores = cross_predict_scores
        
        # 二值化预测标签
        predict_labels = getBinaryTensor(predict_scores)
        
        # 调整真实标签维度
        if groundTruth_labels is not None:
            groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)
        
        # 同时返回预测标签和分数，便于后续处理
        return predict_labels, groundTruth_labels, predict_scores

    def set_class_priors(self, priors: torch.Tensor):
        """设置类别先验并同步正类权重与若干分类器偏置。
        priors: [num_classes] in (0,1)
        - 保存到 buffer `class_priors`
        - 计算并保存 `pos_weight = (1-p)/p`，用于提升召回
        - 尝试将 common_classfier 的线性层 bias 初始化为 logit(prior)
        - 若存在 var_classifier1 的每类线性层，亦初始化其 bias 为 logit(prior_j)
        """
        with torch.no_grad():
            eps = 1e-6
            p = priors.detach().to(self.class_priors.device).float().clamp(eps, 1.0 - eps)
            self.class_priors.copy_(p)
            pos_w = ((1.0 - p) / p).clamp(min=1.0, max=10.0)
            self.pos_weight.copy_(pos_w)
            # 初始化 common_classfier 的 bias 为 logit(prior)
            try:
                # 结构为 [Linear(in->hidden//2), Dropout, ReLU, Linear(hidden//2->1)]
                if isinstance(self.common_classfier, nn.Sequential) and len(self.common_classfier) >= 4 and isinstance(self.common_classfier[3], nn.Linear):
                    logits_bias = torch.log(p / (1.0 - p))
                    self.common_classfier[3].bias.copy_(logits_bias)
            except Exception:
                pass
            # 初始化每类方差分类器的 bias
            try:
                if hasattr(self, 'var_classifier1') and isinstance(self.var_classifier1, torch.nn.ModuleList):
                    logits_bias = torch.log(p / (1.0 - p))
                    for j, head in enumerate(self.var_classifier1):
                        # head: Sequential(..., Linear(hidden//2 -> 1))
                        for layer in reversed(head):
                            if isinstance(layer, torch.nn.Linear) and layer.out_features == 1:
                                layer.bias.copy_(logits_bias[j:j+1])
                                break
            except Exception:
                pass
        return self.class_priors.clone(), self.pos_weight.clone()
    
    def get_cl_labels(self, labels, times = 1): # [batch_size, 6 labels], times
        text_labels = torch.zeros_like(labels) + labels
        visual_labels = torch.zeros_like(labels) + labels
        audio_labels = torch.zeros_like(labels) + labels

        text_cl_labels = torch.zeros_like(text_labels, dtype=torch.long)
        visual_cl_labels = torch.zeros_like(visual_labels, dtype=torch.long)
        audio_cl_labels = torch.zeros_like(audio_labels, dtype=torch.long)

        example_idx, label_idx = torch.where(text_labels >= 0.5) 
        text_cl_labels[example_idx, label_idx] = label_idx 
        example_idx, label_idx = torch.where(text_labels < 0.5) 
        text_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1 

        example_idx, label_idx = torch.where(visual_labels >= 0.5) 
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 2
        example_idx, label_idx = torch.where(visual_labels < 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 3

        example_idx, label_idx = torch.where(audio_labels >= 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 4
        example_idx, label_idx = torch.where(audio_labels < 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 5

        cl_labels = torch.stack([text_cl_labels, visual_cl_labels, audio_cl_labels], dim=1) # 64, 3, 6
        # The above step divides labels for each modality into 0~11, with each label split into two values: 0,6; 1,7; 2,8; 3,9; 4,10; 5,11
        cl_labels = cl_labels.to(torch.int)
        if times > 1:
            final_cl_labels = torch.cat([cl_labels, cl_labels], dim=1)
            for i in range(2, times):
                final_cl_labels = torch.cat([final_cl_labels, cl_labels], dim=1)
        else:
            final_cl_labels = cl_labels
        return final_cl_labels
    
    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float() # Find elements with similar labels across rows and columns
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask

    def dequeue_and_enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        """将当前批次的特征与标签入队，维护一个循环队列。
        - feats: [N, D]
        - labels: [N, 1]
        """
        if not (hasattr(self, 'queue') and hasattr(self, 'queue_label') and hasattr(self, 'queue_ptr')):
            return
        with torch.no_grad():
            # 保持与队列 dtype/device 一致，避免 cat/mask 时类型不匹配
            dev = self.queue.device
            feats = feats.detach().to(dev)
            labels = labels.detach().to(dev).to(self.queue_label.dtype)
            queue_size = self.queue.shape[0]
            feat_dim = self.queue.shape[1]
            # 如果特征维度不匹配，进行截断或填充以避免错误
            if feats.shape[1] != feat_dim:
                if feats.shape[1] > feat_dim:
                    feats = feats[:, :feat_dim]
                else:
                    pad = feat_dim - feats.shape[1]
                    feats = torch.nn.functional.pad(feats, (0, pad))
            batch_size = feats.shape[0]
            ptr = int(self.queue_ptr.item())

            if batch_size >= queue_size:
                # 批次过大：只保留最后 queue_size 项
                self.queue.copy_(feats[-queue_size:])
                self.queue_label.copy_(labels[-queue_size:])
                self.queue_ptr[0] = 0
                return

            end = ptr + batch_size
            if end <= queue_size:
                self.queue[ptr:end].copy_(feats)
                self.queue_label[ptr:end].copy_(labels)
            else:
                first = queue_size - ptr
                self.queue[ptr:].copy_(feats[:first])
                self.queue_label[ptr:].copy_(labels[:first])
                remain = end - queue_size
                if remain > 0:
                    self.queue[:remain].copy_(feats[first:first+remain])
                    self.queue_label[:remain].copy_(labels[first:first+remain])
            self.queue_ptr[0] = (ptr + batch_size) % queue_size

    def compute_dynamic_pos_weight(self, labels: torch.Tensor) -> torch.Tensor:
        """根据当前批次标签动态计算正类加权，返回逐元素权重张量。
        - 对每个类别计算正类比例 p_j，pos_weight_j = clamp(((1-p_j)/(p_j+1e-6)), 1.0, 5.0)
        - 逐元素权重: weight = y*pos_weight + (1-y)*1.0
        """
        # labels: [B, C]
        eps = 1e-6
        # 防止混合精度下 half 类型导致不稳定
        labels_f = labels.detach().float()
        pos_ratio = labels_f.mean(dim=0)  # [C]
        pos_weight = ((1.0 - pos_ratio) / (pos_ratio + eps)).clamp(min=1.0, max=5.0)  # [C]
        # 广播到批次维度
        pos_weight_b = pos_weight.unsqueeze(0).expand_as(labels_f)
        weights = labels_f * pos_weight_b + (1.0 - labels_f) * 1.0
        return weights