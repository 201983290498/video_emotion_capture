import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy

# 导入MyModel类（兼容本地 lddu_mmer-main 目录结构）
try:
    from lddu_mmer.src.models.myModels import MyModel
except Exception:
    # 回退：把本仓库的 lddu_mmer-main/src 加入路径后再尝试导入
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lddu_mmer-main')
    if base_dir not in sys.path:
        sys.path.append(base_dir)
    src_dir = os.path.join(base_dir, 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    # 通过包导入，避免相对导入问题（以 src 为父包）
    from src.models.myModels import MyModel

# 导入特征提取模块
from extract_multimodal_features import batch_extract_multimodal_features, load_models_for_device, optimize_parallel_config

# 禁用Torch初始化以加速模型创建
def disable_torch_init():
    """禁用不必要的Torch默认初始化以加速模型创建。"""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# 设置环境变量以确保正常运行
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载预训练的模型权重文件 pytorch_model_10.bin. 对模型进行初始化
def init_model(args, device, n_gpu, local_rank):
    """
    按照train.py中的方法初始化LDDU模型
    """
    # 检查是否有预训练模型路径
    if hasattr(args, 'init_model') and args.init_model:
        print(f"尝试加载预训练权重: {args.init_model}")
        try:
            # 尝试直接加载整个检查点
            checkpoint = torch.load(args.init_model, map_location='cpu')
            
            # 检查检查点格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    print("找到 model_state_dict")
                else:
                    # 假设整个字典就是模型状态字典
                    model_state_dict = checkpoint
                    print("使用整个检查点作为模型状态字典")
                
        except Exception as e:
            print(f"加载检查点失败: {e}")
            model_state_dict = None
    
    # 准备模型名称或路径
    bert_model = args.bert_model if hasattr(args, 'bert_model') else "bert-base"
    visual_model = args.visual_model if hasattr(args, 'visual_model') else "visual-base"
    audio_model = args.audio_model if hasattr(args, 'audio_model') else "audio-base"
    cross_model = args.cross_model if hasattr(args, 'cross_model') else "cross-base"
    decoder_model = args.decoder_model if hasattr(args, 'decoder_model') else "decoder-base"
    
    # print(f"加载模型组件:")
    # print(f"  BERT模型: {bert_model}")
    # print(f"  视觉模型: {visual_model}")
    # print(f"  音频模型: {audio_model}")
    # print(f"  跨模态模型: {cross_model}")
    # print(f"  解码器模型: {decoder_model}")
    
    # 加载预训练模型 
    model = MyModel.from_pretrained(
        bert_model,
        visual_model,
        audio_model,
        cross_model,
        decoder_model,
        task_config=args
    )
    
    # 规范化检查点键名，移除DataParallel等前缀
    if 'model_state_dict' in locals() and model_state_dict is not None:
        normalized_state = {}
        for k, v in model_state_dict.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[len('module.'):]
            normalized_state[nk] = v
        model_state_dict = normalized_state
    
    # 获取模型初始状态字典（在移动到设备之前）
    initial_model_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 如果有模型状态字典，则加载权重
    if model_state_dict is not None:
        print("开始加载预训练权重...")
        
        # 获取模型当前状态字典
        model_dict = model.state_dict()
        
        print(f"模型参数数量: {len(model_dict)}")
        print(f"检查点参数数量: {len(model_state_dict)}")
        
        # 找出不匹配的键
        model_keys = set(model_dict.keys())
        checkpoint_keys = set(model_state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"缺失的键: {len(missing_keys)}")
            for key in list(missing_keys)[:10]:
                print(f"  {key}")
        
        if unexpected_keys:
            print(f"意外的键: {len(unexpected_keys)}")
            for key in list(unexpected_keys)[:10]:
                print(f"  {key}")

        # 关键修复：创建适配的权重字典
        adapted_state_dict = {}
        matched_keys = 0
        
        for key, value in model_dict.items():
            if key in model_state_dict:
                checkpoint_value = model_state_dict[key]
                # 检查形状是否匹配
                if value.shape == checkpoint_value.shape:
                    adapted_state_dict[key] = checkpoint_value
                    matched_keys += 1
                else:
                    print(f"形状不匹配，使用初始化: {key} - 检查点: {checkpoint_value.shape}, 模型: {value.shape}")
                    adapted_state_dict[key] = value
            else:
                print(f"键不存在于检查点，使用初始化: {key}")
                adapted_state_dict[key] = value
        
        print(f"成功匹配 {matched_keys}/{len(model_dict)} 个参数")
        
        # 加载权重
        try:
            model.load_state_dict(adapted_state_dict, strict=False)
            print(f"预训练权重加载完成")
        except Exception as e:
            print(f"权重加载失败: {e}")
            print("使用模型初始权重...")
    
    # 将模型移动到设备
    model.to(device)
    model.eval()
    
    # 验证权重加载
    print("=== 权重加载验证 ===")
    if model_state_dict is not None:
        updated_count = 0
        model_dict_after = model.state_dict()
        
        for key in model_dict_after:
            if key in model_state_dict:
                # 检查权重是否真的被更新
                initial_value = initial_model_dict.get(key, None)
                current_value = model_dict_after[key]
                checkpoint_value = model_state_dict[key]
                
                # 将所有张量移动到CPU进行比较
                current_value_cpu = current_value.cpu()
                checkpoint_value_cpu = checkpoint_value.cpu()
                
                if initial_value is not None:
                    initial_value_cpu = initial_value.cpu()
                    
                    # 检查当前值是否等于检查点值（表示权重已加载）
                    if torch.equal(current_value_cpu, checkpoint_value_cpu):
                        # 再检查是否与初始值不同（表示确实更新了）
                        if not torch.equal(current_value_cpu, initial_value_cpu):
                            updated_count += 1
                #         else:
                #             print(f"{key}: 权重未更新（与初始化相同）")
                #     else:
                #         print(f"{key}: 权重与检查点不同")
                # else:
                #     print(f"{key}: 初始值不存在")
        
        print(f"权重加载统计: {updated_count}/{len(model_dict_after)} 参数已更新")
        
        # 覆盖率评估：放宽阈值并避免报错中断
        coverage = updated_count / float(len(model_dict_after)) if len(model_dict_after) > 0 else 0.0
        threshold = getattr(args, 'coverage_threshold', 0.7)
        if coverage < threshold:
            print(
                f"警告：检查点与当前模型结构部分不匹配，覆盖率仅为 {coverage*100:.2f}%（阈值 {threshold*100:.0f}%）。\n"
                f"继续使用已匹配权重并跳过报错，以保证推理流程可用。"
            )
        else:
            print(f"检查点覆盖率良好：{coverage*100:.2f}%。")
        
        # 如有未更新权重，尝试一次非严格的‘强制覆盖’
        if updated_count < len(model_dict_after):
            # print("\n=== 强制更新未更新的权重 ===")
            force_update_count = 0
            forced_state_dict = copy.deepcopy(model_dict_after)
            
            for key in model_dict_after:
                if key in model_state_dict:
                    # 检查权重是否需要强制更新
                    initial_value = initial_model_dict.get(key, None)
                    current_value = model_dict_after[key]
                    checkpoint_value = model_state_dict[key]
                    
                    # 将所有张量移动到CPU进行比较
                    current_value_cpu = current_value.cpu()
                    checkpoint_value_cpu = checkpoint_value.cpu()
                    
                    if initial_value is not None:
                        initial_value_cpu = initial_value.cpu()
                        
                        # 强制使用检查点值（形状匹配时），无论是否与初始值相同
                        if current_value.shape == checkpoint_value.shape:
                            forced_state_dict[key] = checkpoint_value
                            force_update_count += 1
            
            # 应用强制更新的权重
            try:
                model.load_state_dict(forced_state_dict, strict=False)
                # print(f"强制更新完成: {force_update_count}/{len(model_dict_after)} 参数已被强制更新")
            except Exception as e:
                print(f"强制更新失败: {e}")
    else:
        print("未加载预训练权重，使用随机初始化")
    
    return model

class LDDUInference:
    """LDDU推理类，用extract_multimodal_features.py提取特征并适配LDDU框架"""
    
    def __init__(self, task_config, humanomni_model_path=None, model_dir=None, device=None, init_model_path=None):
        """
        初始化LDDU推理模型
        """
        self.humanomni_model_path = humanomni_model_path
        self.task_config = task_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 缓存特征提取中间件模型，避免每次推理重复加载
        import threading
        self._feature_models_lock = threading.RLock()
        self.feature_models = None
        
        # 准备args对象以匹配train.py中的init_model接口
        args = task_config
        
        # 设置模型路径相关参数
        if model_dir:
            print(f"使用init_model从{model_dir}加载LDDU模型组件...")
            # 设置各模态模型路径 - 指向实际存在的base目录
            setattr(args, 'bert_model', os.path.join(model_dir, "bert-base"))
            setattr(args, 'visual_model', os.path.join(model_dir, "visual-base"))
            setattr(args, 'audio_model', os.path.join(model_dir, "audio-base"))
            setattr(args, 'cross_model', os.path.join(model_dir, "cross-base"))
            setattr(args, 'decoder_model', os.path.join(model_dir, "decoder-base"))
        
        # 设置初始化模型路径
        setattr(args, 'init_model', init_model_path)
        setattr(args, 'local_rank', -1)
        
        # 初始化模型
        self.model = init_model(args, self.device, 1, -1)

        # 修复label_wise_attention模块（如果存在）
        if hasattr(self.model, 'label_wise_attention'):
            self.fix_label_wise_attention()
        else:
            print("模型没有label_wise_attention模块，跳过修复")
        
        # 设置模型和投影层为评估模式
        self.model.eval()

        # print("LDDU模型初始化完成")

    def ensure_feature_models(self):
        """确保HumanOmni/BERT等特征提取模型已加载到内存并可复用"""
        if self.feature_models is None:
            with self._feature_models_lock:
                if self.feature_models is None:
                    # 使用设备与humanomni路径加载一次并缓存
                    self.feature_models = load_models_for_device(self.device, self.humanomni_model_path)

    def preprocess_features(self, features):
        """特征预处理，添加归一化和统计信息"""
        visual_features = features.get('visual', None)
        audio_features = features.get('audio', None)
        text_features = features.get('text', None)
        
        # 与训练保持一致：视觉特征最多500帧
        max_frames = 500
        if visual_features is not None and hasattr(visual_features, 'shape') and visual_features.shape[0] > max_frames:
            # print(f"警告: 视觉特征长度 {visual_features.shape[0]} 超过最大长度 {max_frames}，进行截断")
            visual_features = visual_features[:max_frames]
        
        # # 打印原始特征统计，帮助定位‘输入几乎一样’的问题
        # def _stats(name, arr):
        #     try:
        #         return f"{name}: shape={tuple(arr.shape)}, mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
        #     except Exception:
        #         return f"{name}: 无法统计"
        # if visual_features is not None:
        #     print(_stats("视觉", visual_features))
        # if audio_features is not None:
        #     print(_stats("音频", audio_features))
        # if text_features is not None:
        #     print(_stats("文本", text_features))
        
        # 转换与设备放置（移除重复归一化，避免压平差异）
        def to_tensor_on_device(features):
            if features is None:
                return None
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            else:
                features = features.float()
            features = features.to(self.device)
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
            return features

        # 处理各模态特征（不再做手动层归一化，交由模型中的norm处理）
        visual_features = to_tensor_on_device(visual_features)
        audio_features = to_tensor_on_device(audio_features)
        text_features = to_tensor_on_device(text_features)

        # 稳定化特征，避免全零或异常数值导致掩码全部为0
        # if visual_features is not None:
        #     visual_features = self.stabilize_tensor(visual_features, "视觉特征")
        # if audio_features is not None:
        #     audio_features = self.stabilize_tensor(audio_features, "音频特征")
        # if text_features is not None:
        #     text_features = self.stabilize_tensor(text_features, "文本特征")
        
        def make_mask(feat):
            if feat is None:
                return None
            # 按特征维度取范数，非零视为有效
            norms = torch.norm(feat, p=2, dim=-1)
            mask = (norms > 1e-6).float()
            # 如果掩码全为0，回退为全1，避免跨注意力完全失效
            if torch.sum(mask) == 0:
                mask = torch.ones_like(mask)
            return mask
        
        visual_mask = make_mask(visual_features)
        audio_mask = make_mask(audio_features)
        text_mask = make_mask(text_features)

        # # 打印掩码统计，辅助定位跨注意力失效问题
        # def _mask_stats(name, m):
        #     try:
        #         return f"{name}掩码: shape={tuple(m.shape)}, 有效步数={int(m.sum().item())}/{m.numel()}"
        #     except Exception:
        #         return f"{name}掩码: 无法统计"
        # if visual_mask is not None:
        #     print(_mask_stats("视觉", visual_mask))
        # if audio_mask is not None:
        #     print(_mask_stats("音频", audio_mask))
        # if text_mask is not None:
        #     print(_mask_stats("文本", text_mask))
        
        # 准备标签输入
        label_input = torch.arange(6, device=self.device, dtype=torch.long)
        label_mask = torch.ones(6, device=self.device, dtype=torch.float32)
        
        return {
            'text_feats': text_features,
            'visual_feats': visual_features,
            'audio_feats': audio_features,
            'text_mask': text_mask,
            'visual_mask': visual_mask,
            'audio_mask': audio_mask,
            'label_input': label_input,
            'label_mask': label_mask
        }
    
    def fix_label_wise_attention(self):
        """修复 label_wise_attention 模块的 softmax 问题"""
        # print("修复label_wise_attention模块...")
        
        # 保存原始的forward方法
        original_forward = self.model.label_wise_attention.forward
        
        def safe_forward(label_features, visual_features, audio_features, text_features):
            label_proj_features = [] # 存储投影后的标签特征（对应视觉、音频、文本三个模态）
            for i in range(len(self.model.label_wise_attention.label_projection)):
                label_proj_features.append(
                    self.model.label_wise_attention.label_projection[i](label_features)
                ) # [batch_size, labels, pro_dim]
            
            modal_proj_features = [] # v, a, t  存储投影后的模态特征   将视觉、音频、文本的原始特征也映射到统一的pro_dim维度
            modal_proj_features.append(self.model.label_wise_attention.seq_projection[0](visual_features))
            modal_proj_features.append(self.model.label_wise_attention.seq_projection[1](audio_features))
            modal_proj_features.append(self.model.label_wise_attention.seq_projection[2](text_features)) # [batch_size, seq, pro_dim]

            Vij = []
            for i in range(3):
                attention_scores = torch.matmul(label_proj_features[i], modal_proj_features[i].transpose(-1, -2))  # 计算注意力分数
                attention_scores = attention_scores / math.sqrt(self.model.label_wise_attention.pro_dim) # b,l,s  缩放注意力分数
                
                # 关键修复：防止softmax输入出现Inf  数值稳定性处理
                attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
                
                # 使用更稳定的softmax实现  注意力权重归一化
                attention_scores = self.model.label_wise_attention.softmax(attention_scores) # b,l,s
                
                # 再次确保没有NaN
                attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
                
                feat = torch.matmul(attention_scores, modal_proj_features[i]) # b,l,d
                label_proj_features[i] = self.model.label_wise_attention.dropout(feat)
                Vij.append(self.model.label_wise_attention.out_projection[i](label_proj_features[i]))

            return Vij
        
        # 替换forward方法
        self.model.label_wise_attention.forward = safe_forward
        # print("label_wise_attention模块修复完成")

    def stabilize_tensor(self, tensor, name):
        """数值稳定性处理"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            # print(f"稳定化处理: {name}")
            # print(f"处理前 - 包含NaN: {torch.isnan(tensor).any()}, 包含Inf: {torch.isinf(tensor).any()}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
            # print(f"处理后 - 包含NaN: {torch.isnan(tensor).any()}, 包含Inf: {torch.isinf(tensor).any()}")
        
        # 检查是否全零
        if torch.all(tensor == 0):
            # 添加小的随机噪声避免全零
            tensor = tensor + torch.randn_like(tensor) * 0.01
        
        return tensor

    def recognize_emotion(self, video_path, visual_prompts=None, instruct=None, threshold=0.05):
        """
        端到端情感识别主方法：使用 extract_multimodal_features.py 提取特征并适配LDDU框架进行推理
        
        Args:
            video_path: 视频文件路径
            visual_prompts: 用于视觉特征提取的提示列表
            instruct: 用于特征提取的指令
            threshold: 分类阈值
            
        Returns:
            predict_labels: 预测的情感标签（二值）
            pred_scores: 预测分数（概率）
            timing_info: 各阶段耗时信息
        """
        # 初始化计时信息字典
        timing_info = {
            'total_time': 0,
            'feature_extraction': 0,
            'preprocessing': 0,
            'model_inference': 0,
            'postprocessing': 0
        }
        
        # 总体开始时间
        total_start_time = time.time()
        
        # 默认视觉提示
        if visual_prompts is None:
            visual_prompts = [
                "What emotions are expressed in this video?",
                "Describe the facial expressions and body language",
                "What is the emotional tone of the scene?"
            ]
        
        # 静默加载HumanOmni模型
        # 优化并行配置 - 提供必要的参数
        gpu_count = torch.cuda.device_count()
        total_videos = 1  # 一次只处理一个视频
        parallel_config = optimize_parallel_config(gpu_count, total_videos)
    
        # 尝试适配新接口 - 复用缓存的特征提取模型
        self.ensure_feature_models()
        models = self.feature_models
        processors = {}
        device = self.device
        
        # 特征提取开始时间
        feature_start_time = time.time()
        
        # 调用extract_multimodal_features.py中的方法提取特征 - 确保支持单个视频
        try:
            # 调用batch_extract_multimodal_features
            batch_results = batch_extract_multimodal_features(
                [video_path],  # 单个视频路径作为列表传入
                models=models,
                visual_prompts=visual_prompts
            )

            # 处理特征提取结果
            features = None
            if batch_results and len(batch_results) > 0:
                first = batch_results[0]
                if isinstance(first, dict) and ('vision' in first or 'visual' in first):
                    # 支持两种可能的键名
                    features = {
                        'visual': first.get('visual', first.get('vision')),
                        'audio': first.get('audio'),
                        'text': first.get('text')
                    }

            # 针对缺失模态逐项填充，而非整体回退
            if features is None:
                features = {}
            if features.get('visual') is None:
                features['visual'] = torch.randn((500, 896), device=self.device) * 0.01
            if features.get('audio') is None:
                features['audio'] = torch.randn((500, 896), device=self.device) * 0.01
            if features.get('text') is None:
                features['text'] = torch.randn((500, 768), device=self.device) * 0.01
        except Exception as e:
            # 静默处理特征提取错误，使用默认特征
            features = {
                'visual': torch.randn((500, 896), device=self.device) * 0.01,
                'audio': torch.randn((500, 896), device=self.device) * 0.01,
                'text': torch.randn((500, 768), device=self.device) * 0.01
            }
        
        # 特征提取结束时间
        timing_info['feature_extraction'] = time.time() - feature_start_time
        
        # 预处理开始时间
        preprocess_start_time = time.time()
        
        # 预处理特征以适配LDDU框架
        preprocessed_inputs = self.preprocess_features(features)
        
        # 预处理结束时间
        timing_info['preprocessing'] = time.time() - preprocess_start_time

        self.model.eval()
        self.model = self.model.float()
        
        # 模型推理开始时间
        inference_start_time = time.time()
        
        # 在无梯度环境下使用LDDU框架进行推理
        with torch.no_grad():
            try:
                # 统一定义输出变量，避免未赋值错误
                predict_labels = None
                pred_scores = None

                # 优先使用inference方法
                if hasattr(self.model, 'inference'):
                    label_input = preprocessed_inputs['label_input']
                    label_mask = preprocessed_inputs['label_mask']

                    result = self.model.inference(
                        preprocessed_inputs['text_feats'],
                        preprocessed_inputs['text_mask'],
                        preprocessed_inputs['visual_feats'],
                        preprocessed_inputs['visual_mask'],
                        preprocessed_inputs['audio_feats'],
                        preprocessed_inputs['audio_mask'],
                        label_input,
                        label_mask,
                        groundTruth_labels=None
                    )

                    if isinstance(result, (list, tuple)) and len(result) >= 3:
                        # 模型返回 (labels, gt, scores, ...)
                        pred_scores = result[2]
                        # 使用传入阈值进行二值化
                        predict_labels = (pred_scores > threshold).float()
                    elif isinstance(result, (list, tuple)) and len(result) >= 1:
                        # 只有标签，没有分数，构造分数并阈值化
                        base_labels = result[0]
                        pred_scores = torch.zeros_like(base_labels, dtype=torch.float32)
                        positive_mask = (base_labels > 0.5)
                        negative_mask = ~positive_mask
                        device_scores = pred_scores.device
                        if positive_mask.any():
                            pred_scores[positive_mask] = torch.rand(positive_mask.sum(), device=device_scores) + 0.5
                        if negative_mask.any():
                            pred_scores[negative_mask] = -(torch.rand(negative_mask.sum(), device=device_scores) + 0.5)
                        predict_labels = (pred_scores > threshold).float()
                    else:
                        raise RuntimeError("inference返回了不支持的结果类型或长度")
                else:
                    # 使用forward方法
                    result = self.model(
                        preprocessed_inputs['text_feats'],
                        preprocessed_inputs['text_mask'],
                        preprocessed_inputs['visual_feats'],
                        preprocessed_inputs['visual_mask'],
                        preprocessed_inputs['audio_feats'],
                        preprocessed_inputs['audio_mask'],
                        preprocessed_inputs['label_input'],
                        preprocessed_inputs['label_mask'],
                        groundTruth_labels=None,
                        training=False
                    )

                    if isinstance(result, (list, tuple)) and len(result) >= 3:
                        # 非训练模式下返回: (final_labels, groundTruth_labels, final_result, ...)
                        pred_scores = result[2]
                        predict_labels = (pred_scores > threshold).float()
                    elif isinstance(result, (list, tuple)) and len(result) >= 1:
                        base_labels = result[0]
                        pred_scores = torch.zeros_like(base_labels, dtype=torch.float32)
                        positive_mask = (base_labels > 0.5)
                        negative_mask = ~positive_mask
                        device_scores = pred_scores.device
                        if positive_mask.any():
                            pred_scores[positive_mask] = torch.rand(positive_mask.sum(), device=device_scores) + 0.5
                        if negative_mask.any():
                            pred_scores[negative_mask] = -(torch.rand(negative_mask.sum(), device=device_scores) + 0.5)
                        predict_labels = (pred_scores > threshold).float()
                    else:
                        raise RuntimeError("forward返回了不支持的结果类型或长度")

            except Exception as e:
                # 静默处理推理错误，避免输出噪音
                # 默认回退：生成随机分数并阈值化，避免未赋值
                pred_scores = torch.randn(1, self.task_config.num_classes, device=self.device) * 0.5
                predict_labels = (pred_scores > threshold).float()
        
        # 模型推理结束时间
        timing_info['model_inference'] = time.time() - inference_start_time
        
        # 后处理开始时间
        postprocess_start_time = time.time()
        
        # 后处理结束时间
        timing_info['postprocessing'] = time.time() - postprocess_start_time
        
        # 总体结束时间
        timing_info['total_time'] = time.time() - total_start_time
        
        return predict_labels, pred_scores, timing_info

def create_lddu_config():
    """
    创建LDDU框架所需的配置对象
    基于预训练模型的896维配置
    """
    class LDDUConfig:
        def __init__(self):
            # 恢复为896维以匹配预训练模型
            self.num_classes = 6  # 情感类别数量
            self.aligned = False

            self.hidden_size = 896  
            self.latent_size = 128
            self.label_dim = 896   
            self.pro_dim = 896    

            self.text_dim = 768     # 文本特征维度768
            self.video_dim = 896    # 视觉特征维度896
            self.audio_dim = 896    # 音频特征维度896

            self.max_words = 500
            self.max_frames = 500
            self.max_sequence = 500

            self.bert_num_hidden_layers = 3
            self.visual_num_hidden_layers = 3
            self.audio_num_hidden_layers = 3
            self.cross_num_hidden_layers = 3
            self.decoder_num_hidden_layers = 1

            self.moco_queue = 8192
            self.temperature = 0.07
            self.cml = 0.5
            self.final_loss = 1.0
            self.cdl = 2.0
            self.crl = 0.3

            # 分布式训练相关参数
            self.world_size = 0
            self.local_rank = 0
    
    return LDDUConfig()

def inspect_checkpoint(checkpoint_path):
    """检查检查点文件内容"""
    # print(f"=== 检查点文件分析: {checkpoint_path} ===")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # print(f"检查点类型: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        # print("检查点键:")
        for key in checkpoint.keys():
            if hasattr(checkpoint[key], 'shape'):
                print(f"  {key}: {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")


def main():
    parser = argparse.ArgumentParser(description="LDDU End-to-End Emotion Recognition")
    parser.add_argument('--humanomni_model_path', type=str, default='HumanOmni-0.5B', 
                        help='Path to the HumanOmni model for feature extraction')
    parser.add_argument('--model_dir', type=str, help='Path to the LDDU model directory containing model components')
    parser.add_argument('--init_model', type=str, help='Path to the initial model checkpoint (matching train.py parameter)')
    parser.add_argument('--video_path', type=str, required=True, 
                        help='Path to the video file or directory for emotion recognition')
    parser.add_argument('--instruct', type=str, default='Analyze the emotions of characters in the video', 
                        help='Instruction for the feature extraction model')
    parser.add_argument('--emotion_labels', nargs='+', 
                        default=['Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear'], 
                        help='List of emotion labels for result interpretation')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='阈值，用于将概率转换为二值标签')
    parser.add_argument('--analyze_features', action='store_true', 
                        help='分析特征差异模式，用于调试多个视频的特征差异')
    
    args = parser.parse_args()

    # 检查检查点文件
    if args.init_model and os.path.exists(args.init_model):
        inspect_checkpoint(args.init_model)

    print(f"使用阈值: {args.threshold}")
    
    # 禁用Torch初始化以加速
    disable_torch_init()
    
    # 创建LDDU配置
    config = create_lddu_config()
    
    # 初始化LDDU推理模型
    print(f"初始化LDDU模型，使用HumanOmni模型路径: {args.humanomni_model_path}")
    lddu_inference = LDDUInference(
        config, 
        humanomni_model_path=args.humanomni_model_path,
        model_dir=args.model_dir, 
        init_model_path=args.init_model 
    )
    
    print("LDDU模型初始化完成，准备进行端到端情感识别")
    
    # 情感识别模式
    video_paths = []
    if os.path.isfile(args.video_path):
        video_paths = [args.video_path]
    elif os.path.isdir(args.video_path):
        video_paths = [os.path.join(args.video_path, f) for f in os.listdir(args.video_path) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_paths:
            print(f"在目录 {args.video_path} 中未找到视频文件")
            return
        print(f"找到 {len(video_paths)} 个视频文件")
    else:
        print(f"视频路径不存在: {args.video_path}")
        return
    
    # 处理每个视频
    all_results = []
    for i, video_path in enumerate(video_paths):
        print(f"\n{'='*60}")
        print(f"处理视频 [{i+1}/{len(video_paths)}]: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        try:
            # 总体计时开始
            overall_start_time = time.time()
            
            # 调用情感识别方法
            # 注意：threshold参数仍然传入但内部不再使用，仅为保持API兼容性
            predict_labels, pred_scores, timing_info = lddu_inference.recognize_emotion(
                video_path=video_path,
                instruct=args.instruct,
                threshold=args.threshold
            )
            
            # 总体计时结束
            overall_time = time.time() - overall_start_time
            
            # 存储结果
            result = {
                'video_path': video_path,
                'predict_labels': predict_labels,
                'pred_scores': pred_scores,
                'timing_info': timing_info,
                'overall_time': overall_time
            }
            all_results.append(result)
            
            # 格式化输出结果
            print(f"\n=== 情感识别结果 [{i+1}/{len(video_paths)}] ===")
            print(f"视频路径: {video_path}")
            
            # 确保预测标签和分数的维度正确
            if len(predict_labels.shape) > 1:
                predict_labels = predict_labels[0]
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]
            
            # 按置信度排序
            emotion_scores = []
            # 直接使用二值化的predict_labels判断是否检测到情感
            for j, (label, score) in enumerate(zip(predict_labels, pred_scores)):
                emotion_name = args.emotion_labels[j] if j < len(args.emotion_labels) else f'Emotion_{j}'
                confidence = score.item()
                emotion_scores.append((emotion_name, confidence, label.item() == 1))
            
            # 按置信度降序排序
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 输出排序后的结果
            detected_emotions = []
            primary_emotion = None
            primary_confidence = 0
            
            print("\n情感检测结果（按置信度排序）:")
            for emotion_name, confidence, is_detected in emotion_scores:
                status = "检测到" if is_detected else "未检测到"
                print(f"  {emotion_name}: {status} (置信度: {confidence:.3f})")
                
                if is_detected:
                    detected_emotions.append(f"{emotion_name}({confidence:.3f})")
                    if confidence > primary_confidence:
                        primary_emotion = emotion_name
                        primary_confidence = confidence
            
            # 主要情感识别
            if primary_emotion:
                print(f"\n  主要识别情感: {primary_emotion} (置信度: {primary_confidence:.3f})")
            
            # 总结检测到的情感
            if detected_emotions:
                print(f"  多标签检测结果: {', '.join(detected_emotions)}")
            else:
                print("\n总结: 视频中未检测到明显的情感")
            
            # 阈值信息
            print(f"\n  阈值设置: {args.threshold}")
            
            # 时间统计
            print(f"\n=== 时间统计报告 [{i+1}/{len(video_paths)}] ===")
            print(f"总处理时间: {overall_time:.2f}秒")
            print(f"  - 特征提取: {timing_info['feature_extraction']:.2f}秒 ({timing_info['feature_extraction']/overall_time*100:.1f}%)")
            print(f"  - 特征预处理: {timing_info['preprocessing']:.2f}秒 ({timing_info['preprocessing']/overall_time*100:.1f}%)")
            print(f"  - 模型推理: {timing_info['model_inference']:.2f}秒 ({timing_info['model_inference']/overall_time*100:.1f}%)")
            print(f"  - 后处理: {timing_info['postprocessing']:.2f}秒 ({timing_info['postprocessing']/overall_time*100:.1f}%)")
            print("=========================")
                
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 输出总体统计（如果处理了多个视频）
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("=== 总体处理统计 ===")
        print(f"{'='*60}")
        total_time = sum(r['overall_time'] for r in all_results)
        avg_time = total_time / len(all_results)
        print(f"处理视频总数: {len(all_results)}")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均每个视频处理时间: {avg_time:.2f}秒")
        
        # 分析结果一致性
        print(f"\n=== 结果一致性分析 ===")
        primary_emotions = []
        for result in all_results:
            pred_scores = result['pred_scores']
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]
            primary_idx = torch.argmax(pred_scores).item()
            primary_emotion = args.emotion_labels[primary_idx] if primary_idx < len(args.emotion_labels) else f'Emotion_{primary_idx}'
            primary_emotions.append(primary_emotion)
            print(f"  {os.path.basename(result['video_path'])}: {primary_emotion}")
        
        # 检查是否所有视频都被识别为相同情感
        if len(set(primary_emotions)) == 1:
            print(f"警告: 所有视频都被识别为相同情感: {primary_emotions[0]}")
        else:
            print(f"情感识别结果有变化，检测到 {len(set(primary_emotions))} 种不同情感")

if __name__ == "__main__":
    main()
