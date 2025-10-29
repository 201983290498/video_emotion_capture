import torch
import os
# import sys
# import threading
import time
# import pickle
# import pandas as pd
# import numpy as np
# import gc
import concurrent.futures
from transformers import (BertModel, BertTokenizer, WhisperProcessor,
                         WhisperForConditionalGeneration)
from humanomni.mm_utils import process_video, process_audio
# from humanomni import model_init
# from humanomni.model.humanomni_arch import HumanOmniMetaForCausalLM
# from humanomni.model.humanomni_model import HumanOmniQwen2ForCausalLM
from humanomni.model import VLLMs, VLLMConfigs
# from humanomni.constants import NUM_FRAMES
# import einops
# import math
import traceback
# from tqdm import tqdm
# import queue
# import psutil

def batch_process_videos(video_paths, visual_processor=None, num_frames=32, aspect_ratio='pad', max_workers=4):
    """批量处理视频，提取帧数据（使用多线程并行处理）
    
    Args:
        video_paths: 视频文件路径列表
        visual_processor: 视觉处理器对象
        num_frames: 采样的帧数
        aspect_ratio: 宽高比处理方式
        max_workers: 最大并行工作线程数
        
    Returns:
        tuple: (处理后的视频帧张量列表, 有效索引列表)
    """
    batch_frames = []
    valid_indices = []
    
    def process_single_video(args):
        """处理单个视频的辅助函数"""
        i, video_path = args
        try:
            frames = process_video(
                video_path, 
                visual_processor, 
                aspect_ratio=aspect_ratio, 
                num_frames=num_frames
            )
            return i, frames, None
        except Exception as e:
            return i, None, str(e)
    
    # 使用线程池并行处理视频
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_single_video, (i, video_path)): i 
            for i, video_path in enumerate(video_paths)
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                idx, frames, error = future.result()
                if frames is not None:
                    batch_frames.append(frames)
                    valid_indices.append(i)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 视频帧处理失败 {video_paths[i]}: {error}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 处理视频帧时出错 {video_paths[i]}: {str(e)}")
    
    # 按原始索引顺序排序
    if batch_frames and valid_indices:
        sorted_pairs = sorted(zip(valid_indices, batch_frames), key=lambda x: x[0])
        valid_indices, batch_frames = zip(*sorted_pairs)
        valid_indices = list(valid_indices)
        batch_frames = list(batch_frames)
    
    return batch_frames, valid_indices

def batch_extract_video_features(video_paths, humanomni_model, visual_processor=None, num_frames=32, aspect_ratio='pad', prompts=None, bert_tokenizer=None, batch_size=16, device=None):
    """批量从视频中提取视觉特征
    
    Args:
        video_paths: 视频文件路径列表
        humanomni_model: HumanOmni模型对象
        visual_processor: 视觉处理器对象
        num_frames: 采样的帧数
        aspect_ratio: 宽高比处理方式
        prompts: 用于指导特征提取的文本提示
        bert_tokenizer: BERT分词器对象
        batch_size: 批量处理大小
        
    Returns:
        list: 处理后的视频特征张量列表
    """
    # 确保模型处于评估模式
    humanomni_model.eval()
    # 使用提供的device或模型的device
    if device is None:
        device = humanomni_model.device
    
    # 性能优化：启用CUDA优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # 获取模型的数据类型
    model_dtype = next(humanomni_model.parameters()).dtype
    # print(f"模型数据类型: {model_dtype}")
    
    batch_video_features = [None] * len(video_paths)
    
    # 批量处理视频 - 优化批量大小以平衡性能和延迟
    optimized_batch_size = min(batch_size, 4)  # 减少批量大小以降低延迟
    for i in range(0, len(video_paths), optimized_batch_size):
        batch_video_paths = video_paths[i:i + optimized_batch_size]
        
        # 使用批量处理函数处理视频帧
        batch_frames, valid_indices = batch_process_videos(
            batch_video_paths, 
            visual_processor, 
            num_frames, 
            aspect_ratio,
            max_workers=min(8, len(batch_video_paths))  # 根据批次大小调整线程数
        )
        
        if not batch_frames:
            print(f"批次 {i//batch_size + 1} 所有视频帧处理都失败了")
            continue
        
        # 确保数据类型与模型匹配
        processed_frames = []
        for frames in batch_frames:
            if frames is not None:
                frames = frames.to(dtype=model_dtype, device=device)
                processed_frames.append(frames)
            else:
                processed_frames.append(None)
        
        # 过滤掉None值
        valid_processed_frames = []
        valid_frame_indices = []
        for idx, frames in zip(valid_indices, processed_frames):
            if frames is not None:
                valid_processed_frames.append(frames)
                valid_frame_indices.append(idx)
        
        if not valid_processed_frames:
            print(f"批次 {i//batch_size + 1} 没有有效的视频帧数据")
            continue
        
        # 准备批量输入格式
        input_images = [(frames, 'video') for frames in valid_processed_frames]
        
        # 准备BERT输入
        inputs_bert_dict = None
        if bert_tokenizer and prompts:
            try:
                # 为每个视频仅使用一个提示，确保批次大小匹配
                prompt_per_video = [prompts[0]] * len(valid_processed_frames) if isinstance(prompts, (list, tuple)) and len(prompts) > 0 else [prompts] * len(valid_processed_frames)
                temp_inputs_bert = bert_tokenizer(
                    prompt_per_video,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs_bert_dict = {}
                for k, v in temp_inputs_bert.items():
                    if isinstance(v, torch.Tensor):
                        inputs_bert_dict[k] = v.to(device, non_blocking=True).long()
            except Exception as e:
                print(f"准备BERT输入时出错: {str(e)}")
                inputs_bert_dict = None
        
        # 批量提取特征 - 使用自动混合精度和优化的内存管理
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=model_dtype):
            # try:
            if inputs_bert_dict is not None:
                video_features_list = humanomni_model.encode_images_or_videos(
                    input_images,
                    device,
                    inputs_bert_dict
                )
            else:
                video_features_list = humanomni_model.encode_images_or_videos(
                    input_images,
                    device
                )
                
            # 及时同步以避免内存泄漏
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 将结果映射回原始位置
            success_count = 0
            for idx, (original_idx, feature) in enumerate(zip(valid_frame_indices, video_features_list)):
                if feature is not None:
                    # 确保特征在正确的设备和数据类型上
                    feature = feature.to(device=device, dtype=model_dtype)
                    batch_video_features[i + original_idx] = feature
                    success_count += 1
                else:
                    print(f"警告: 视频 {batch_video_paths[original_idx]} 的特征提取返回None")
                        
            # except Exception as e:
            #     print(f"批量提取视觉特征时出错: {str(e)}")
            #     traceback.print_exc()
                
            #     # 回退到逐个处理 - 优化内存使用
            #     print("回退到逐个处理...")
            #     individual_success_count = 0
            #     for j, (frames, original_idx) in enumerate(zip(valid_processed_frames, valid_frame_indices)):
            #         try:
            #             single_input = [(frames, 'video')]
                          
            #             if inputs_bert_dict is not None:
            #                 # 为单个视频创建对应的BERT输入
            #                 single_bert_input = {}
            #                 for k, v in inputs_bert_dict.items():
            #                     if isinstance(v, torch.Tensor) and v.shape[0] > j:
            #                         # 确保正确的设备传输
            #                         single_bert_input[k] = v[j:j+1].to(device, non_blocking=True)
            #                 # 使用自动混合精度
            #                 with torch.autocast(device_type=device.type, dtype=model_dtype):
            #                     single_features = humanomni_model.encode_images_or_videos(
            #                         single_input,
            #                         device,
            #                         single_bert_input
            #                     )
            #             else:
            #                 # 使用自动混合精度
            #                 with torch.autocast(device_type=device.type, dtype=model_dtype):
            #                     single_features = humanomni_model.encode_images_or_videos(
            #                         single_input,
            #                         device
            #                     )
                          
            #             # 及时同步和清理
            #             if device.type == 'cuda':
            #                 torch.cuda.synchronize()
            #                 torch.cuda.empty_cache()
                        
            #             if single_features and len(single_features) > 0 and single_features[0] is not None:
            #                 feature = single_features[0].to(device=device, dtype=model_dtype)
            #                 batch_video_features[i + original_idx] = feature
            #                 individual_success_count += 1
            #             else:
            #                 print(f"单个视频处理返回空特征: {batch_video_paths[original_idx]}")
            #         except Exception as single_e:
            #             print(f"单个视频处理失败 {batch_video_paths[original_idx]}: {str(single_e)}")
                
            #     print(f"逐个处理成功 {individual_success_count}/{len(valid_processed_frames)} 个视频")
    
    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 统计结果
    success_count = sum(1 for feat in batch_video_features if feat is not None)
    return batch_video_features

def batch_process_audios(video_paths, sample_rate=16000, processor=None):
    """批量处理音频数据
    
    Args:
        video_paths: 视频文件路径列表
        sample_rate: 采样率
        processor: 音频处理器对象
        
    Returns:
        tuple: (有效音频张量列表, 有效索引列表)
    """
    batch_audio_tensors = []
    valid_indices = []
    
    for i, video_path in enumerate(video_paths):
        try:
            audio_tensor, _ = process_audio(
                video_path, 
                sample_rate=sample_rate, 
                processor=processor
            )
            
            if isinstance(audio_tensor, torch.Tensor):
                # 标准化音频张量格式
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif len(audio_tensor.shape) > 3:
                    audio_tensor = audio_tensor[0] if audio_tensor.size(0) > 0 else audio_tensor
                
                # 处理异常值
                if torch.isnan(audio_tensor).any():
                    audio_tensor = torch.nan_to_num(audio_tensor, nan=0.0)
                if torch.isinf(audio_tensor).any():
                    audio_tensor = torch.nan_to_num(audio_tensor, posinf=1.0, neginf=-1.0)
                
                audio_tensor = audio_tensor.to(dtype=torch.float32, device='cpu')
                batch_audio_tensors.append(audio_tensor)
                valid_indices.append(i)
            else:
                print(f"警告：音频数据不是张量类型: {type(audio_tensor)}")
        except Exception:
            print(f"处理音频时出错 {video_path}")
    
    return batch_audio_tensors, valid_indices

def batch_extract_audio_features(video_paths, humanomni_model, device=None, sample_rate=16000, batch_size=16):
    """批量从视频中提取音频特征（优化的真正批量处理）
    
    Args:
        video_paths: 视频文件路径列表
        humanomni_model: HumanOmni模型对象
        device: 运行设备
        sample_rate: 采样率
        batch_size: 批量处理大小
        
    Returns:
        list: 处理后的音频特征张量列表
    """
    humanomni_model.eval()
    target_device = device if device else humanomni_model.device
    model_dtype = next(humanomni_model.parameters()).dtype
    
    # 获取音频处理器
    audio_processor = None
    if hasattr(humanomni_model, 'get_audio_tower'):
        audio_tower = humanomni_model.get_audio_tower()
        if audio_tower and hasattr(audio_tower, 'audio_processor'):
            audio_processor = audio_tower.audio_processor
    
    batch_audio_features = [None] * len(video_paths)
    
    for i in range(0, len(video_paths), batch_size):
        batch_video_paths = video_paths[i:i + batch_size]
        
        # 批量处理音频
        batch_audio_tensors, valid_indices = batch_process_audios(
            batch_video_paths, sample_rate, audio_processor
        )
        
        if not batch_audio_tensors:
            # 为整个批次创建默认特征
            for j in range(len(batch_video_paths)):
                default_features = torch.randn(500, 896, device=target_device, dtype=model_dtype) * 0.01
                batch_audio_features[i + j] = default_features.cpu()
            continue
        
        try:
            # 批量堆叠音频张量
            max_length = max(tensor.shape[1] for tensor in batch_audio_tensors)
            padded_tensors = []
            
            for tensor in batch_audio_tensors:
                if tensor.shape[1] < max_length:
                    pad_size = max_length - tensor.shape[1]
                    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            
            batched_tensor = torch.cat(padded_tensors, dim=0)
            batched_tensor = batched_tensor.to(device=target_device, dtype=model_dtype, non_blocking=True)
            
            # 批量提取特征
            with torch.no_grad():
                batch_features = humanomni_model.encode_audios(batched_tensor)
            
            # 处理特征并映射回位置
            for idx, (original_idx, feature) in enumerate(zip(valid_indices, batch_features)):
                if feature is not None:
                    if len(feature.shape) == 3 and feature.shape[0] == 1:
                        feature = feature.squeeze(0)
                    batch_audio_features[i + original_idx] = feature.cpu()
                    
        except Exception:
            print("批量提取音频特征时出错")
            for j, audio_tensor in enumerate(batch_audio_tensors):
                original_idx = valid_indices[j]
                # try:
                audio_tensor = audio_tensor.to(device=target_device, dtype=model_dtype, non_blocking=True)
                with torch.no_grad():
                    audio_features = humanomni_model.encode_audios(audio_tensor.unsqueeze(0))
                
                if audio_features is not None and len(audio_features) > 0:
                    feature = audio_features[0]
                    if len(feature.shape) == 3 and feature.shape[0] == 1:
                        feature = feature.squeeze(0)
                    batch_audio_features[i + original_idx] = feature.cpu()
                else:
                    default_features = torch.randn(500, 896, device=target_device, dtype=model_dtype) * 0.01
                    batch_audio_features[i + original_idx] = default_features.cpu()
                # except Exception as e_inner:
                #     print(f"逐个提取音频特征时出错 {batch_video_paths[original_idx]}: {str(e_inner)}")
                #     default_features = torch.randn(500, 896, device=target_device, dtype=model_dtype) * 0.01
                #     batch_audio_features[i + original_idx] = default_features.cpu()
        
        # 为无效音频创建默认特征
        for j in range(len(batch_video_paths)):
            if batch_audio_features[i + j] is None:
                default_features = torch.randn(500, 896, device=target_device, dtype=model_dtype) * 0.01
                batch_audio_features[i + j] = default_features.cpu()
    
    return batch_audio_features

def batch_extract_text_features(video_paths, whisper_model=None, whisper_processor=None, bert_model=None, bert_tokenizer=None, device=None, sample_rate=16000, batch_size=16):
    """批量从视频中提取文本特征（优化的真正批量处理）
    
    Args:
        video_paths: 视频文件路径列表
        whisper_model: Whisper模型对象，用于音频转录
        whisper_processor: Whisper处理器对象
        bert_model: BERT模型对象，用于提取文本特征
        bert_tokenizer: BERT分词器对象
        device: 运行设备
        sample_rate: 采样率
        batch_size: 批量处理大小
        
    Returns:
        list: 处理后的文本特征张量列表
    """
    target_device = device if device else (
        next(bert_model.parameters()).device if bert_model else
        next(whisper_model.parameters()).device if whisper_model else
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    target_dtype = torch.float16
    if bert_model:
        target_dtype = next(bert_model.parameters()).dtype
    elif whisper_model:
        target_dtype = next(whisper_model.parameters()).dtype
    
    batch_text_features = [None] * len(video_paths)
    
    # 第一阶段：批量预处理所有音频
    all_audio_data = []
    for video_path in video_paths:
        audio_tensor, sr = process_audio(video_path, sample_rate=sample_rate, processor=None)
        all_audio_data.append((audio_tensor, video_path))
    
    # 第二阶段：批量处理转录和文本特征
    for i in range(0, len(video_paths), batch_size):
        batch_video_paths = video_paths[i:i + batch_size]
        batch_audio_data = all_audio_data[i:i + batch_size]
        
        # 收集有效音频
        valid_audio_indices = []
        valid_audio_tensors = []
        for j, (audio_tensor, video_path) in enumerate(batch_audio_data):
            if audio_tensor is not None and isinstance(audio_tensor, torch.Tensor):
                valid_audio_indices.append(j)
                valid_audio_tensors.append(audio_tensor)
        
        # 批量转录
        batch_transcriptions = [""] * len(batch_video_paths)
        if whisper_model and whisper_processor and valid_audio_tensors:
            # 批量处理音频转录
            audio_arrays = []
            for audio_tensor in valid_audio_tensors:
                audio_np = audio_tensor.squeeze().cpu().numpy()
                audio_arrays.append(audio_np)
            
            # 使用Whisper批量处理
            inputs = whisper_processor(
                audio_arrays, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            input_features = inputs.input_features.to(device=target_device, dtype=target_dtype, non_blocking=True)
            
            # 批量生成转录
            generated_ids = whisper_model.generate(
                input_features,
                language='en',
                max_new_tokens=128,
                num_beams=1
            )
            transcriptions = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 分配转录结果
            for idx, transcription in zip(valid_audio_indices, transcriptions):
                batch_transcriptions[idx] = transcription
                    
        
        # 批量提取文本特征
        valid_text_indices = [j for j, transcript in enumerate(batch_transcriptions) if transcript.strip()]
        valid_transcripts = [batch_transcriptions[j] for j in valid_text_indices]
        
        if valid_transcripts and bert_model and bert_tokenizer:
            # try:
            # 批量BERT处理
            with torch.no_grad():
                inputs = bert_tokenizer(
                    valid_transcripts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                input_ids = inputs.input_ids.to(device=target_device, dtype=torch.long, non_blocking=True)
                attention_mask = inputs.attention_mask.to(device=target_device, dtype=torch.long, non_blocking=True)
                
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                
                # 计算特征均值并扩展维度
                text_features = torch.mean(last_hidden_state, dim=1)
                text_features = text_features.repeat_interleave(500, dim=0).view(len(valid_transcripts), 500, -1)
                
                # 分配特征
                for idx, feature in zip(valid_text_indices, text_features):
                    batch_text_features[i + idx] = feature.cpu()
                        
            # except Exception:
            #     print("批量提取文本特征时出错")
            #     # 回退到逐个处理
            #     for j, transcript in enumerate(batch_transcriptions):
            #         if transcript.strip() and bert_model and bert_tokenizer:
            #             try:
            #                 inputs = bert_tokenizer(
            #                     transcript,
            #                     return_tensors='pt',
            #                     padding=True,
            #                     truncation=True,
            #                     max_length=512
            #                 )
                            
            #                 input_ids = inputs.input_ids.to(device=target_device, dtype=torch.long, non_blocking=True)
            #                 attention_mask = inputs.attention_mask.to(device=target_device, dtype=torch.long, non_blocking=True)
                            
            #                 outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            #                 last_hidden_state = outputs.last_hidden_state
                            
            #                 text_feature = torch.mean(last_hidden_state, dim=1)
            #                 text_feature = text_feature.repeat(500, 1)
                            
            #                 batch_text_features[i + j] = text_feature.cpu()
            #             except Exception:
            #                 print("逐个提取文本特征时出错")
        
        # 为无效文本创建默认特征
        for j in range(len(batch_video_paths)):
            if batch_text_features[i + j] is None:
                default_features = torch.randn(500, 768, device=target_device, dtype=target_dtype) * 0.01
                batch_text_features[i + j] = default_features.cpu()
    
    return batch_text_features


def get_video_files(folder_path):
    """获取所有视频文件路径"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(root, file))
    return video_files

def batch_extract_multimodal_features(video_paths, models=None, model_path=None, visual_prompts=None, batch_size=16, device=None):
    """批量从视频中提取多模态特征"""
    
    # 初始化模型（如果未提供）
    if models is None:
        models = {}
        # 确定使用的设备
        if device is None:
            # 检查CUDA可用性
            if torch.cuda.is_available():
                device = torch.device('cuda')
                torch.backends.cudnn.benchmark = True
            else:
                device = torch.device('cpu')
        
        # 加载HumanOmni模型
        config = VLLMConfigs["HumanOmni_qwen2"].from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        config.mm_vision_tower = os.path.join(model_path, "vision_tower") if os.path.exists(os.path.join(model_path, "vision_tower")) else config.mm_vision_tower
        config.mm_audio_tower = os.path.join(model_path, "audio_tower") if os.path.exists(os.path.join(model_path, "audio_tower")) else config.mm_audio_tower
        
        # 优化：根据GPU内存选择合适的数据类型
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        torch_dtype = torch.float16 if gpu_memory > 8 else torch.float32
        
        humanomni_model = VLLMs["HumanOmni_qwen2"].from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device)
        
        # 显式设置所有组件的数据类型
        humanomni_model = humanomni_model.to(dtype=torch_dtype)
        
        # 显式加载视觉塔和音频塔并设置数据类型
        vision_tower = humanomni_model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch_dtype)
            if not models.get('visual_processor'):
                models['visual_processor'] = vision_tower.image_processor
        
        audio_tower = humanomni_model.get_audio_tower()
        if audio_tower is not None:
            if not audio_tower.is_loaded:
                audio_tower.load_model()
            audio_tower.to(device=device, dtype=torch_dtype)
            if not models.get('audio_processor'):
                models['audio_processor'] = audio_tower.audio_processor
        
        # 确保模型处于评估模式
        humanomni_model.eval()
        models['humanomni_model'] = humanomni_model
        
        # 加载BERT模型并设置相同的数据类型
        if not models.get('bert_tokenizer') or not models.get('bert_model'):
            if not models.get('bert_tokenizer'):
                models['bert_tokenizer'] = BertTokenizer.from_pretrained("/data/testmllm/models/bert-base-uncased")
            models['bert_model'] = BertModel.from_pretrained(
                "/data/testmllm/models/bert-base-uncased", 
                torch_dtype=torch_dtype,
                use_cache=True
            ).to(device).eval()
    
    optimized_batch_size = min(batch_size, 16)  # 增加批量大小以提高并行效率
    
    # print(f"开始批量提取多模态特征，视频数量: {len(video_paths)}，批量大小: {optimized_batch_size}")
    
    # 批量提取各种特征
    default_prompts = [
        "What emotions are expressed in this video?",
        "Describe the facial expressions and body language",
        "What is the emotional tone of the scene?"
    ]
    prompts_to_use = visual_prompts if visual_prompts is not None else default_prompts
    
    visual_features_list = batch_extract_video_features(
        video_paths, 
        models['humanomni_model'],
        models.get('visual_processor', None),
        prompts=prompts_to_use,
        bert_tokenizer=models.get('bert_tokenizer', None),
        batch_size=optimized_batch_size,
        device=device
    )
    
    # 打印视觉特征统计
    try:
        for idx, vf in enumerate(visual_features_list):
            if vf is not None:
                m = float(vf.mean().cpu())
                s = float(vf.std().cpu())
                # print(f"视觉特征[{idx}] mean={m:.6f}, std={s:.6f}")
    except Exception:
        pass

    audio_features_list = batch_extract_audio_features(
        video_paths, 
        models['humanomni_model'],
        device=device,
        batch_size=optimized_batch_size
    )
    
    # 打印音频特征统计
    try:
        for idx, af in enumerate(audio_features_list):
            if af is not None:
                m = float(af.mean().cpu())
                s = float(af.std().cpu())
                # print(f"音频特征[{idx}] mean={m:.6f}, std={s:.6f}")
    except Exception:
        pass

    text_features_list = None
    if all(key in models for key in ['whisper_model', 'whisper_processor', 'bert_model', 'bert_tokenizer']):
        text_features_list = batch_extract_text_features(
            video_paths,
            whisper_model=models.get('whisper_model'),
            whisper_processor=models.get('whisper_processor'),
            bert_model=models.get('bert_model'),
            bert_tokenizer=models.get('bert_tokenizer'),
            device=device,
            batch_size=optimized_batch_size
        )
    
    # 打印文本特征统计
    try:
        if text_features_list:
            for idx, tf in enumerate(text_features_list):
                if tf is not None:
                    m = float(tf.mean().cpu())
                    s = float(tf.std().cpu())
                    # print(f"文本特征[{idx}] mean={m:.6f}, std={s:.6f}")
    except Exception:
        pass

    # 统一打包返回结果，兼容 lddu_inference 期望的结构
    results = []
    total = len(video_paths)
    for i in range(total):
        vf = visual_features_list[i] if i < len(visual_features_list) else None
        af = audio_features_list[i] if i < len(audio_features_list) else None
        tf = None
        if text_features_list and i < len(text_features_list):
            tf = text_features_list[i]

        # 若文本特征未提供，则使用低幅度噪声填充，避免全零
        if tf is None:
            # 尝试与BERT模型dtype保持一致；若不可用，则回退到FP32
            try:
                target_dtype = next(models['bert_model'].parameters()).dtype if models and ('bert_model' in models) else torch.float32
            except Exception:
                target_dtype = torch.float32
            tf = (torch.randn(500, 768, dtype=target_dtype) * 0.01).cpu()

        results.append({
            'visual': vf,
            'audio': af,
            'text': tf
        })

    return results


# 辅助：并行配置与模型加载（供 lddu_inference 使用）

def optimize_parallel_config(gpu_count: int, total_videos: int):
    """根据GPU数量与任务规模返回一个简单的并行配置字典 - 优化版本"""
    try:
        # 对于实时情感分析，优化批量大小和工作线程数
        # 减少批量大小以降低延迟，减少工作线程数以节省资源
        batch_size = min(4, max(1, total_videos))  # 减少批量大小
        num_workers = max(1, min(2, os.cpu_count() or 1))  # 减少工作线程数
        return {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }
    except Exception:
        return {'batch_size': 1, 'num_workers': 1}


def load_models_for_device(device=None, model_path=None):
    """加载HumanOmni、Whisper与BERT，返回 models 字典供特征提取使用"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)

    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'R1-Omni-0.5B')

    # 配置与数据类型选择
    config = VLLMConfigs["HumanOmni_qwen2"].from_pretrained(
        model_path,
        trust_remote_code=True
    )
    vision_tower_path = os.path.join(model_path, "vision_tower")
    audio_tower_path = os.path.join(model_path, "audio_tower")
    if os.path.exists(vision_tower_path):
        config.mm_vision_tower = vision_tower_path
    if os.path.exists(audio_tower_path):
        config.mm_audio_tower = audio_tower_path

    if device.type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            torch_dtype = torch.float16 if gpu_memory > 8 else torch.float32
        except Exception:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 加载HumanOmni主模型
    humanomni_model = VLLMs["HumanOmni_qwen2"].from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device).eval()

    # 子塔与处理器
    visual_processor = None
    audio_processor = None
    vision_tower = humanomni_model.get_vision_tower()
    if vision_tower is not None:
        if not getattr(vision_tower, 'is_loaded', True):
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch_dtype)
        visual_processor = getattr(vision_tower, 'image_processor', None)

    audio_tower = humanomni_model.get_audio_tower()
    if audio_tower is not None:
        if not getattr(audio_tower, 'is_loaded', True):
            audio_tower.load_model()
        audio_tower.to(device=device, dtype=torch_dtype)
        audio_processor = getattr(audio_tower, 'audio_processor', None)

    # 加载BERT（带回退）
    bert_tokenizer = None
    bert_model = None
    try:
        bert_tokenizer = BertTokenizer.from_pretrained("/data/testmllm/models/bert-base-uncased")
        bert_model = BertModel.from_pretrained(
            "/data/testmllm/models/bert-base-uncased",
            torch_dtype=torch_dtype,
            use_cache=True
        ).to(device).eval()
    except Exception:
        try:
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert_model = BertModel.from_pretrained(
                "bert-base-uncased",
                torch_dtype=torch_dtype,
                use_cache=True
            ).to(device).eval()
        except Exception:
            pass

    # 加载Whisper（用于语音转录，严格本地加载，避免远程请求）
    whisper_processor = None
    whisper_model = None
    try:
        whisper_processor = WhisperProcessor.from_pretrained(
            "/data/testmllm/models/whisper-large-v3",
            local_files_only=True
        )
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "/data/testmllm/models/whisper-large-v3",
            torch_dtype=torch_dtype,
            local_files_only=True
        ).to(device).eval()
    except Exception:
        # 本地模型不可用时，保持None，跳过语音转录以避免404
        whisper_processor = None
        whisper_model = None

    return {
        'humanomni_model': humanomni_model,
        'visual_processor': visual_processor,
        'audio_processor': audio_processor,
        'bert_tokenizer': bert_tokenizer,
        'bert_model': bert_model,
        'whisper_processor': whisper_processor,
        'whisper_model': whisper_model,
    }

# 强制关闭 HuggingFace 远程访问，避免 httpx 404 噪音
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')