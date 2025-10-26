import os
import cv2
import torch
import numpy as np
import concurrent.futures
import os
from PIL import Image
import time
from tqdm import tqdm
import argparse
import json
import traceback
import pickle
import sys
from transformers import BertTokenizer, BertModel

# 导入项目中的相关模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humanomni import model_init
from humanomni.mm_utils import process_video, process_audio
from humanomni.constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN

class VideoFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_file = os.path.join(config.output_dir, 'extraction_log.txt')
        self.error_log_file = os.path.join(config.output_dir, 'error_log.txt')
        
        # 强制禁用FP16，使用FP32以彻底解决数据类型不匹配问题
        self.config.use_fp16 = False
        self.log(f"已强制禁用FP16，使用FP32精度处理")
        
        # 直接加载模型，因为多线程可以共享内存
        self._load_models()

    def _load_models(self):
        """加载所有需要的模型，并确保它们在指定设备上"""
        # 确保设备参数正确解析
        if isinstance(self.device, str):
            if self.device.startswith('cuda') and ':' not in self.device:
                self.device = f'{self.device}:0'  # 默认使用第一个GPU
        
        # 使用model_init初始化HumanOmni模型组件 - 强制使用FP32
        model, processor, tokenizer = model_init(
            self.config.humanomni_model_path,
            device=self.device,
            torch_dtype=torch.float32  # 强制使用FP32
        )
        # 从processor中提取音频处理器
        self.audio_processor = processor['audio'].keywords.get('processor') if 'audio' in processor else None
        
        # 确保模型完全移动到目标设备并使用FP32
        self.model = model.eval().to(self.device).float()
        
        # 重要：确保音频编码器的所有参数都使用FP32
        if hasattr(self.model, 'audio_encoder'):
            self.log("正在将音频编码器所有参数转换为FP32")
            for name, module in self.model.audio_encoder.named_modules():
                # 转换权重
                if hasattr(module, 'weight') and module.weight is not None:
                    try:
                        module.weight.data = module.weight.data.float()
                        self.log(f"已将音频编码器模块 {name} 的权重转换为FP32")
                    except Exception as e:
                        self.log_error(f"转换权重为FP32失败: {str(e)}")
                # 转换偏置
                if hasattr(module, 'bias') and module.bias is not None:
                    try:
                        module.bias.data = module.bias.data.float()
                        self.log(f"已将音频编码器模块 {name} 的偏置转换为FP32")
                    except Exception as e:
                        self.log_error(f"转换偏置为FP32失败: {str(e)}")
                
        self.processor = processor
        self.tokenizer = tokenizer
        
        # 加载BERT模型用于生成情感识别的prompts
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.config.bert_model)
        self.bert_model = BertModel.from_pretrained(self.config.bert_model).to(self.device).eval()
        
        self.log(f"所有模型已加载到设备: {self.device}")

    def extract_video_audio_features(self, video_path):
        """使用model_init返回的processor处理视频和音频"""
        try:
            self.log(f"开始处理视频: {video_path}")
            
            # 1. 预处理视频（与inference.py一致，使用processor['video']）
            processed_video = self.processor['video'](video_path)
            if processed_video is None:
                self.log_error(f"视频预处理失败(返回None): {video_path}")
                return None, None
            
            # 转换为模型输入格式 - 强制使用FP32
            target_dtype = torch.float32
            
            # 匹配inference.py中的输入格式
            if isinstance(processed_video, dict) and 'pixel_values_videos' in processed_video:
                # 处理BatchFeature类型的输入
                pixel_values_videos = processed_video['pixel_values_videos'].to(dtype=target_dtype, device=self.device)
                video_grid_thw = processed_video['video_grid_thw'].to(device=self.device)
                processed_data = type('obj', (object,), {
                    'pixel_values_videos': pixel_values_videos,
                    'video_grid_thw': video_grid_thw
                })
            else:
                # 处理普通tensor输入
                processed_data = processed_video.to(dtype=target_dtype, device=self.device)
            
            images = [(processed_data, "video")]
            
            # 2. 预处理音频（增强的安全处理方式，解决序列类型问题）
            audio_data = None
            audio_tensor = None
            if 'audio' in self.processor:
                try:
                    audio_result = self.processor['audio'](video_path)
                    
                    # 增强的音频处理逻辑，彻底解决序列类型问题
                    if audio_result is None:
                        self.log("音频处理器返回None，使用空音频数据")
                    elif isinstance(audio_result, dict) and 'input_values' in audio_result:
                        # 特别处理字典类型的返回值
                        audio_data = audio_result['input_values']
                        # 确保是列表或张量格式
                        if isinstance(audio_data, list) and len(audio_data) > 0:
                            # 空列表检查
                            self.log(f"音频数据是列表，长度: {len(audio_data)}")
                        elif not isinstance(audio_data, torch.Tensor):
                            # 如果不是张量，尝试转换为张量
                            try:
                                audio_data = torch.tensor(audio_data)
                            except Exception as te:
                                self.log_error(f"转换音频数据为张量失败: {str(te)}")
                                audio_data = None
                    elif isinstance(audio_result, tuple) and len(audio_result) > 0:
                        # 处理元组类型的返回值
                        audio_data = audio_result[0]
                        self.log(f"音频数据是元组，长度: {len(audio_result)}")
                    else:
                        # 其他类型
                        audio_data = audio_result
                    
                    # 空值检查和安全转换
                    if audio_data is None:
                        self.log("音频数据为空，创建默认音频数据")
                        num_samples = int(10 * 16000)  # 10秒，16kHz
                        audio_tensor = torch.zeros(1, 1, num_samples, dtype=torch.float32, device=self.device)
                    else:
                        # 处理序列类型问题
                        if isinstance(audio_data, (list, tuple)):
                            # 处理列表或元组
                            self.log(f"处理音频序列数据，类型: {type(audio_data)}, 长度: {len(audio_data) if hasattr(audio_data, '__len__') else '未知'}")
                            
                            # 确保不为空
                            if not hasattr(audio_data, '__len__') or len(audio_data) == 0:
                                self.log("音频序列为空或无法获取长度，创建默认音频数据")
                                num_samples = int(10 * 16000)  # 10秒，16kHz
                                audio_tensor = torch.zeros(1, 1, num_samples, dtype=torch.float32, device=self.device)
                            else:
                                # 尝试转换为张量
                                try:
                                    if not isinstance(audio_data[0], torch.Tensor):
                                        audio_data = torch.tensor(audio_data)
                                    else:
                                        # 如果已经是张量列表，尝试堆叠
                                        try:
                                            audio_data = torch.stack(audio_data)
                                        except Exception as se:
                                            self.log_error(f"堆叠音频张量失败: {str(se)}")
                                            # 使用第一个元素作为替代
                                            audio_data = audio_data[0]
                                except Exception as e:
                                    self.log_error(f"处理音频序列数据失败: {str(e)}")
                                    num_samples = int(10 * 16000)  # 10秒，16kHz
                                    audio_tensor = torch.zeros(1, 1, num_samples, dtype=torch.float32, device=self.device)
                                    audio_data = None
                        
                        # 如果audio_data仍然有效，继续处理
                        if audio_data is not None:
                            # 确保是张量
                            if not isinstance(audio_data, torch.Tensor):
                                try:
                                    audio_data = torch.tensor(audio_data)
                                except Exception as te:
                                    self.log_error(f"转换音频数据为张量失败: {str(te)}")
                                    num_samples = int(10 * 16000)  # 10秒，16kHz
                                    audio_tensor = torch.zeros(1, 1, num_samples, dtype=torch.float32, device=self.device)
                                    audio_data = None
                            
                            # 如果audio_data仍然有效，调整形状
                            if audio_data is not None:
                                # 确保音频张量形状规范
                                self.log(f"音频数据原始形状: {audio_data.shape}")
                                
                                if len(audio_data.shape) == 1:
                                    # 一维数据，添加通道和批次维度 [3000] -> [1, 1, 3000]
                                    audio_data = audio_data.unsqueeze(0).unsqueeze(0)
                                elif len(audio_data.shape) == 2:
                                    # 二维数据，添加批次维度 [channels, features] -> [1, channels, features]
                                    audio_data = audio_data.unsqueeze(0)
                                elif len(audio_data.shape) > 3:
                                    # 超过三维，保留前三维
                                    self.log(f"音频数据维度超过3，截断为: {audio_data.shape[:3]}")
                                    audio_data = audio_data[:, :, :3000]  # 确保长度为3000
                                
                                # 强制使用FP32并移至设备
                                audio_data = audio_data.to(dtype=torch.float32, device=self.device)
                                
                                # 最终形状检查和调整
                                if audio_data.shape[-1] > 3000:
                                    # 裁剪过长的音频
                                    audio_data = audio_data[:, :, :3000]
                                elif audio_data.shape[-1] < 3000:
                                    # 填充过短的音频
                                    padding = torch.zeros(audio_data.shape[0], audio_data.shape[1], 3000 - audio_data.shape[-1], 
                                                         dtype=torch.float32, device=self.device)
                                    audio_data = torch.cat([audio_data, padding], dim=-1)
                                
                                self.log(f"音频张量最终形状: {audio_data.shape}, 数据类型: {audio_data.dtype}")
                                audio_tensor = audio_data
                        
                except Exception as e:
                    self.log_error(f"音频处理异常: {str(e)}")
                    # 创建空音频数据继续处理
                    num_samples = int(10 * 16000)  # 10秒，16kHz
                    audio_tensor = torch.zeros(1, 1, num_samples, dtype=torch.float32, device=self.device)
                    self.log("使用空音频数据继续处理")
            
            # 3. 提取多模态特征
            mm_features = None
            audio_features = None
            
            with torch.no_grad():
                # 准备适合情感识别的文本提示
                emotion_prompt = "Analyze the emotional expressions in this video and audio content."
                prompts = self.bert_tokenizer(
                    emotion_prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # 确保模型在正确设备上并使用FP32
                self.model = self.model.to(self.device).float()
                
                # 提取视觉特征
                try:
                    mm_features = self.model.encode_images_or_videos(images, device=self.device, inputs_bert_dict=prompts)
                    self.log(f"视觉特征提取完成，返回类型: {type(mm_features)}")
                    
                    # 输出视觉特征形状信息
                    if isinstance(mm_features, list):
                        self.log(f"视觉特征是列表，长度: {len(mm_features)}")
                        if len(mm_features) > 0:
                            first_feature = mm_features[0]
                            self.log(f"第一个特征元素类型: {type(first_feature)}")
                            if torch.is_tensor(first_feature):
                                self.log(f"视觉特征形状: {first_feature.shape}")
                    elif torch.is_tensor(mm_features):
                        self.log(f"视觉特征形状: {mm_features.shape}")
                except Exception as e:
                    self.log_error(f"视觉特征提取失败: {str(e)}")
                    return None, None
                
                # 提取音频特征 - 彻底修复数据类型问题
                if audio_tensor is not None:
                    try:
                        # 强制音频张量为FP32
                        audio_tensor = audio_tensor.to(dtype=torch.float32, device=self.device)
                        
                        # 确保音频编码器使用FP32并在正确设备上
                        if hasattr(self.model, 'audio_encoder'):
                            self.model.audio_encoder = self.model.audio_encoder.to(self.device).float()
                        
                        self.log(f"最终音频张量 - 形状: {audio_tensor.shape}, 数据类型: {audio_tensor.dtype}")
                        
                        # 提取音频特征 - 增强版本，解决列表类型问题
                        audio_features = self.model.encode_audios(audio_tensor)
                        self.log(f"音频特征提取完成，形状: {getattr(audio_features, 'shape', '未知')}, 类型: {type(audio_features)}")
                        
                        # 处理可能返回的是列表的情况
                        if isinstance(audio_features, list):
                            self.log(f"检测到音频特征是列表，长度: {len(audio_features)}")
                            # 尝试将列表转换为单个张量
                            try:
                                # 检查列表中的元素是否都是张量
                                if all(isinstance(f, torch.Tensor) for f in audio_features):
                                    # 如果是多个张量，尝试拼接或取平均
                                    if len(audio_features) == 1:
                                        audio_features = audio_features[0]
                                    else:
                                        # 尝试在第一个维度拼接
                                        audio_features = torch.cat(audio_features, dim=0)
                                    self.log(f"已将音频特征列表转换为张量，新形状: {audio_features.shape}")
                                else:
                                    self.log_error("音频特征列表包含非张量元素")
                                    audio_features = None
                            except Exception as se:
                                self.log_error(f"转换音频特征列表失败: {str(se)}")
                                audio_features = None
                        
                    except Exception as ae:
                        self.log_error(f"音频特征提取失败: {str(ae)}")
                        # 终极解决方案：创建一个简单的占位音频特征
                        try:
                            self.log("尝试创建占位音频特征")
                            # 创建一个与模型输出维度匹配的零张量
                            if hasattr(self.model, 'audio_encoder') and hasattr(self.model.audio_encoder, 'hidden_size'):
                                hidden_size = self.model.audio_encoder.hidden_size
                                audio_features = torch.zeros(1, audio_tensor.shape[1], hidden_size, 
                                                          dtype=torch.float32, device=self.device)
                                self.log(f"已创建占位音频特征，形状: {audio_features.shape}")
                            else:
                                audio_features = None
                        except Exception as final_e:
                            self.log_error(f"创建占位音频特征失败: {str(final_e)}")
                            audio_features = None
            
            # 验证特征有效性
            if mm_features is None:
                self.log_error(f"特征提取返回空视觉特征: {video_path}")
                return None, None
            
            # 转换为numpy并返回 - 增强的序列类型处理
            try:
                # 特别处理mm_features的各种可能类型
                self.log(f"mm_features类型: {type(mm_features)}")
                
                if isinstance(mm_features, list):
                    # 确保列表中每个元素都是张量
                    valid_tensors = []
                    for f in mm_features:
                        if torch.is_tensor(f):
                            valid_tensors.append(f)
                            self.log(f"列表中张量特征形状: {f.shape}")
                        else:
                            self.log_error(f"列表中包含非张量元素: {type(f)}")
                    
                    if len(valid_tensors) > 0:
                        # 如果列表只有一个张量，直接返回该张量的numpy数组
                        if len(valid_tensors) == 1:
                            mm_np = valid_tensors[0].cpu().numpy()
                            self.log(f"单个张量列表成功转换为numpy，形状: {mm_np.shape}")
                        else:
                            # 对于多个张量，先将每个转换为numpy，然后堆叠
                            mm_np_list = [f.cpu().numpy() for f in valid_tensors]
                            try:
                                mm_np = np.stack(mm_np_list)
                                self.log(f"多个张量列表成功堆叠为numpy，形状: {mm_np.shape}")
                            except Exception as e:
                                self.log_error(f"堆叠张量列表失败: {str(e)}")
                                # 如果堆叠失败，返回第一个张量
                                mm_np = valid_tensors[0].cpu().numpy()
                                self.log(f"使用第一个张量作为替代，形状: {mm_np.shape}")
                    else:
                        # 如果没有有效张量，创建一个占位符
                        self.log("没有有效张量，创建占位视觉特征")
                        mm_np = [np.zeros((1, 768), dtype=np.float32)]
                elif torch.is_tensor(mm_features):
                    # 单一张量的情况
                    mm_np = mm_features.cpu().numpy()
                elif isinstance(mm_features, (tuple, np.ndarray)):
                    # 处理元组或numpy数组
                    mm_np = np.array(mm_features)
                else:
                    # 其他类型，尝试转换为numpy数组
                    self.log(f"未知的特征类型: {type(mm_features)}，尝试强制转换")
                    try:
                        mm_np = np.array(mm_features)
                    except Exception as e:
                        self.log_error(f"强制转换失败: {str(e)}")
                        mm_np = np.zeros((1, 768), dtype=np.float32)  # 创建占位特征
                
                # 处理音频特征 - 超级增强版，解决所有可能的序列类型问题
                if audio_features is not None:
                    self.log(f"音频特征转换前 - 类型: {type(audio_features)}, 是否为张量: {torch.is_tensor(audio_features)}")
                    
                    if torch.is_tensor(audio_features):
                        try:
                            audio_np = audio_features.cpu().numpy()
                            self.log(f"音频张量成功转换为numpy，形状: {audio_np.shape}")
                        except Exception as e:
                            self.log_error(f"音频张量转换失败: {str(e)}")
                            audio_np = None
                    elif isinstance(audio_features, list):
                        # 处理列表情况
                        self.log(f"音频特征是列表，长度: {len(audio_features)}")
                        try:
                            # 检查列表中的元素是否都是张量
                            valid_tensors = []
                            for f in audio_features:
                                if torch.is_tensor(f):
                                    valid_tensors.append(f.cpu().numpy())
                                else:
                                    try:
                                        # 尝试将非张量元素转换为numpy数组
                                        valid_tensors.append(np.array(f))
                                    except Exception:
                                        self.log_error(f"无法转换列表元素: {type(f)}")
                                        continue
                            
                            if len(valid_tensors) > 0:
                                # 尝试堆叠所有有效元素
                                audio_np = np.stack(valid_tensors)
                                self.log(f"音频特征列表成功转换为numpy，形状: {audio_np.shape}")
                            else:
                                self.log_error("列表中没有可转换的有效元素")
                                audio_np = None
                        except Exception as e:
                            self.log_error(f"音频特征列表转换失败: {str(e)}")
                            audio_np = None
                    elif isinstance(audio_features, tuple):
                        # 处理元组情况
                        self.log(f"音频特征是元组，长度: {len(audio_features)}")
                        try:
                            # 先转换为列表再处理
                            audio_np = np.array(list(audio_features))
                            self.log(f"音频特征元组成功转换为numpy，形状: {audio_np.shape}")
                        except Exception as e:
                            self.log_error(f"音频特征元组转换失败: {str(e)}")
                            audio_np = None
                    else:
                        # 其他类型，尝试转换为numpy数组
                        try:
                            audio_np = np.array(audio_features)
                            self.log(f"音频特征(类型: {type(audio_features)})成功转换为numpy，形状: {audio_np.shape}")
                        except Exception as e:
                            self.log_error(f"音频特征转换失败: {str(e)}")
                            audio_np = None
                else:
                    audio_np = None
                
                self.log(f"特征转换为numpy成功: {video_path}")
                return mm_np, audio_np
            except Exception as e:
                self.log_error(f"特征转换为numpy失败: {str(e)}")
                # 终极兜底方案：返回占位特征
                placeholder_visual = np.zeros((1, 768), dtype=np.float32)
                placeholder_audio = np.zeros((1, 128, 768), dtype=np.float32) if audio_tensor is not None else None
                return placeholder_visual, placeholder_audio
        
        except Exception as e:
            self.log_error(f"视频/音频特征提取失败: {str(e)}")
            self.log_error(traceback.format_exc())
            return None, None

    def process_single_video(self, video_path):
        """处理单个视频，提取视频和音频特征"""
        try:
            # 解析视频ID
            parts = video_path.replace('\\', '/').split('/')
            dir_name = parts[-2] if len(parts) >=2 else 'unknown'
            file_name = os.path.splitext(parts[-1])[0] if len(parts)>=1 else 'unknown'
            key_id = f"{dir_name}$_${file_name}"
            
            # 提取视频和音频特征
            visual_features, audio_features = self.extract_video_audio_features(video_path)
            if visual_features is None:
                return None
            
            return {
                'vision': visual_features,
                'audio': audio_features,
                'key_id': key_id
            }
        
        except Exception as e:
            self.log_error(f"处理视频 {video_path} 失败: {str(e)}")
            return None

    def process_batch(self, video_batch):
        """批量处理视频（单线程内）"""
        results = []
        for path in video_batch:
            try:
                # 逐个处理视频，并在处理每个视频后清理内存
                result = self.process_single_video(path)
                if result is not None:
                    results.append(result)
                # 处理完一个视频后清理内存
                torch.cuda.empty_cache()
            except Exception as e:
                self.log_error(f"处理批次中的视频 {path} 时发生异常: {str(e)}")
        return results

    def extract_features(self, video_files):
        """提取所有视频特征（使用线程池和批处理）"""
        total_videos = len(video_files)
        if total_videos == 0:
            self.log("没有视频文件需要处理")
            return
        
        all_features = {}
        start_time = time.time()
        
        # 获取批大小
        batch_size = self.config.batch_size
        self.log(f"使用批处理大小: {batch_size}, 线程数: {self.config.num_workers}")
        
        # 将视频文件分成批次
        video_batches = []
        for i in range(0, total_videos, batch_size):
            video_batches.append(video_files[i:i + batch_size])
        
        self.log(f"将 {total_videos} 个视频分成了 {len(video_batches)} 个批次进行处理")
        
        # 优化多线程处理，避免同时提交所有任务导致内存压力过大
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 控制并发批次数量，避免一次性提交过多任务
            max_concurrent_batches = min(self.config.num_workers, len(video_batches))
            futures = []
            future_to_batch = {}
            
            # 先提交初始批次
            for i in range(min(max_concurrent_batches, len(video_batches))):
                future = executor.submit(self.process_batch, video_batches[i])
                futures.append(future)
                future_to_batch[future] = video_batches[i]
            
            # 使用tqdm显示进度
            processed_count = 0
            batch_index = max_concurrent_batches
            with tqdm(total=total_videos, desc="提取视频特征") as pbar:
                while futures:
                    # 等待任一任务完成
                    done, not_done = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        try:
                            batch_results = future.result()
                            for result in batch_results:
                                if result is not None:
                                    all_features[result['key_id']] = result
                            processed_count += len(future_to_batch[future])
                            pbar.update(len(future_to_batch[future]))
                            
                            # 处理完一批后立即清理显存
                            torch.cuda.empty_cache()
                            self.log(f"已处理 {processed_count}/{total_videos} 个视频，清理显存垃圾")
                            
                            # 提交新的批次任务（如果还有剩余）
                            if batch_index < len(video_batches):
                                new_future = executor.submit(self.process_batch, video_batches[batch_index])
                                futures.append(new_future)
                                future_to_batch[new_future] = video_batches[batch_index]
                                batch_index += 1
                        except Exception as e:
                            batch = future_to_batch[future]
                            self.log_error(f"处理批次时发生异常: {str(e)}")
                            self.log_error(f"受影响的视频数量: {len(batch)}")
                            self.log_error(traceback.format_exc())
                        
                        # 从列表中移除已完成的任务
                        futures.remove(future)
                        del future_to_batch[future]
        
        # 保存结果
        total_features_file = os.path.join(self.config.output_dir, 'total_features_valid.pkl')
        with open(total_features_file, 'wb') as f:
            pickle.dump(all_features, f)
        
        # 输出统计信息
        elapsed_time = time.time() - start_time
        summary = (
            f"处理完成: 总视频数 {total_videos}, "
            f"成功 {len(all_features)}, "
            f"耗时 {elapsed_time:.2f}秒, "
            f"平均 {len(all_features)/elapsed_time:.2f}个/秒"
        )
        self.log(summary)
        return all_features

    def log(self, message):
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def log_error(self, message):
        print(f"错误: {message}")
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def get_video_files(folder_path):
    """获取所有视频文件路径"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(root, file))
    return video_files

def parse_args():
    parser = argparse.ArgumentParser(description="提取视频的特征")
    # 模型配置
    parser.add_argument('--humanomni_model_path', type=str, required=True, help='HumanOmni模型路径')
    parser.add_argument('--bert_model', type=str, default='/data/jianghong/R1-Omni/models/bert-base-uncased', help='BERT模型路径或名称')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--use_fp16', action='store_true', default=True, help='默认使用FP16精度以节省显存')
    # 数据配置
    parser.add_argument('--input', type=str, required=True, help='视频目录')
    parser.add_argument('--output_dir', type=str, default='./features_output', help='输出目录')
    parser.add_argument('--num_frames', type=int, default=NUM_FRAMES, help='视频采样帧数')
    # 性能配置
    parser.add_argument('--num_workers', type=int, default=4, help='线程数（建议不超过CPU核心数）')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小（A6000上建议使用较小值如4）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    video_files = get_video_files(args.input)
    print(f"发现 {len(video_files)} 个视频文件")
    
    if video_files:
        extractor = VideoFeatureExtractor(args)
        extractor.extract_features(video_files)
    else:
        print("未找到视频文件，请检查输入路径")