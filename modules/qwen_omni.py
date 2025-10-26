"""
本地Qwen3-Omni模型推理模块
支持多模态情感识别
"""
import os
import cv2
import logging
import time
import torch
from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer
from PIL import Image

# 尝试导入Qwen3-Omni相关模块
try:
    from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
    QWEN3_OMNI_AVAILABLE = True
except ImportError:
    QWEN3_OMNI_AVAILABLE = False
    print("警告: Qwen3-Omni模块不可用，将使用通用模型")

# 尝试导入Qwen2.5-Omni相关模块
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    QWEN25_OMNI_AVAILABLE = True
except ImportError:
    QWEN25_OMNI_AVAILABLE = False
    print("警告: Qwen2.5-Omni模块不可用")

# 尝试导入vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM不可用，将仅使用HuggingFace模式")

logger = logging.getLogger(__name__)

class LocalQwenOmni:
    """本地Qwen3-Omni模型推理器   根据模型路径自动判定类型Qwen3-Omni或Qwen2.5-Omni"""
    
    def __init__(self, model_path: str, device: str = "auto", max_gpus: int = 4, use_vllm: bool = False):
        """
        初始化本地Qwen Omni模型
        Args:
            model_path: 模型路径
            device: 设备类型 ("auto", "cpu", "cuda")
            max_gpus: 最大GPU数量
            use_vllm: 是否使用vLLM加载模型
        """
        self.model_path = model_path
        self.max_gpus = max_gpus
        self.device = device if device == 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = self._detect_model_type(model_path)
        self.use_vllm = use_vllm and VLLM_AVAILABLE 
        # 加载模型
        self._load_model()
    
    def _detect_model_type(self, model_path: StopIteration) -> str:
        path = model_path.lower()
        if "qwen3" in path or "qwen-3" in path:
            return "qwen3-omni"
        if "qwen2.5" in path or "qwen-2.5" in path:
            return "qwen2.5-omni"
        
        return "qwen3-omni" if QWEN3_OMNI_AVAILABLE else "qwen2.5-omni"
    
    def _load_model(self):
        """
        加载模型的主方法
        """
        # 根据模型类型选择对应的模型类
        if self.model_type == "qwen3-omni" and QWEN3_OMNI_AVAILABLE:
            model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
            processor_cls = Qwen3OmniMoeProcessor
        elif self.model_type == "qwen2.5-omni" and QWEN25_OMNI_AVAILABLE:
            model_cls = Qwen2_5OmniForConditionalGeneration
            processor_cls = Qwen2_5OmniProcessor
        else:
            raise Exception(f"不支持的模型类型: {self.model_type}")
        # 加载模型
        self.processor = processor_cls.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        if self.use_vllm:
            vllm_kwargs = {
                "model": self.model_path,
                "trust_remote_code": True,
                "max_model_len": 8192,  # 可根据需要调整
                "tensor_parallel_size": min(self.max_gpus, torch.cuda.device_count())
            }
            self.model = LLM(**vllm_kwargs)
        else:
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "device_map": "auto"
            }
            self.model = model_cls.from_pretrained(self.model_path, **model_kwargs)
            
        logger.info(f"{self.model_type}模型初始化完成")
        
    
    def analyze_video_emotion(self, video_path: str, audio_path: str = None, prompt: str = None) -> Dict[str, Any]:
        """分析视频情感（支持音频+视频多模态输入）"""
        try:
            # 前置检查
            if not (self.model and self.tokenizer and os.path.exists(video_path)):
                return {'success': False, 'error': '模型未加载或视频不存在', 'timestamp': time.time()}
            frames = self._extract_key_frames(video_path, max_frames=8)
            if not frames:
                return {'success': False, 'error': '无法提取视频帧', 'timestamp': time.time()}

            # 音频处理
            audio_data = audio_path if audio_path and os.path.exists(audio_path) else self._extract_audio_from_video(video_path)
            prompt = prompt or "请你仔细分析这个视频表情的情感，用1~3个单词概括。"

            # 多模态推理
            try:
                conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image", "image": f} for f in frames]}]
                inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_tensors="pt")
                if self.device != "cpu":
                    inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
                                                  pad_token_id=self.processor.tokenizer.eos_token_id)
                response = self.processor.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"多模态处理失败，降级文本推理: {e}")
                text_input = f"{prompt}\n\n[仅文本分析，实际分析了{len(frames)}帧]"
                inputs = self.processor.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                if self.device != "cpu":
                    inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
                                                  pad_token_id=self.processor.tokenizer.eos_token_id)
                response = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            return {"success": True, "result": self._parse_emotion_response(response), "raw_response": response,
                    "has_audio": audio_data is not None, "frames_analyzed": len(frames), "timestamp": time.time()}

        except Exception as e:
            logger.error(f"视频情感分析失败: {e}")
            return {"success": False, "error": str(e), "timestamp": time.time()}
    
    def _extract_key_frames(self, video_path: str, max_frames: int = 8) -> List[Image.Image]:
        """提取视频关键帧"""
        frames = []
        try:
            # OpenCV 的 VideoCapture 不支持上下文管理协议，不能用 with
            cap = cv2.VideoCapture(video_path)
            try:
                if not cap.isOpened():
                    logger.error(f"无法打开视频文件: {video_path}")
                    return frames
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                step = max(1, total // max_frames)
                count = 0
                for idx in range(0, total, step):
                    if count >= max_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                        count += 1
                logger.info(f"成功提取 {len(frames)} 个关键帧")
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"提取视频帧失败: {e}")
        return frames
    
    def _parse_emotion_response(self, response: str) -> Dict[str, Any]:
        """解析情感分析响应"""
        emotion_map = {
            '快乐': 'happy', '高兴': 'happy', '开心': 'happy',
            '悲伤': 'sad', '难过': 'sad', '伤心': 'sad',
            '愤怒': 'angry', '生气': 'angry',
            '惊讶': 'surprised', '震惊': 'surprised',
            '恐惧': 'fear', '害怕': 'fear',
            '厌恶': 'disgust', '恶心': 'disgust',
            '中性': 'neutral', '平静': 'neutral'
        }
        emotion = next((e for k, e in emotion_map.items() if k in response), 'neutral')
        return {'emotion': emotion, 'confidence': 0.8, 'description': response.strip(), 'raw_response': response}
    
    
    def _extract_audio_from_video(self, video_path: str) -> str:
        """从视频中提取音频"""
        import subprocess, tempfile
        try:
            tmp = tempfile.mktemp(suffix='.wav')
            subprocess.run(
                ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                 '-ar', '16000', '-ac', '1', '-y', tmp],
                check=True, capture_output=True, text=True
            )
            return tmp
        except Exception as e:
            logger.error(f'提取音频失败: {e}')
            return None
    
    def analyze_audio_emotion(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """分析音频情感"""
        try:
            if not (self.model and self.tokenizer and os.path.exists(audio_path)):
                return {'success': False, 'error': '模型未加载或音频不存在', 'timestamp': time.time()}

            prompt = prompt or "请分析这段音频中的主要情感，用1~3个情感词语概括。"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": audio_path}]}]

            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
            if self.device != "cpu":
                inputs = inputs.to(self.device)

            outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True,
                                            pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            return {"success": True, "result": self._parse_emotion_response(response), "raw_response": response,
                    "timestamp": time.time()}

        except Exception as e:
            logger.error(f"音频情感分析失败: {e}")
            return {"success": False, "error": str(e), "timestamp": time.time()}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "max_gpus": self.max_gpus,
            "use_vllm": self.use_vllm,
            "model_loaded": (hasattr(self, 'model') and self.model is not None) or 
                           (hasattr(self, 'vllm_model') and self.vllm_model is not None),
            "processor_loaded": hasattr(self, 'processor') and self.processor is not None,
            "tokenizer_loaded": hasattr(self, 'tokenizer') and self.tokenizer is not None,
            "qwen3_available": QWEN3_OMNI_AVAILABLE,
            "qwen25_available": QWEN25_OMNI_AVAILABLE,
            "vllm_available": VLLM_AVAILABLE
        }