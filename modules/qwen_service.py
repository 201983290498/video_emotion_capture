# -*- coding: utf-8 -*-
"""
Qwen3-Omni模型服务
将模型加载和使用分离，提供统一的服务接口
"""
import os
import logging
import threading
import time
from typing import Dict, Any, Optional
from .qwen_omni import LocalQwenOmni

logger = logging.getLogger(__name__)

class QwenModelService:
    """Qwen3-Omni模型服务类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(QwenModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化服务"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.model = None
        self.model_path = None
        self.is_loading = False
        self.is_loaded = False
        self.load_error = None
        self._load_lock = threading.Lock()
        
        logger.info("Qwen模型服务初始化完成")
    
    def load_model(self, model_path: str, device: str = "auto", max_gpus: int = 4) -> bool:
        """
        加载并缓存Qwen3-Omni模型实例
        Args:
            model_path: 模型路径
            device: 设备类型
            max_gpus: 最大GPU数量
        Returns:
            bool: 加载是否成功
        """
        with self._load_lock:
            if self.is_loaded and self.model_path == model_path:
                logger.info("模型已加载，跳过重复加载")
                return True
            
            if self.is_loading:
                logger.warning("模型正在加载中，请等待...")
                return False
            
            self.is_loading = True
            self.load_error = None
            
            try:
                logger.info(f"开始加载Qwen3-Omni模型: {model_path}")
                
                # 如果已有模型，先清理
                if self.model is not None:
                    self._cleanup_model()
                
                # 加载新模型
                self.model = LocalQwenOmni(
                    model_path=model_path,
                    device=device,
                    max_gpus=max_gpus
                )
                
                # 检查模型是否加载成功
                if self.model.model is None:
                    raise Exception("Qwen3-Omni模型加载失败，请检查模型路径和transformers版本")
                
                # 检查processor是否加载成功
                if self.model.processor is None:
                    raise Exception("Qwen3-Omni processor加载失败")
                
                # 检查tokenizer是否可用
                if self.model.tokenizer is None:
                    logger.warning("Tokenizer未正确加载，可能影响功能")
                
                self.model_path = model_path
                self.is_loaded = True
                logger.info("Qwen3-Omni模型加载成功")
                return True
                
            except Exception as e:
                self.load_error = str(e)
                logger.error(f"模型加载失败: {e}")
                self.model = None
                self.is_loaded = False
                return False
            finally:
                self.is_loading = False
    
    def analyze_video_emotion(self, video_path: str, audio_path: str = None, prompt: str = None) -> Dict[str, Any]:
        """
        分析视频情感
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径（可选）
            prompt: 自定义提示词（可选）
        Returns:
            Dict: 情感分析结果
        """
        if not self.is_model_ready():
            return {
                'success': False,
                'error': '模型未准备就绪',
                'details': self.get_status()
            }
        
        try:
            logger.info(f"开始分析视频情感: {video_path}")
            result = self.model.analyze_video_emotion(   # 为什么是调用自己？
                video_path=video_path,
                audio_path=audio_path,
                prompt=prompt
            )
            
            # 添加服务信息
            result['service_info'] = {
                'model_path': self.model_path,
                'timestamp': time.time(),
                'video_path': video_path
            }
            
            logger.info("视频情感分析完成")
            return result
            
        except Exception as e:
            logger.error(f"视频情感分析失败: {e}")
            return {
                'success': False,
                'error': f'分析失败: {str(e)}',
                'timestamp': time.time(),
                'video_path': video_path
            }
    
    def analyze_audio_emotion(self, audio_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        分析音频情感
        Args:
            audio_path: 音频文件路径
            prompt: 自定义提示词（可选）
        Returns:
            Dict: 情感分析结果
        """
        if not self.is_model_ready():
            return {
                'success': False,
                'error': '模型未准备就绪',
                'details': self.get_status()
            }
        
        try:
            logger.info(f"开始分析音频情感: {audio_path}")
            result = self.model.analyze_audio_emotion(       # 为什么是调用自己？
                audio_path=audio_path,
                prompt=prompt
            )
            
            # 添加服务信息
            result['service_info'] = {
                'model_path': self.model_path,
                'timestamp': time.time(),
                'audio_path': audio_path
            }
            
            logger.info("音频情感分析完成")
            return result
            
        except Exception as e:
            logger.error(f"音频情感分析失败: {e}")
            return {
                'success': False,
                'error': f'分析失败: {str(e)}',
                'timestamp': time.time(),
                'audio_path': audio_path
            }
    
    def is_model_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return (self.is_loaded and 
                self.model is not None and 
                self.model.model is not None and 
                self.model.processor is not None and
                not self.is_loading)
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'is_loaded': self.is_loaded,
            'is_loading': self.is_loading,
            'model_path': self.model_path,
            'load_error': self.load_error,
            'model_ready': self.is_model_ready(),
            'model_info': self.model.get_model_info() if self.model else None
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_model_ready():
            return {'error': '模型未准备就绪'}
        
        return self.model.get_model_info()
    
    def _cleanup_model(self):
        """清理模型资源"""
        if self.model is not None:
            try:
                # 清理GPU内存
                if hasattr(self.model, 'model') and self.model.model is not None:
                    del self.model.model
                if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                    del self.model.tokenizer
                if hasattr(self.model, 'processor') and self.model.processor is not None:
                    del self.model.processor
                
                # 删除模型实例
                del self.model
                
                # 清理CUDA缓存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                logger.info("模型资源清理完成")
            except Exception as e:
                logger.warning(f"模型资源清理失败: {e}")
    
    def unload_model(self):
        """卸载模型"""
        with self._load_lock:
            if self.model is not None:
                self._cleanup_model()
                self.model = None
            
            self.is_loaded = False
            self.model_path = None
            self.load_error = None
            logger.info("模型已卸载")
    



# 全局服务实例
qwen_service = QwenModelService()

def get_qwen_service() -> QwenModelService:
    """获取Qwen模型服务实例"""
    return qwen_service