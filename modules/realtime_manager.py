"""
实时处理管理器
负责协调视频队列、处理器和情感分析
"""
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from .video_queue import VideoQueueManager
from .video_processor import VideoProcessor
from .qwen_service import get_qwen_service

logger = logging.getLogger(__name__)

class RealtimeManager:
    """实时处理管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实时处理管理器
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化组件
        self.queue_manager = VideoQueueManager()
        self.video_processor = VideoProcessor(
            output_dir=config.get('video_folder', './videos'),
            frame_rate=config.get('frame_rate', 10)
        )
        
        # 获取Qwen模型服务
        self.qwen_service = get_qwen_service()
        
        # 初始化本地模型
        model_path = config.get('qwen_model_path')
        if model_path:
            try:
                success = self.qwen_service.load_model(
                    model_path=model_path,
                    max_gpus=config.get('max_gpus', 4)
                )
                if success:
                    logger.info("本地Qwen3-Omni模型服务初始化成功")
                else:
                    logger.error("本地Qwen3-Omni模型服务初始化失败")
            except Exception as e:
                logger.error(f"本地模型服务初始化失败: {e}")
        else:
            logger.warning("未配置本地模型路径")
        
        # 实时处理控制
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.processing_flags: Dict[str, bool] = {}
        self.emotion_callbacks: Dict[str, Callable] = {}
        
        # 启动视频处理器
        self.video_processor.start_worker()
    
    def start_realtime_processing(self, session_id: str, emotion_callback: Callable = None):
        """
        开始实时处理
        Args:
            session_id: 会话ID
            emotion_callback: 情感分析结果回调函数
        """
        if session_id in self.processing_flags and self.processing_flags[session_id]:
            logger.warning(f"会话 {session_id} 的实时处理已在运行")
            return
        
        self.processing_flags[session_id] = True
        if emotion_callback:
            self.emotion_callbacks[session_id] = emotion_callback
        
        # 启动处理线程
        thread = threading.Thread(
            target=self._realtime_processing_loop,
            args=(session_id,),
            daemon=True
        )
        self.processing_threads[session_id] = thread
        thread.start()
        
        logger.info(f"开始实时处理: session_id={session_id}")
    
    def stop_realtime_processing(self, session_id: str):
        """停止实时处理"""
        if session_id in self.processing_flags:
            self.processing_flags[session_id] = False
        
        # 等待线程结束
        if session_id in self.processing_threads:
            thread = self.processing_threads[session_id]
            if thread.is_alive():
                thread.join(timeout=5)
            del self.processing_threads[session_id]
        
        # 清理回调
        if session_id in self.emotion_callbacks:
            del self.emotion_callbacks[session_id]
        
        logger.info(f"停止实时处理: session_id={session_id}")
    
    def _realtime_processing_loop(self, session_id: str):
        """实时处理主循环"""
        logger.info(f"实时处理循环开始: session_id={session_id}")
        
        # 等待5秒让队列积累数据
        time.sleep(5)
        
        while self.processing_flags.get(session_id, False):
            try:
                # 获取队列
                video_queue = self.queue_manager.get_queue(session_id)
                
                # 获取最近5秒的数据
                video_frames, audio_chunks = video_queue.get_recent_data(duration=5.0)
                
                if video_frames:
                    # 添加视频合成任务
                    self.video_processor.add_composition_task(
                        video_frames=video_frames,
                        audio_chunks=audio_chunks,
                        session_id=session_id,
                        callback=lambda result: self._on_video_composed(session_id, result)
                    )
                    
                    logger.debug(f"添加视频合成任务: session_id={session_id}, frames={len(video_frames)}")
                
                # 每秒处理一次
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"实时处理循环错误: {e}")
                time.sleep(1)
        
        logger.info(f"实时处理循环结束: session_id={session_id}")
    
    def _on_video_composed(self, session_id: str, result: Dict[str, Any]):
        """视频合成完成回调"""
        if not result.get('success'):
            logger.error(f"视频合成失败: {result.get('error')}")
            return
        
        video_path = result.get('video_path')
        if not video_path:
            return
        
        # 进行情感分析
        if self.qwen_service.is_model_ready():
            self._analyze_emotion_async(session_id, video_path)
        else:
            logger.warning("Qwen模型服务未准备就绪，跳过情感分析")
    
    def _analyze_emotion_async(self, session_id: str, video_path: str):
        """异步情感分析"""
        def analyze():
            try:
                result = self.qwen_service.analyze_video_emotion(video_path)
                
                # 调用回调函数
                callback = self.emotion_callbacks.get(session_id)
                if callback:
                    callback({
                        'session_id': session_id,
                        'video_path': video_path,
                        'emotion_result': result,
                        'timestamp': time.time()
                    })
                
            except Exception as e:
                logger.error(f"情感分析失败: {e}")
        
        # 在新线程中执行情感分析
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def add_video_frame(self, session_id: str, frame_data):
        """添加视频帧"""
        video_queue = self.queue_manager.get_queue(session_id)
        video_queue.add_video_frame(frame_data)
    
    def add_audio_chunk(self, session_id: str, audio_data):
        """添加音频块"""
        video_queue = self.queue_manager.get_queue(session_id)
        video_queue.add_audio_chunk(audio_data)
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        video_queue = self.queue_manager.get_queue(session_id)
        queue_info = video_queue.get_queue_info()
        
        return {
            'session_id': session_id,
            'is_processing': self.processing_flags.get(session_id, False),
            'queue_info': queue_info,
            'has_emotion_analyzer': self.emotion_analyzer is not None
        }
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """获取所有会话状态"""
        all_queues = self.queue_manager.get_all_queues_info()
        
        status = {
            'total_sessions': len(all_queues),
            'video_processor': self.video_processor.get_status(),
            'sessions': {}
        }
        
        for session_id in all_queues:
            status['sessions'][session_id] = self.get_session_status(session_id)
        
        return status
    
    def cleanup_session(self, session_id: str):
        """清理会话"""
        # 停止实时处理
        self.stop_realtime_processing(session_id)
        
        # 移除队列
        self.queue_manager.remove_queue(session_id)
        
        logger.info(f"会话清理完成: session_id={session_id}")
    
    def shutdown(self):
        """关闭管理器"""
        # 停止所有实时处理
        for session_id in list(self.processing_flags.keys()):
            self.stop_realtime_processing(session_id)
        
        # 停止视频处理器
        self.video_processor.stop_worker()
        
        logger.info("实时处理管理器已关闭")