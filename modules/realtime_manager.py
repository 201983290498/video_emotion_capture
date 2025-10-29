"""
实时处理管理器
负责协调视频队列、处理器和情感分析
"""
import time
import threading
import logging
import statistics
import json
import os
from collections import deque
from typing import Dict, Any, Optional, Callable
from .video_queue import VideoQueueManager
from .video_processor import VideoProcessor
from .lddu_service import get_lddu_service

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
        
        # 获取 LDDU 服务（不在此处加载模型，避免重复初始化）
        self.lddu_service = get_lddu_service()
        self.lddu_ready = getattr(self.lddu_service, 'is_loaded', False)
        
        # 实时处理控制
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.processing_flags: Dict[str, bool] = {}
        self.emotion_callbacks: Dict[str, Callable] = {}
        
        # 推理时间统计 (保留最近100次记录，仅 LDDU)
        self.inference_times = {
            'lddu': deque(maxlen=100)
        }
        self.inference_stats = {
            'lddu': {'count': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0}
        }
        self.stats_lock = threading.Lock()
        # 推理日志文件路径与写入锁
        self.inference_log_file = config.get('inference_log_file')
        self.log_lock = threading.Lock()
        
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
    
    def _update_inference_stats(self, model_type: str, inference_time: float):
        """更新推理时间统计"""
        with self.stats_lock:
            # 添加到时间队列
            self.inference_times[model_type].append(inference_time)
            
            # 更新统计信息
            stats = self.inference_stats[model_type]
            stats['count'] += 1
            stats['total_time'] += inference_time
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['min_time'] = min(stats['min_time'], inference_time)
            stats['max_time'] = max(stats['max_time'], inference_time)
        
        # 追加写入NDJSON日志（不阻塞统计锁）
        self._append_inference_log(model_type, inference_time, stats)
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """获取推理时间统计"""
        with self.stats_lock:
            stats = {}
            for model_type in ['lddu']:
                recent_times = list(self.inference_times[model_type])
                model_stats = self.inference_stats[model_type].copy()
                
                if recent_times:
                    model_stats['recent_avg'] = statistics.mean(recent_times)
                    model_stats['recent_std'] = statistics.stdev(recent_times) if len(recent_times) > 1 else 0
                    model_stats['recent_count'] = len(recent_times)
                    model_stats['fps'] = 1.0 / model_stats['recent_avg'] if model_stats['recent_avg'] > 0 else 0
                else:
                    model_stats['recent_avg'] = 0
                    model_stats['recent_std'] = 0
                    model_stats['recent_count'] = 0
                    model_stats['fps'] = 0
                
                stats[model_type] = model_stats
            
            return stats
    
    def stop_realtime_processing(self, session_id: str):
        """停止实时处理"""
        if session_id in self.processing_flags:
            self.processing_flags[session_id] = False
        
        # 等待线程结束
        thread = self.processing_threads.get(session_id)
        if thread:
            if thread.is_alive():
                thread.join(timeout=5)
            self.processing_threads.pop(session_id, None)
        
        # 清理回调
        self.emotion_callbacks.pop(session_id, None)
        
        logger.info(f"停止实时处理: session_id={session_id}")
    
    def _realtime_processing_loop(self, session_id: str):
        """实时处理主循环 - 每5秒合成一次视频进行情感分析"""
        logger.info(f"实时处理循环开始: session_id={session_id} (每5秒合成一次)")
        
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
                    
                    # 减少日志输出，只在debug模式下显示详细信息
                    logger.debug(f"合成5秒视频片段: session_id={session_id}, frames={len(video_frames)}")
                
                # 每5秒处理一次，平衡实时性和性能
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"实时处理循环错误: {e}")
                time.sleep(5)  # 错误时也等待5秒
        
        logger.info(f"实时处理循环结束: session_id={session_id}")
    
    def _on_video_composed(self, session_id: str, result: Dict[str, Any]):
        """视频合成完成回调"""
        if not result.get('success'):
            logger.error(f"视频合成失败: {result.get('error')}")
            return
        
        video_path = result.get('video_path')
        if not video_path:
            return
        
        # 进行情感分析，仅使用 LDDU
        if self.lddu_service and self.lddu_service.is_loaded:
            self._analyze_emotion_async_lddu(session_id, video_path)
        else:
            logger.warning("LDDU 未就绪，跳过情感分析")
    
    # Qwen 情感识别路径已移除
    
    def _analyze_emotion_async_lddu(self, session_id: str, video_path: str):
        """异步情感分析(LDDU)"""
        def analyze():
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 使用默认阈值进行分析，返回完整结果结构
                result = self.lddu_service.analyze_video_emotion(video_path)
                
                # 记录结束时间并更新统计
                end_time = time.time()
                inference_time = end_time - start_time
                self._update_inference_stats('lddu', inference_time)
                
                # 输出耗时同时包含视频编号/文件名
                video_basename = os.path.basename(video_path) if video_path else "unknown"
                video_id = None
                try:
                    name_no_ext = os.path.splitext(video_basename)[0]
                    parts = name_no_ext.split('_')
                    # 约定文件名: video_<session_id>_<timestamp>.mp4
                    if len(parts) >= 3:
                        video_id = parts[-1]
                except Exception:
                    pass
                if video_id:
                    logger.info(f"LDDU情感分析耗时: {inference_time:.3f}s | video_id={video_id} | file={video_basename}")
                else:
                    logger.info(f"LDDU情感分析耗时: {inference_time:.3f}s | file={video_basename}")
                
                callback = self.emotion_callbacks.get(session_id)
                if callback:
                    # 提取情感结果，仅返回标签
                    emotion_text = "未检测到情感"
                    if result and result.get('success') and result.get('emotion'):
                        emotion_text = result['emotion']
                    
                    callback({
                        'emotion': emotion_text,
                        'timestamp': time.time(),
                        'video_name': "实时分析",
                        'inference_time': inference_time,
                        'model_type': 'lddu'
                    })
            except Exception as e:
                logger.error(f"LDDU 情感分析失败: {e}")
        threading.Thread(target=analyze, daemon=True).start()
    
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
            'has_emotion_analyzer': True
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
        # 停止所有实时处理（并集确保不遗漏）
        sessions = set(self.processing_flags.keys()) | set(self.processing_threads.keys()) | set(self.emotion_callbacks.keys()) | set(self.queue_manager.get_all_queues_info().keys())
        for session_id in list(sessions):
            try:
                self.stop_realtime_processing(session_id)
            except Exception as e:
                logger.warning(f"停止会话失败 {session_id}: {e}")
        
        # 停止视频处理器
        self.video_processor.stop_worker()
        
        logger.info("实时处理管理器已关闭")
    
    def _append_inference_log(self, model_type: str, inference_time: float, stats: Dict[str, Any]):
        """将一次推理记录写入NDJSON日志文件"""
        if not self.inference_log_file:
            return
        try:
            log_item = {
                'ts': time.time(),
                'model': model_type,
                'inference_time': inference_time,
                'count': stats.get('count', 0),
                'avg_time': stats.get('avg_time', 0),
                'min_time': stats.get('min_time', 0),
                'max_time': stats.get('max_time', 0),
                'source': 'app_runtime'
            }
            line = json.dumps(log_item, ensure_ascii=False)
            with self.log_lock:
                with open(self.inference_log_file, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")
        except Exception as e:
            # 使用debug级别，避免影响正常运行
            logger.debug(f"写入推理日志失败: {e}")