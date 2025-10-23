"""
视频队列管理模块
实现10秒视频队列，每秒合成最新5秒数据
"""
import time
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TimestampedData:
    """带时间戳的数据"""
    def __init__(self, data, timestamp: float):
        self.data = data
        self.timestamp = timestamp

class VideoQueue:
    """视频数据队列管理器"""
    
    def __init__(self, max_duration: float = 10.0):
        """
        初始化视频队列
        Args:
            max_duration: 队列最大时长（秒）
        """
        self.max_duration = max_duration
        self.video_frames = deque()  # 视频帧队列
        self.audio_chunks = deque()  # 音频块队列
        self.lock = threading.RLock()
        self._last_cleanup = time.time()
        
    def add_video_frame(self, frame_data):
        """添加视频帧"""
        with self.lock:
            timestamp = time.time()
            self.video_frames.append(TimestampedData(frame_data, timestamp))
            self._cleanup_old_data()
    
    def add_audio_chunk(self, audio_data):
        """添加音频块"""
        with self.lock:
            timestamp = time.time()
            self.audio_chunks.append(TimestampedData(audio_data, timestamp))
            self._cleanup_old_data()
    
    def get_recent_data(self, duration: float = 5.0) -> Tuple[List, List]:
        """
        获取最近指定时长的数据
        Args:
            duration: 获取数据的时长（秒）
        Returns:
            (video_frames, audio_chunks)
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - duration
            
            # 获取最近的视频帧
            recent_video = []
            for frame in reversed(self.video_frames):
                if frame.timestamp >= cutoff_time:
                    recent_video.insert(0, frame.data)
                else:
                    break
            
            # 获取最近的音频块
            recent_audio = []
            for chunk in reversed(self.audio_chunks):
                if chunk.timestamp >= cutoff_time:
                    recent_audio.insert(0, chunk.data)
                else:
                    break
            
            return recent_video, recent_audio
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        current_time = time.time()
        
        # 避免频繁清理
        if current_time - self._last_cleanup < 1.0:
            return
            
        cutoff_time = current_time - self.max_duration
        
        # 清理过期视频帧
        while self.video_frames and self.video_frames[0].timestamp < cutoff_time:
            self.video_frames.popleft()
        
        # 清理过期音频块
        while self.audio_chunks and self.audio_chunks[0].timestamp < cutoff_time:
            self.audio_chunks.popleft()
        
        self._last_cleanup = current_time
    
    def get_queue_info(self) -> Dict:
        """获取队列状态信息"""
        with self.lock:
            current_time = time.time()
            
            video_count = len(self.video_frames)
            audio_count = len(self.audio_chunks)
            
            # 计算数据时长
            video_duration = 0
            if self.video_frames:
                oldest_video = self.video_frames[0].timestamp
                video_duration = current_time - oldest_video
            
            audio_duration = 0
            if self.audio_chunks:
                oldest_audio = self.audio_chunks[0].timestamp
                audio_duration = current_time - oldest_audio
            
            return {
                'video_frames': video_count,
                'audio_chunks': audio_count,
                'video_duration': video_duration,
                'audio_duration': audio_duration,
                'max_duration': self.max_duration
            }
    
    def clear(self):
        """清空队列"""
        with self.lock:
            self.video_frames.clear()
            self.audio_chunks.clear()

class VideoQueueManager:
    """视频队列管理器"""
    
    def __init__(self):
        self.queues: Dict[str, VideoQueue] = {}
        self.lock = threading.RLock()
    
    def get_queue(self, session_id: str) -> VideoQueue:
        """获取或创建指定会话的队列"""
        with self.lock:
            if session_id not in self.queues:
                self.queues[session_id] = VideoQueue()
            return self.queues[session_id]
    
    def remove_queue(self, session_id: str):
        """移除指定会话的队列"""
        with self.lock:
            if session_id in self.queues:
                del self.queues[session_id]
    
    def get_all_queues_info(self) -> Dict:
        """获取所有队列的状态信息"""
        with self.lock:
            info = {}
            for session_id, queue in self.queues.items():
                info[session_id] = queue.get_queue_info()
            return info