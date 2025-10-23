"""
视频处理模块
负责视频合成、音视频同步等功能
"""
import os
import cv2
import wave
import tempfile
import threading
import queue
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, output_dir: str, frame_rate: int = 10):
        """
        初始化视频处理器
        Args:
            output_dir: 输出目录
            frame_rate: 帧率
        """
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.processing_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def start_worker(self):
        """启动工作线程"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("视频处理工作线程已启动")
    
    def stop_worker(self):
        """停止工作线程"""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("视频处理工作线程已停止")
    
    def _worker_loop(self):
        """工作线程主循环"""
        while self.is_running:
            try:
                # 从队列获取任务，超时1秒
                task = self.processing_queue.get(timeout=1)
                self._process_task(task)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理视频任务失败: {e}")
    
    def _process_task(self, task: Dict[str, Any]):
        """处理单个任务"""
        task_type = task.get('type')
        
        if task_type == 'compose_video':
            self._compose_video_task(task)
        else:
            logger.warning(f"未知任务类型: {task_type}")
    
    def _compose_video_task(self, task: Dict[str, Any]):
        """处理视频合成任务"""
        try:
            video_frames = task.get('video_frames', [])
            audio_chunks = task.get('audio_chunks', [])
            session_id = task.get('session_id', 'unknown')
            callback = task.get('callback')
            
            if not video_frames:
                logger.warning("没有视频帧数据，跳过合成")
                return
            
            # 生成输出文件名
            timestamp = int(time.time())
            output_filename = f"video_{session_id}_{timestamp}.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 合成视频
            success = self._create_video_with_audio(
                video_frames, audio_chunks, output_path
            )
            
            if success and callback:
                callback({
                    'success': True,
                    'video_path': output_path,
                    'filename': output_filename,
                    'session_id': session_id
                })
            elif callback:
                callback({
                    'success': False,
                    'error': '视频合成失败',
                    'session_id': session_id
                })
                
        except Exception as e:
            logger.error(f"视频合成任务失败: {e}")
            if task.get('callback'):
                task['callback']({
                    'success': False,
                    'error': str(e),
                    'session_id': task.get('session_id', 'unknown')
                })
    
    def _create_video_with_audio(self, video_frames: List, audio_chunks: List, output_path: str) -> bool:
        """创建带音频的视频文件"""
        temp_video_path = None
        temp_audio_path = None
        
        try:
            # 创建临时视频文件
            temp_video_path = self._create_temp_video(video_frames)
            if not temp_video_path:
                return False
            
            # 创建临时音频文件
            if audio_chunks:
                temp_audio_path = self._create_temp_audio(audio_chunks)
            
            # 合并音视频
            if temp_audio_path:
                success = self._merge_audio_video(temp_video_path, temp_audio_path, output_path)
            else:
                # 只有视频，直接转换格式
                success = self._convert_video_format(temp_video_path, output_path)
            
            return success
            
        except Exception as e:
            logger.error(f"创建视频失败: {e}")
            return False
        finally:
            # 清理临时文件
            for temp_path in [temp_video_path, temp_audio_path]:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    def _create_temp_video(self, video_frames: List) -> Optional[str]:
        """创建临时视频文件"""
        if not video_frames:
            return None
        
        try:
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix='.avi')
            os.close(temp_fd)
            
            # 获取第一帧来确定视频尺寸
            first_frame = video_frames[0]
            if isinstance(first_frame, str):
                # 如果是base64编码的图片
                import base64
                img_data = base64.b64decode(first_frame)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = first_frame
            
            height, width = frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_path, fourcc, self.frame_rate, (width, height))
            
            # 写入所有帧
            for frame_data in video_frames:
                if isinstance(frame_data, str):
                    # base64解码
                    import base64
                    img_data = base64.b64decode(frame_data)
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = frame_data
                
                if frame is not None:
                    # 确保帧尺寸一致
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
            
            out.release()
            logger.info(f"临时视频文件创建成功: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"创建临时视频失败: {e}")
            return None
    
    def _create_temp_audio(self, audio_chunks: List) -> Optional[str]:
        """创建临时音频文件"""
        if not audio_chunks:
            return None
        
        try:
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # 合并音频数据
            combined_audio = b''.join(audio_chunks)
            
            # 写入WAV文件
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(44100)  # 44.1kHz
                wav_file.writeframes(combined_audio)
            
            logger.info(f"临时音频文件创建成功: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"创建临时音频失败: {e}")
            return None
    
    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """使用FFmpeg合并音视频"""
        try:
            (
                ffmpeg
                .output(
                    ffmpeg.input(video_path),
                    ffmpeg.input(audio_path),
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    strict='experimental'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"音视频合并成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"音视频合并失败: {e}")
            return False
    
    def _convert_video_format(self, input_path: str, output_path: str) -> bool:
        """转换视频格式为MP4"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vcodec='libx264')
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"视频格式转换成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"视频格式转换失败: {e}")
            return False
    
    def add_composition_task(self, video_frames: List, audio_chunks: List, 
                           session_id: str, callback=None):
        """添加视频合成任务"""
        task = {
            'type': 'compose_video',
            'video_frames': video_frames,
            'audio_chunks': audio_chunks,
            'session_id': session_id,
            'callback': callback
        }
        self.processing_queue.put(task)
        logger.info(f"视频合成任务已添加到队列: session_id={session_id}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.processing_queue.qsize()
    
    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return {
            'is_running': self.is_running,
            'queue_size': self.get_queue_size(),
            'output_dir': self.output_dir,
            'frame_rate': self.frame_rate
        }