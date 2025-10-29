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
                # 从队列获取任务，超时 1 秒
                task = self.processing_queue.get(timeout=1)
                self._process_task(task)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理视频任务失败: {e}")

    def _decode_frame(self, frame_data):
        """将输入帧数据解码为OpenCV图像。
        支持纯base64字符串或numpy数组/已有图像。
        失败返回None。
        """
        try:
            # 如果已经是numpy图像
            if hasattr(frame_data, 'shape'):
                return frame_data

            # 如果是字节串，直接视为JPEG/PNG字节
            if isinstance(frame_data, (bytes, bytearray)):
                nparr = np.frombuffer(frame_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    try:
                        b0 = frame_data[0] if len(frame_data) > 0 else None
                        b1 = frame_data[1] if len(frame_data) > 1 else None
                        logger.warning(f"cv2.imdecode失败 bytes_len={len(frame_data)} head={b0},{b1}")
                    except Exception:
                        pass
                    # 回退到 PIL 解码
                    try:
                        from PIL import Image, ImageFile
                        from io import BytesIO
                        ImageFile.LOAD_TRUNCATED_IMAGES = True
                        pil_img = Image.open(BytesIO(bytes(frame_data)))
                        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        logger.warning("PIL解码也失败，丢弃该帧(bytes)")
                        return None
                return img

            # 如果是字符串，先按base64解码
            if isinstance(frame_data, str):
                import base64, re
                # 处理可能残留的 data URL 前缀
                if frame_data.startswith('data:image'):
                    parts = frame_data.split(',', 1)
                    frame_data = parts[1] if len(parts) > 1 else frame_data
                # 清理所有空白符
                s = re.sub(r"\s+", "", frame_data.strip())
                # 补齐'='填充
                missing_padding = (-len(s)) % 4
                if missing_padding:
                    s += '=' * missing_padding
                try:
                    img_bytes = base64.b64decode(s, validate=False)
                except Exception:
                    # 回退到 urlsafe base64
                    img_bytes = base64.urlsafe_b64decode(s)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    try:
                        b0 = img_bytes[0] if len(img_bytes) > 0 else None
                        b1 = img_bytes[1] if len(img_bytes) > 1 else None
                        logger.warning(f"cv2.imdecode失败(base64) bytes_len={len(img_bytes)} head={b0},{b1}")
                    except Exception:
                        pass
                    # 回退到 PIL 解码
                    try:
                        from PIL import Image, ImageFile
                        from io import BytesIO
                        ImageFile.LOAD_TRUNCATED_IMAGES = True
                        pil_img = Image.open(BytesIO(img_bytes))
                        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        logger.warning("PIL解码也失败，丢弃该帧(base64)")
                        return None
                return img

            return None
        except Exception as e:
            logger.debug(f"帧解码失败: {e}")
            return None
    
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
            frame = None
            for candidate in video_frames:
                frame = self._decode_frame(candidate)
                if frame is not None:
                    break
            if frame is None:
                logger.error("无有效首帧，无法创建临时视频")
                return None
            
            height, width = frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_path, fourcc, self.frame_rate, (width, height))
            
            # 写入所有帧
            for frame_data in video_frames:
                frame = self._decode_frame(frame_data)
                
                if frame is not None:
                    # 确保帧尺寸一致
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                else:
                    # 跳过不可解码的帧
                    continue
            
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
            # 将所有块统一为字节
            normalized_chunks = []
            for chunk in audio_chunks:
                if isinstance(chunk, (bytes, bytearray)):
                    normalized_chunks.append(bytes(chunk))
                elif isinstance(chunk, str):
                    # 兜底：如果仍有字符串，尝试按base64解码；失败则按utf-8字节处理
                    try:
                        import base64
                        s = chunk.strip().replace('\n', '').replace('\r', '')
                        missing_padding = (-len(s)) % 4
                        if missing_padding:
                            s += '=' * missing_padding
                        normalized_chunks.append(base64.b64decode(s, validate=False))
                    except Exception:
                        normalized_chunks.append(chunk.encode('utf-8', errors='ignore'))
                else:
                    # 非预期类型，跳过
                    continue

            combined_audio = b''.join(normalized_chunks)

            # 判断是否为 WebM (Matroska) 格式，根据 EBML 魔数 \x1A\x45\xDF\xA3
            is_webm = combined_audio[:4] == b'\x1aE\xdf\xa3'
            is_wav = combined_audio[:4] == b'RIFF'

            if is_webm:
                # 直接写入 .webm，交由 FFmpeg 重编码到 AAC
                temp_fd, temp_path = tempfile.mkstemp(suffix='.webm')
                os.close(temp_fd)
                with open(temp_path, 'wb') as f:
                    f.write(combined_audio)
                logger.info(f"临时音频(WebM)文件创建成功: {temp_path}")
                return temp_path
            elif is_wav:
                # 已含有 WAV 头，直接写出
                temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                with open(temp_path, 'wb') as f:
                    f.write(combined_audio)
                logger.info(f"临时音频(WAV)文件创建成功: {temp_path}")
                return temp_path
            else:
                # 视为原始PCM，包装为WAV写入
                temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位
                    wav_file.setframerate(44100)  # 44.1kHz
                    wav_file.writeframes(combined_audio)
                logger.info(f"临时音频(PCM->WAV)文件创建成功: {temp_path}")
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