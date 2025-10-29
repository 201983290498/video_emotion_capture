# -*- coding: utf-8 -*-
"""
配置文件
包含应用的所有配置参数
"""

import os
from datetime import timedelta

# 加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装python-dotenv，跳过
    pass

class Config:
    """基础配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = False
    
    # 服务器配置
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', '5001'))
    
    # 文件存储配置
    UPLOAD_FOLDER = 'uploads'
    VIDEO_FOLDER = 'videos'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # 音视频配置
    VIDEO_CODEC = 'libx264'
    AUDIO_CODEC = 'aac'
    VIDEO_BITRATE = '1000k'
    AUDIO_BITRATE = '128k'
    FRAME_RATE = 10
    AUDIO_SAMPLE_RATE = 44100
    
    # 缓存配置
    BUFFER_DURATION = 10  # 缓存10秒的数据
    SAVE_INTERVAL = 1    # 每1秒保存一次视频
    
    # OpenAI API配置
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY') or ''
    OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE') or 'https://api.openai.com/v1'
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-3.5-turbo'
    OPENAI_MAX_TOKENS = int(os.environ.get('OPENAI_MAX_TOKENS', '1000'))
    OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE', '0.7'))
    OPENAI_TIMEOUT = int(os.environ.get('OPENAI_TIMEOUT', '30'))
    
    # 大模型API配置（向后兼容）
    LLM_API_URL = os.environ.get('LLM_API_URL') or 'https://api.openai.com/v1/chat/completions'
    LLM_API_KEY = OPENAI_API_KEY
    LLM_MODEL = OPENAI_MODEL
    
    # 本地Qwen3-Omni模型配置
    QWEN_MODEL_PATH = os.environ.get('QWEN_MODEL_PATH') or os.path.expanduser('~/models/Qwen/Qwen3-Omni-30B-A3B-Instruct')
    MAX_GPUS = int(os.environ.get('MAX_GPUS') or '4')  # 最大GPU数量 (0-3)
    LLM_MAX_TOKENS = 1000
    LLM_TEMPERATURE = 0.7

    # LDDU + HumanOmni 配置
    HUMANOMNI_MODEL_PATH = os.environ.get('HUMANOMNI_MODEL_PATH') or '/data/testmllm/models/R1-Omni-0.5B'
    LDDU_MODEL_DIR = os.environ.get('LDDU_MODEL_DIR') or '/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/src/models'
    LDDU_INIT_CHECKPOINT = os.environ.get('LDDU_INIT_CHECKPOINT') or '/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/cpkt_align/pytorch_model_7.bin.'
    EMOTION_LABELS = ['Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']
    
    # 聊天配置
    CHAT_HISTORY_LIMIT = int(os.environ.get('CHAT_HISTORY_LIMIT', '20'))  # 每个会话保留的历史消息数量
    CHAT_SESSION_TIMEOUT = int(os.environ.get('CHAT_SESSION_TIMEOUT', '3600'))  # 会话超时时间（秒）
    
    # WebSocket配置
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 推理性能日志文件（NDJSON），支持通过环境变量覆盖
    INFERENCE_LOG_FILE = os.environ.get('INFERENCE_LOG_FILE') or 'logs/inference_speed.ndjson'
    
    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        # 创建必要的目录
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.VIDEO_FOLDER, exist_ok=True)
        # 创建日志目录
        log_dir = os.path.dirname(Config.INFERENCE_LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # 生产环境下的安全配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-change-me'


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """获取当前配置"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])