# -*- coding: utf-8 -*-
"""
OpenAI API客户端模块
提供与OpenAI API的交互功能，包括聊天对话、错误处理和重试机制
"""

import logging
from typing import Dict, List
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API客户端"""
    
    def __init__(self):
        """初始化OpenAI客户端"""
        self.client = None
        # 优先使用新的OpenAI配置，向后兼容旧配置
        self.api_key = Config.OPENAI_API_KEY or Config.LLM_API_KEY
        self.api_base = Config.OPENAI_API_BASE
        self.model = Config.OPENAI_MODEL or Config.LLM_MODEL
        self.max_tokens = Config.OPENAI_MAX_TOKENS or Config.LLM_MAX_TOKENS
        self.temperature = Config.OPENAI_TEMPERATURE or Config.LLM_TEMPERATURE
        self.timeout = Config.OPENAI_TIMEOUT
        self.is_available = False
        
        # 对话历史存储
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_length = Config.CHAT_HISTORY_LIMIT
        
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            if not self.api_key or self.api_key == 'your-api-key-here':
                logger.warning("OpenAI API密钥未配置，聊天功能将使用本地回复")
                return
            
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            
            # 测试连接
            self._test_connection()
            self.is_available = True
            logger.info(f"OpenAI客户端初始化成功 - 模型: {self.model}")
            
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {e}")
            self.is_available = False
    
    def _test_connection(self):
        """测试API连接"""
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=10
            )
        except Exception as e:
            logger.error(f"OpenAI API连接测试失败: {e}")
            raise
    
    def get_chat_response(self, message: str, session_id: str = "default") -> str:
        """
        获取聊天响应
        
        Args:
            message: 用户消息
            session_id: 会话ID，用于维护对话历史
            
        Returns:
            AI回复内容
        """
        if not self.is_available:
            return self._get_fallback_response(message)
        
        try:
            # 获取或创建对话历史
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = [
                    {
                        "role": "system",
                        "content": "你是一个智能的AI助手，专门帮助用户使用视频录制和情感分析应用。你可以回答关于应用功能、技术问题和一般性问题。请用友好、专业的语气回复用户。"
                    }
                ]
            
            # 添加用户消息到历史
            self.conversation_history[session_id].append({
                "role": "user",
                "content": message
            })
            
            # 限制对话历史长度（保留系统消息 + 最近N轮对话）
            max_messages = self.max_history_length * 2 + 1  # 系统消息 + N轮用户助手对话
            if len(self.conversation_history[session_id]) > max_messages:
                system_msg = self.conversation_history[session_id][0]
                recent_msgs = self.conversation_history[session_id][-(self.max_history_length * 2):]
                self.conversation_history[session_id] = [system_msg] + recent_msgs
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history[session_id],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            # 提取回复内容
            assistant_reply = response.choices[0].message.content.strip()
            # 添加助手回复到历史
            self.conversation_history[session_id].append({
                "role": "assistant",
                "content": assistant_reply
            })
            
            logger.info(f"OpenAI API调用成功，会话: {session_id}")
            return assistant_reply
            
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return self._get_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """
        获取降级回复（当API不可用时）
        
        Args:
            message: 用户消息
            
        Returns:
            降级回复内容
        """
        message_lower = message.lower()
        
        # 问候语
        if any(word in message_lower for word in ['你好', 'hello', 'hi', '您好']):
            return "您好！我是AI助手。当前处于离线模式，但仍可为您提供基本的功能指导和帮助。"
        
        # 功能相关问题
        elif any(word in message_lower for word in ['功能', '能做什么', '怎么用', '录制', '视频', '情感', '分析']):
            return "本应用支持视频录制和情感分析功能。您可以开启摄像头进行录制，系统会自动分析视频中的情感表达。如需详细帮助，请查看界面上的功能按钮。"
        
        # 技术问题
        elif any(word in message_lower for word in ['错误', '问题', '不工作', '失败', 'api', '配置']):
            return "遇到技术问题时，请检查：1) 浏览器摄像头权限 2) 网络连接状态 3) API配置是否正确。如问题持续，建议刷新页面重试。"
        
        # 默认回复
        else:
            return f"感谢您的提问。虽然当前处于离线模式，但我可以帮您了解视频录制、情感分析等功能。有什么具体问题吗？"
    
    def clear_conversation_history(self, session_id: str = "default"):
        """
        清除对话历史
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"已清除会话历史: {session_id}")
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话历史列表
        """
        return self.conversation_history.get(session_id, [])
    
    def is_api_available(self) -> bool:
        """
        检查API是否可用
        
        Returns:
            API可用状态
        """
        return self.is_available


# 全局OpenAI客户端实例
openai_client = None


def get_openai_client() -> OpenAIClient:
    """
    获取OpenAI客户端实例（单例模式）
    
    Returns:
        OpenAI客户端实例
    """
    global openai_client
    if openai_client is None:
        openai_client = OpenAIClient()
    return openai_client


def initialize_openai_client():
    """初始化OpenAI客户端（兼容性函数）"""
    return get_openai_client()