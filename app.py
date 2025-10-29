"""
重构后的视频录制应用主文件
使用模块化架构，支持本地Qwen3-Omni和实时视频处理
"""
import os
import logging
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import time
from config import Config
from modules import RealtimeManager, get_openai_client, initialize_openai_client
from modules import get_lddu_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)

# 创建SocketIO实例
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
realtime_manager = None
openai_client = None
lddu_service = None


def init_realtime_manager():
    """初始化实时管理器"""
    global realtime_manager
    global lddu_service
    
    config = {
        'video_folder': Config.VIDEO_FOLDER,
        'frame_rate': Config.FRAME_RATE,
        'qwen_model_path': getattr(Config, 'QWEN_MODEL_PATH', None),
        'max_gpus': getattr(Config, 'MAX_GPUS', 4),
        # 新增 LDDU 配置
        'humanomni_model_path': getattr(Config, 'HUMANOMNI_MODEL_PATH', None),
        'lddu_model_dir': getattr(Config, 'LDDU_MODEL_DIR', None),
        'lddu_init_checkpoint': getattr(Config, 'LDDU_INIT_CHECKPOINT', None),
        'emotion_labels': getattr(Config, 'EMOTION_LABELS', None),
        # 推理日志文件路径
        'inference_log_file': getattr(Config, 'INFERENCE_LOG_FILE', None),
    }
    
    realtime_manager = RealtimeManager(config)
    logger.info("实时管理器初始化完成")

    # 初始化 LDDU 服务
    lddu_service = get_lddu_service()
    ok = lddu_service.load_model(
        humanomni_model_path=Config.HUMANOMNI_MODEL_PATH,
        model_dir=Config.LDDU_MODEL_DIR,
        init_model_path=getattr(Config, 'LDDU_INIT_CHECKPOINT', None),
        emotion_labels=getattr(Config, 'EMOTION_LABELS', None)
    )
    if ok:
        logger.info("LDDU Service 初始化成功")
    else:
        logger.error(f"LDDU Service 初始化失败: {lddu_service.load_error}")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/videos')
def list_videos():
    """获取视频列表"""
    try:
        videos = []
        if os.path.exists(Config.VIDEO_FOLDER):
            for filename in os.listdir(Config.VIDEO_FOLDER):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    filepath = os.path.join(Config.VIDEO_FOLDER, filename)
                    videos.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'created': os.path.getctime(filepath)
                    })
        return jsonify({'videos': videos})
    except Exception as e:
        logger.error(f"获取视频列表失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos')
def api_list_videos():
    """API: 获取视频列表"""
    return list_videos()

@app.route('/videos/<filename>')
def serve_video(filename):
    """提供视频文件"""
    return send_from_directory(Config.VIDEO_FOLDER, filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI聊天接口"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message.strip():
            return jsonify({'error': '消息不能为空'}), 400
        
        # 使用OpenAI客户端生成响应  
        response = generate_ai_response(message, session_id)
        
        return jsonify({
            'response': response,  # 返回对话结果
            'timestamp': time.time(),
            'session_id': session_id,  # 会话ID
            'api_available': openai_client.is_api_available() if openai_client else False
        })
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """清除聊天历史"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if openai_client:
            openai_client.clear_conversation_history(session_id)
        
        return jsonify({
            'message': '聊天历史已清除',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"清除聊天历史失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """获取聊天历史"""
    try:
        session_id = request.args.get('session_id', 'default')
        
        if openai_client:
            history = openai_client.get_conversation_history(session_id)
            # 过滤掉系统消息，只返回用户和助手的对话
            filtered_history = [msg for msg in history if msg.get('role') != 'system']
        else:
            filtered_history = []
        
        return jsonify({
            'history': filtered_history,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}")
        return jsonify({'error': str(e)}), 500

def generate_ai_response(message: str, session_id: str = "default") -> str:
    """生成AI响应"""
    try:
        # 使用OpenAI客户端生成响应
        if openai_client:
            return openai_client.get_chat_response(message, session_id)
        else:
            logger.warning("OpenAI客户端未初始化，使用降级回复")
            return _get_fallback_response(message)
            
    except Exception as e:
        logger.error(f"生成AI响应失败: {e}")
        return "抱歉，我现在无法处理您的请求。请稍后再试或联系技术支持。"

def _get_fallback_response(message: str) -> str:
    """降级回复函数（当OpenAI客户端不可用时）"""
    message_lower = message.lower()
    
    # 问候语
    if any(word in message_lower for word in ['你好', 'hello', 'hi', '您好']):
        return "您好！我是AI助手，很高兴为您服务。虽然当前AI服务不可用，但我仍然可以为您提供基本的帮助和指导。"  # AI服务不可用??
    
    # 关于功能的问题
    elif any(word in message_lower for word in ['功能', '能做什么', '怎么用']):
        return "我可以帮您：\n1. 实时录制视频和音频\n2. 进行情感分析\n3. 管理和播放录制的视频\n4. 回答您的问题\n\n您想了解哪个功能的详细信息？"
    
    # 关于录制的问题
    elif any(word in message_lower for word in ['录制', '录像', '视频']):
        return "关于视频录制功能：\n• 点击'开启摄像头'开始\n• 点击'开始录制'进行录制\n• 系统会自动保存10秒队列，每5秒合成一次视频\n• 录制的视频会显示在左侧列表中"
    
    # 关于情感分析的问题
    elif any(word in message_lower for word in ['情感', '分析', '识别']):
        return "情感分析功能：\n• 使用Qwen3-Omni模型分析视频中的情感\n• 支持音频和视频的多模态分析\n• 分析结果会实时显示在界面上\n• 目前模型正在加载中，请稍后再试"
    
    # 技术问题
    elif any(word in message_lower for word in ['错误', '问题', '不工作', '失败']):
        return "如果遇到技术问题：\n1. 请检查摄像头和麦克风权限\n2. 确保浏览器支持WebRTC\n3. 查看控制台是否有错误信息\n4. 尝试刷新页面重新开始\n\n具体是什么问题呢？"
    
    # 默认智能回复
    else:
        responses = [
            f"关于'{message}'，这是一个很有趣的话题。您想了解更多相关信息吗？",
            f"我理解您提到了'{message}'。有什么具体的问题我可以帮您解答吗？",
            f"感谢您的提问。关于'{message}'，我建议您可以尝试使用我们的视频录制功能来探索更多可能性。",
            "我正在学习中，感谢您的耐心。您可以问我关于视频录制、情感分析或系统功能的问题。"
        ]
        import random
        return random.choice(responses)

@app.route('/status')
def get_status():
    """获取系统状态 返回所有会话与视频处理器状态"""
    try:
        if realtime_manager:
            status = realtime_manager.get_all_sessions_status()
        else:
            status = {'error': '实时管理器未初始化'}
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return jsonify({'error': str(e)}), 500

# SocketIO事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info(f"客户端已连接: {request.sid}")
    emit('connected', {'message': '连接成功'})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    logger.info(f"客户端已断开: {request.sid}")
    
    # 清理会话
    if realtime_manager:
        realtime_manager.cleanup_session(request.sid)

@socketio.on('start_recording')
def handle_start_recording():
    """开始录制"""
    try:
        session_id = request.sid
        logger.info(f"开始录制: session_id={session_id}")
        
        # 定义情感分析结果回调
        def emotion_callback(result):
            # 保持前端事件统一
            socketio.emit('emotion_result', result, room=session_id)
        
        # 开始实时处理
        if realtime_manager:
            realtime_manager.start_realtime_processing(session_id, emotion_callback)
        
        emit('recording_started', {'session_id': session_id})
        
    except Exception as e:
        logger.error(f"开始录制失败: {e}")
        emit('error', {'message': str(e)})

@socketio.on('stop_recording')
def handle_stop_recording():
    """停止录制"""
    try:
        session_id = request.sid
        logger.info(f"停止录制: session_id={session_id}")
        if realtime_manager:
            realtime_manager.stop_realtime_processing(session_id)
        emit('recording_stopped', {'session_id': session_id})
    except Exception as e:
        logger.error(f"停止录制失败: {e}")
        emit('error', {'message': str(e)})


@socketio.on('video_frame')
def handle_video_frame(data):
    """处理视频帧并返回ACK"""
    try:
        session_id = request.sid
        frame_data = data.get('frame')
        client_ts = data.get('client_ts')
        
        if frame_data and realtime_manager:
            # 移除data URL前缀
            if isinstance(frame_data, str) and frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            # 清理空白并补齐填充，避免后端解码失败
            img_bytes = None
            if isinstance(frame_data, str):
                import base64
                s = frame_data.strip().replace('\n', '').replace('\r', '')
                missing_padding = (-len(s)) % 4
                if missing_padding:
                    s += '=' * missing_padding
                try:
                    img_bytes = base64.b64decode(s, validate=False)
                except Exception:
                    try:
                        img_bytes = base64.urlsafe_b64decode(s)
                    except Exception as e:
                        logger.warning(f"视频帧base64解码失败，丢弃该帧: {e}")
                        img_bytes = None
            elif isinstance(frame_data, (bytes, bytearray)):
                img_bytes = bytes(frame_data)
            else:
                img_bytes = None

            if img_bytes:
                realtime_manager.add_video_frame(session_id, img_bytes)
            # 返回ACK，携带服务器时间戳与队列状态
            status = realtime_manager.get_session_status(session_id) if realtime_manager else {}
            emit('video_frame_ack', {
                'session_id': session_id,
                'server_ts': time.time(),
                'client_ts': client_ts,
                'queue_info': status.get('queue_info', {})
            })
        
    except Exception as e:
        logger.error(f"处理视频帧失败: {e}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """处理音频块"""
    try:
        session_id = request.sid
        audio_data = data.get('audio')
        client_ts = data.get('client_ts')
        
        if audio_data and realtime_manager:
            # 支持 DataURL 与纯 base64 两种形式，统一解码为字节
            decoded_bytes = None
            try:
                if isinstance(audio_data, str):
                    # 去掉可能的 data:audio 前缀
                    if audio_data.startswith('data:audio'):
                        audio_data = audio_data.split(',')[1]
                    # 清理空白并补齐填充
                    s = audio_data.strip().replace('\n', '').replace('\r', '')
                    missing_padding = (-len(s)) % 4
                    if missing_padding:
                        s += '=' * missing_padding
                    try:
                        decoded_bytes = base64.b64decode(s, validate=False)
                    except Exception:
                        decoded_bytes = base64.urlsafe_b64decode(s)
                elif isinstance(audio_data, (bytes, bytearray)):
                    decoded_bytes = bytes(audio_data)
            except Exception as e:
                logger.warning(f"音频块base64解码失败，丢弃该块: {e}")
                decoded_bytes = None

            if decoded_bytes:
                realtime_manager.add_audio_chunk(session_id, decoded_bytes)
            status = realtime_manager.get_session_status(session_id) if realtime_manager else {}
            emit('audio_chunk_ack', {
                'session_id': session_id,
                'server_ts': time.time(),
                'client_ts': client_ts,
                'queue_info': status.get('queue_info', {})
            })
    except Exception as e:
        logger.error(f"处理音频块失败: {e}")


@app.route('/api/inference/stats', methods=['GET'])
def get_inference_stats():
    """获取推理时间统计"""
    try:
        if not realtime_manager:
            return jsonify({'error': '实时管理器未初始化'}), 503
        
        stats = realtime_manager.get_inference_stats()
        
        return jsonify({
            'stats': stats,
            'timestamp': time.time(),
            'message': '推理统计获取成功'
        })
        
    except Exception as e:
        logger.error(f"获取推理统计失败: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import time
    
    # 确保输出目录存在
    os.makedirs(Config.VIDEO_FOLDER, exist_ok=True)
    
    # 初始化OpenAI客户端
    try:
        openai_client = initialize_openai_client()
        logger.info("OpenAI客户端初始化成功")
    except Exception as e:
        logger.warning(f"OpenAI客户端初始化失败: {e}")
        openai_client = None
    
    # 初始化实时管理器
    init_realtime_manager()
    
    try:
        logger.info("启动视频录制应用...")
        logger.info(f"应用启动在 http://localhost:{Config.PORT}")
        
        # 启动应用
        socketio.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("应用正在关闭...")
    finally:
        # 清理资源
        if realtime_manager:
            realtime_manager.shutdown()
        logger.info("应用已关闭")