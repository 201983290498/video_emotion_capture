CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100   # 损失计算中忽略的标签索引

# Image arguments
IMAGE_TOKEN_INDEX = -200  # 图像令牌索引（负索引表示特殊令牌，不占用常规词表位置）
IMAGE_TOKEN_PATCH = -300  # 图像补丁令牌索引（负索引表示特殊令牌，不占用常规词表位置）
DEFAULT_IMAGE_TOKEN = "<image>"    # 图像整体令牌
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # 图像分块补丁令牌
DEFAULT_IM_START_TOKEN = "<im_start>"  # 图像内容起始标记
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"   # 图像占位符（预处理或调试时使用）

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 8  # 处理视频时默认采样的帧数
MAX_FRAMES = 32 # 最大允许采样的帧数（避免计算量过大）
NUM_FRAMES_PER_SECOND = 1  # 每秒视频采样的帧数（控制时间粒度）

# Audio arguments
AUDIO_TOKEN_INDEX = -202   # 音频令牌的索引
DEFAULT_AUDIO_TOKEN = "<audio>"  # 文本中表示音频的默认令牌
# 模态令牌到索引的映射
MODAL_INDEX_MAP = {
    "<audio>": -202,
    "<video>": -201,
    "<image>": -200,
}
# 索引到模态令牌的反向映射
MODAL_INDEX_REMAP = {v: k for k, v in MODAL_INDEX_MAP.items()}
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'AUDIO': "<au_start>", 'THERMAL': "<th_start>", 'DEPTH': "<de_start>"}
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'AUDIO': "<au_end>", 'THERMAL': "<th_end>", 'DEPTH': "<de_end>"}