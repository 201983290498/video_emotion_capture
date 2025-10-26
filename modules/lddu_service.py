import os
import sys
import time
import torch
from typing import Dict, Any, List, Optional

# 将 R1-Omni-main 目录加入 Python 路径，以便导入 lddu_inference.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
R1_OMNI_DIR = os.path.join(PROJECT_ROOT, 'R1-Omni-main')
if R1_OMNI_DIR not in sys.path:
    sys.path.append(R1_OMNI_DIR)

# 动态导入 lddu_inference 模块
try:
    from lddu_inference import LDDUInference, create_lddu_config
except Exception as e:
    LDDUInference = None
    create_lddu_config = None
    _IMPORT_ERROR = str(e)
else:
    _IMPORT_ERROR = None

DEFAULT_EMOTION_LABELS = ['Happy', 'Sad', 'Anger', 'Surprise', 'Disgust', 'Fear']

class LDDUService:
    def __init__(self):
        self.is_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference: Optional[LDDUInference] = None
        self.emotion_labels: List[str] = DEFAULT_EMOTION_LABELS
        self.load_error: Optional[str] = None
        self.model_info: Dict[str, Any] = {}

    def load_model(self,
                   humanomni_model_path: str,
                   model_dir: str,
                   init_model_path: Optional[str] = None,
                   emotion_labels: Optional[List[str]] = None) -> bool:
        """
        加载 HumanOmni + LDDU 模型，用于特征提取与情感推理。
        """
        if _IMPORT_ERROR:
            self.load_error = f"导入 lddu_inference 失败: {_IMPORT_ERROR}"
            self.is_loaded = False
            return False

        try:
            cfg = create_lddu_config()
            if emotion_labels:
                self.emotion_labels = emotion_labels
                # 保证类别数量与标签一致
                if hasattr(cfg, 'num_classes'):
                    cfg.num_classes = len(emotion_labels)

            self.inference = LDDUInference(
                task_config=cfg,
                humanomni_model_path=humanomni_model_path,
                model_dir=model_dir,
                device=self.device,
                init_model_path=init_model_path
            )
            self.is_loaded = True
            self.model_info = {
                'humanomni_model_path': humanomni_model_path,
                'lddu_model_dir': model_dir,
                'lddu_init_checkpoint': init_model_path,
                'device': str(self.device),
                'num_labels': len(self.emotion_labels)
            }
            return True
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            self.inference = None
            return False

    def analyze_video_emotion(self,
                              video_path: str,
                              visual_prompts: Optional[List[str]] = None,
                              threshold: float = 0.3) -> Dict[str, Any]:
        """
        使用 LDDU 推理进行情感识别，并返回统一结构。
        """
        if not self.is_loaded or not self.inference:
            return {
                'success': False,
                'error': 'LDDU Service 未加载',
                'details': {'load_error': self.load_error}
            }
        if not video_path or not os.path.exists(video_path):
            return {'success': False, 'error': f'视频文件不存在: {video_path}'}

        start = time.time()
        try:
            predict_labels, pred_scores, timing_info = self.inference.recognize_emotion(
                video_path=video_path,
                visual_prompts=visual_prompts,
                threshold=threshold
            )

            # 取 batch 第一项
            if predict_labels.ndim > 1:
                predict_labels = predict_labels[0]
            if pred_scores.ndim > 1:
                pred_scores = pred_scores[0]

            scores = pred_scores.detach().cpu().numpy().tolist()
            labels_bin = predict_labels.detach().cpu().numpy().tolist()

            pairs = list(zip(self.emotion_labels, scores, labels_bin))
            pairs.sort(key=lambda x: x[1], reverse=True)

            primary_emotion, primary_confidence = None, 0.0
            # 优先选择被检测的标签，否则取 top-1
            for name, conf, detected in pairs:
                if int(detected) == 1:
                    primary_emotion, primary_confidence = name, float(conf)
                    break
            if primary_emotion is None and pairs:
                primary_emotion, primary_confidence = pairs[0][0], float(pairs[0][1])

            return {
                'success': True,
                'emotion': primary_emotion,
                'confidence': primary_confidence,
                'multi_labels': [
                    {'label': name, 'score': float(conf), 'detected': bool(detected)}
                    for name, conf, detected in pairs
                ],
                'timing': timing_info,
                'video_path': video_path,
                'timestamp': time.time(),
                'service': 'lddu_mmer + humanomni',
                'model_info': self.model_info
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'video_path': video_path,
                'timestamp': time.time()
            }

# 单例服务工厂
_lddu_service_instance: Optional[LDDUService] = None

def get_lddu_service() -> LDDUService:
    global _lddu_service_instance
    if _lddu_service_instance is None:
        _lddu_service_instance = LDDUService()
    return _lddu_service_instance
