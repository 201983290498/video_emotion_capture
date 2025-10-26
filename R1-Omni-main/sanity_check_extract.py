import torch
from extract_multimodal_features import load_models_for_device, batch_extract_multimodal_features

# 替换为你本地的三段短视频路径（建议10-30秒以内）
video_paths = [
    "/data/testmllm/project/video_capture/R1-Omni-main/data/0.mp4",
    "/data/testmllm/project/video_capture/R1-Omni-main/data/5.mp4",
    "/data/testmllm/project/video_capture/R1-Omni-main/data/6.mp4",
]

# 指向 HumanOmni 权重所在目录（必须包含 vision_tower/ 与 audio_tower/）
model_path = "/data/testmllm/models/R1-Omni-0.5B"

models = load_models_for_device(device=("cuda" if torch.cuda.is_available() else "cpu"),
                                model_path=model_path)

results = batch_extract_multimodal_features(video_paths, models=models, batch_size=4)

for i, r in enumerate(results):
    vf, af, tf = r['visual'], r['audio'], r['text']
    print(f"=== Sample {i} ===")
    # 视觉
    if vf is not None:
        print("visual:", tuple(vf.shape), float(vf.mean()), float(vf.std()), vf.dtype, vf.device)
    else:
        print("visual: None")
    # 音频
    if af is not None:
        print("audio:", tuple(af.shape), float(af.mean()), float(af.std()), af.dtype, af.device)
    else:
        print("audio: None")
    # 文本
    if tf is not None:
        print("text:", tuple(tf.shape), float(tf.mean()), float(tf.std()), tf.dtype, tf.device)
    else:
        print("text: None")