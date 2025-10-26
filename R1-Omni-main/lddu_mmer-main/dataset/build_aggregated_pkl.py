import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# 将项目根目录加入路径，便于导入跨目录模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from extract_multimodal_features import load_models_for_device, batch_extract_multimodal_features


def generate_key_id(video_path):
    """根据视频路径生成 key_id: 文件夹名$_$视频文件名末尾数字。
    例：/.../Raw_/-a55Q6RWvTA/3.mp4 -> -a55Q6RWvTA$_$3
    若文件名末尾无数字，则退化为使用去扩展名的文件名。
    """
    parts = video_path.replace('\\', '/').split('/')
    folder_name = parts[-2] if len(parts) >= 2 else 'unknown'
    base = os.path.splitext(parts[-1])[0] if len(parts) >= 1 else 'unknown'
    import re
    m = re.search(r'(\d+)$', base)
    segment = m.group(1) if m else base
    return f"{folder_name}$_${segment}"


def safe_parse_tgt(tgt_val):
    """将 label.csv 中的 tgt 字段解析为列表格式（情感索引或6维值）。
    兼容: 字符串如 '[4, 5, 9]', 列表, 数字, 其他均回退为空列表。
    """
    if tgt_val is None:
        return []
    try:
        # 若是字符串且形似列表，则尝试 eval
        if isinstance(tgt_val, str):
            s = tgt_val.strip()
            if s.startswith('[') and s.endswith(']'):
                parsed = eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                else:
                    return [parsed]
            else:
                # 非列表字符串，直接包一层
                return [s]
        # 若是 numpy 类型，转 list
        if isinstance(tgt_val, np.ndarray):
            return tgt_val.tolist()
        # 若是列表/元组
        if isinstance(tgt_val, (list, tuple)):
            return list(tgt_val)
        # 若是数字
        if isinstance(tgt_val, (int, float)):
            return [tgt_val]
    except Exception:
        return []
    return []


def clamp_frames(arr: np.ndarray, max_frames: int) -> np.ndarray:
    """截断为不超过 max_frames 的长度（按第一维）。"""
    if arr is None:
        return None
    if arr.shape[0] > max_frames:
        return arr[:max_frames]
    return arr


def ensure_text(text_arr: np.ndarray, target_len: int) -> np.ndarray:
    """确保文本特征存在；若为 None 则填充零，形状 (target_len, 768)。"""
    if text_arr is None:
        return np.zeros((target_len, 768), dtype=np.float32)
    return text_arr


def to_numpy_fp32(x):
    """安全转换到 float32 的 numpy 数组，兼容 torch.Tensor / numpy.ndarray / 序列。
    避免 NumPy 2.0 关于 __array__(copy=...) 的弃用警告。
    """
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    try:
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return None


def collect_video_files(videos_root: str):
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    paths = []
    for root, _, files in os.walk(videos_root):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in video_exts:
                paths.append(os.path.join(root, f))
    return paths


def main():
    parser = argparse.ArgumentParser(description='聚合三模态特征与标签，生成用于训练的 pkl')
    parser.add_argument('--videos_root', type=str, default=os.path.join(ROOT_DIR, 'lddu_mmer-main/dataset/dataset1/Raw5'), help='原始视频根目录')
    parser.add_argument('--label_csv', type=str, default=os.path.join(ROOT_DIR, 'lddu_mmer-main/dataset/dataset1/label.csv'), help='标签 CSV 路径')
    parser.add_argument('--output_pkl', type=str, default=os.path.join(ROOT_DIR, 'lddu_mmer-main/dataset/dataset1/aggregated_features_3.pkl'), help='输出 pkl 路径')
    parser.add_argument('--model_path', type=str, default=os.path.join('/data/testmllm/models', 'R1-Omni-0.5B'), help='HumanOmni 模型目录（包含vision_tower、audio_tower等）')
    parser.add_argument('--device', type=str, default=('cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')), help='设备：cuda 或 cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小（用于批量提取特征）')
    parser.add_argument('--max_frames', type=int, default=500, help='序列最大长度（按需截断）')
    args = parser.parse_args()

    # 读取标签 CSV
    if not os.path.exists(args.label_csv):
        raise FileNotFoundError(f'label_csv 不存在: {args.label_csv}')
    label_df = pd.read_csv(args.label_csv)
    # 构建 key_id -> (mode, tgt, raw_text) 映射
    label_map = {}
    for _, row in label_df.iterrows():
        key_id = row.get('key_id')
        # 清理空值与多余空格
        if key_id is None:
            continue
        try:
            import math
            if isinstance(key_id, float) and math.isnan(key_id):
                continue
        except Exception:
            pass
        key_id = str(key_id).strip()
        mode = row.get('mode', 'test')
        tgt = safe_parse_tgt(row.get('tgt'))
        raw_text = row.get('raw_text', '')
        label_map[key_id] = {'mode': mode if mode in ['train', 'valid', 'test'] else 'test', 'tgt': tgt, 'raw_text': raw_text}

    # 收集视频文件
    video_paths = collect_video_files(args.videos_root)
    if not video_paths:
        raise RuntimeError(f'在 {args.videos_root} 未找到视频文件')

    # 加载模型
    models = load_models_for_device(device=args.device, model_path=args.model_path)

    # 批量提取特征
    print(f'开始批量提取多模态特征，共 {len(video_paths)} 个视频，batch_size={args.batch_size}，device={args.device}')
    results = batch_extract_multimodal_features(video_paths, models=models, batch_size=args.batch_size)

    # 初始化结果字典
    result_dict = {
        'train': {
            'raw_text': [], 'audio': [], 'vision': [], 'id': [], 'text': [],
            'annotations': [], 'classification_labels': [], 'regression_label': [], 'tgt': []
        },
        'valid': {
            'raw_text': [], 'audio': [], 'vision': [], 'id': [], 'text': [],
            'annotations': [], 'classification_labels': [], 'regression_label': [], 'tgt': []
        },
        'test': {
            'raw_text': [], 'audio': [], 'vision': [], 'id': [], 'text': [],
            'annotations': [], 'classification_labels': [], 'regression_label': [], 'tgt': []
        }
    }

    # 聚合
    success, skipped = 0, 0
    unmatched_keys = []
    for vp, res in tqdm(zip(video_paths, results), total=len(video_paths), desc='聚合样本'):
        try:
            key_id = generate_key_id(vp)
            lm = label_map.get(key_id)
            if lm is None:
                if len(unmatched_keys) < 10:
                    unmatched_keys.append(key_id)
                skipped += 1
                continue
            split = lm['mode']

            vf = res.get('visual')
            af = res.get('audio')
            tf = res.get('text')

            # 转 numpy 且类型统一为 float32（安全转换以规避 NumPy 2.0 警告）
            vf = to_numpy_fp32(vf)
            af = to_numpy_fp32(af)
            tf = to_numpy_fp32(tf)

            # 替换音频中的 -inf
            if af is not None:
                af[af == -np.inf] = 0

            # 统一截断长度：以各模态最短长度为参考（再与 max_frames 取最小）
            ref_len = None
            for arr in [vf, af, tf]:
                if arr is not None:
                    ref_len = arr.shape[0] if ref_len is None else min(ref_len, arr.shape[0])
            if ref_len is None:
                skipped += 1
                continue
            ref_len = min(ref_len, args.max_frames)

            vf = clamp_frames(vf, ref_len) if vf is not None else np.zeros((ref_len, 896), dtype=np.float32)
            af = clamp_frames(af, ref_len) if af is not None else np.zeros((ref_len, 896), dtype=np.float32)
            tf = ensure_text(clamp_frames(tf, ref_len), ref_len)

            # 写入结果
            result_dict[split]['vision'].append(vf)
            result_dict[split]['audio'].append(af)
            result_dict[split]['text'].append(tf)
            result_dict[split]['id'].append(key_id)
            result_dict[split]['raw_text'].append(lm.get('raw_text', ''))
            result_dict[split]['annotations'].append([])
            result_dict[split]['classification_labels'].append(0.0)
            result_dict[split]['regression_label'].append(0.0)
            result_dict[split]['tgt'].append(lm['tgt'])
            success += 1
        except Exception:
            skipped += 1
            continue

    # 保存
    out_dir = os.path.dirname(args.output_pkl)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(result_dict, f)

    # 打印统计
    print(f'保存聚合 pkl 到: {args.output_pkl}')
    print(f'成功样本: {success}, 跳过样本: {skipped}')
    if unmatched_keys:
        print('未匹配的 key_id 示例（最多10个）:')
        for k in unmatched_keys:
            print('  ', k)
        print(f'label.csv 中可用 key 数量: {len(label_map)}')
    for split in ['train', 'valid', 'test']:
        n = len(result_dict[split]['id'])
        print(f'{split}: {n} 条')
        if n:
            v0 = result_dict[split]['vision'][0]
            a0 = result_dict[split]['audio'][0]
            t0 = result_dict[split]['text'][0]
            print(f'  vision[0]: {tuple(v0.shape)} dtype={v0.dtype}')
            print(f'  audio[0]: {tuple(a0.shape)} dtype={a0.dtype}')
            print(f'  text[0]: {tuple(t0.shape)} dtype={t0.dtype}')


if __name__ == '__main__':
    main()