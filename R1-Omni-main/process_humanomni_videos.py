import sys
import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
import concurrent.futures
import multiprocessing
import json
import glob
import joblib

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_multimodal_features import extract_multimodal_features

# 全局模型缓存变量，用于存储已加载的模型
global_models = None

# 进程本地存储，避免多进程中的重复初始化
class ProcessLocalStorage:
    _instance = None
    
    def __init__(self):
        self.models = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ProcessLocalStorage()
        return cls._instance


def get_global_models(model_path=None, gpu_ids=None):
    # 使用进程本地存储，避免多进程中的重复初始化
    local_storage = ProcessLocalStorage.get_instance() 
    if local_storage.models is None:
        print(f"[{time.strftime('%H:%M:%S')}] 初始化模型...")
        # 首次调用，加载模型并创建缓存 使用一个实际存在的视频文件
        dummy_video_path = "/data/jianghong/lddu_mmer/dataset/dataset1/4.mp4"
        
        # 调用extract_multimodal_features来初始化模型
        features = extract_multimodal_features(
            video_path=dummy_video_path,
            model_path=model_path,
            visual_prompts=["Dummy prompt for model initialization"],
            gpu_ids=gpu_ids
        )
        if features and 'models' in features:
            local_storage.models = features['models']
            print(f"[{time.strftime('%H:%M:%S')}] 模型初始化成功")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 警告：模型初始化返回空结果")
            local_storage.models = {}
    return local_storage.models


def get_video_files(folder_path):
    """获取文件夹中所有视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files


def extract_key_id_from_path(video_path):
    """从视频路径提取key_id"""
    try:
        path_parts = video_path.replace('\\', '/').split('/')
        if 'Raw' in path_parts:
            idx = path_parts.index('Raw')
            if len(path_parts) > idx + 2:
                folder_id = path_parts[idx + 1]
                file_id = os.path.splitext(path_parts[idx + 2])[0]
                key_id = f"{folder_id}$_${file_id}"
                return key_id
        # 如果路径不符合预期格式，返回文件名
        return os.path.basename(video_path)
    except:
        return os.path.basename(video_path)


def standardize_feature_shape(feature, target_shape, feature_name="unknown"):
    """标准化特征形状，确保所有特征具有相同的维度"""
    if feature is None:
        return np.zeros(target_shape, dtype=np.float32)
    
    # 转换为numpy数组
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    
    if not isinstance(feature, np.ndarray):
        return np.zeros(target_shape, dtype=np.float32)
    
    current_shape = feature.shape
    
    # 如果形状已经匹配，直接返回
    if current_shape == target_shape:
        return feature
    
    # 如果维度不匹配，进行调整
    if len(current_shape) != len(target_shape):
        print(f"警告: {feature_name} 特征维度不匹配: 当前 {current_shape}, 目标 {target_shape}")
        return np.zeros(target_shape, dtype=np.float32)
    
    # 调整形状
    result = np.zeros(target_shape, dtype=np.float32)
    
    # 计算每个维度的切片
    slices = []
    for i in range(len(target_shape)):
        dim = min(current_shape[i], target_shape[i])
        slices.append(slice(0, dim))
    
    # 将原始数据复制到目标数组中
    result[tuple(slices)] = feature[tuple(slices)]
    
    return result


def process_single_video(video_file, models, model_path, emotion_visual_prompts, key_id_to_label, gpu_ids=None):
    """处理单个视频并返回处理结果"""
    try:
        # 从视频路径提取key_id
        key_id = extract_key_id_from_path(video_file)
        
        # 使用全局模型缓存提取特征
        features = extract_multimodal_features(
            video_path=video_file,
            models=models,
            model_path=model_path,
            visual_prompts=emotion_visual_prompts,
            gpu_ids=gpu_ids
        )
        
        if features is None:
            return {"status": "error", "key_id": key_id, "video_path": video_file}
        
        if features.get('visual_features') is None:
            return {"status": "error", "key_id": key_id, "video_path": video_file}
        
        # 标准化特征形状
        visual_features = standardize_feature_shape(
            features.get('visual_features'), 
            (500, 896), 
            "visual"
        )
        audio_features = standardize_feature_shape(
            features.get('audio_features'), 
            (500, 896), 
            "audio"
        )
        # text_features = standardize_feature_shape(
        #     features.get('text_features'), 
        #     (1, 1536), 
        #     "text"
        # )
        
        # 更新特征
        features['visual_features'] = visual_features
        features['audio_features'] = audio_features
        # features['text_features'] = text_features
        
        # 确定数据集分割
        split = 'test'
        label_info = None
        
        # 尝试根据key_id查找标签
        if key_id in key_id_to_label:
            label_info = key_id_to_label[key_id]
            mode = label_info.get('mode', '').lower()
            if mode == 'train':
                split = 'train'
            elif mode == 'valid':
                split = 'valid'
        
        # 构建返回结果
        result = {
            "status": "success",
            "key_id": key_id,
            "split": split,
            "video_path": video_file,
            "features": features,
            "label_info": label_info
        }
        
        return result
    except Exception as e:
        sys.stderr.write(f"处理 {video_file} 时出错: {e}\n")
        return {"status": "error", "key_id": "", "video_path": video_file}
    finally:
        # 显式释放GPU内存，避免内存泄漏
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()


def save_dataset_chunk(dataset, chunk_file):
    """保存数据集分块"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
        
        # 使用临时文件，避免写入过程中被中断导致文件损坏
        temp_file = chunk_file + '.tmp'
        
        with open(temp_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # 重命名临时文件为目标文件
        os.rename(temp_file, chunk_file)
        print(f"[{time.strftime('%H:%M:%S')}] 数据集分块已保存至: {chunk_file}")
        return True
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 保存数据集分块失败: {e}")
        return False


def load_dataset_chunks(output_path):
    """加载所有数据集分块，跳过加载失败的分块，返回成功加载的分块列表和统计信息"""
    chunk_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    chunk_pattern = os.path.join(chunk_dir, f"{base_name}_chunk_*.pkl")
    
    chunk_files = sorted(glob.glob(chunk_pattern))
    if not chunk_files:
        return None, None
    
    print(f"[{time.strftime('%H:%M:%S')}] 找到 {len(chunk_files)} 个数据集分块")
    
    successful_chunks = []
    failed_chunks = []
    failed_chunk_files = []
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                # chunk_data = pickle.load(f)
                chunk_data = joblib.load(f)
            
            # 验证分块数据的基本结构
            if not isinstance(chunk_data, dict):
                raise ValueError(f"分块数据不是字典类型: {type(chunk_data)}")
            
            # 检查必要的键是否存在
            required_splits = ['train', 'valid', 'test']
            for split in required_splits:
                if split not in chunk_data:
                    raise ValueError(f"分块缺少必要的分割: {split}")
            
            successful_chunks.append({
                'file': chunk_file,
                'data': chunk_data
            })
            print(f"[{time.strftime('%H:%M:%S')}] 成功加载分块: {os.path.basename(chunk_file)}")
            
        except Exception as e:
            failed_chunks.append(chunk_file)
            failed_chunk_files.append(chunk_file)
            print(f"[{time.strftime('%H:%M:%S')}] 加载分块 {chunk_file} 失败: {e}")
    
    # 输出加载统计信息
    print(f"[{time.strftime('%H:%M:%S')}] 分块加载统计: 成功 {len(successful_chunks)} 个, 失败 {len(failed_chunks)} 个")
    if failed_chunk_files:
        print(f"[{time.strftime('%H:%M:%S')}] 失败的分块文件: {', '.join([os.path.basename(f) for f in failed_chunk_files])}")
    
    stats = {
        'total_chunks': len(chunk_files),
        'successful_chunks': len(successful_chunks),
        'failed_chunks': len(failed_chunks),
        'failed_files': failed_chunk_files
    }
    
    return successful_chunks, stats


def standardize_chunk_features(chunk_data):
    """标准化分块中的特征形状"""
    print(f"[{time.strftime('%H:%M:%S')}] 标准化分块特征形状...")
    
    target_shapes = {
        'vision': (500, 896),
        'audio': (500, 896), 
        'text': (1, 1536)
    }
    
    for split in ['train', 'valid', 'test']:
        if split not in chunk_data:
            continue
            
        for feature_type, target_shape in target_shapes.items():
            if feature_type in chunk_data[split]:
                standardized_features = []
                for i, feature in enumerate(chunk_data[split][feature_type]):
                    try:
                        # 标准化特征形状
                        standardized_feature = standardize_feature_shape(
                            feature, target_shape, f"{split}.{feature_type}[{i}]"
                        )
                        standardized_features.append(standardized_feature)
                    except Exception as e:
                        print(f"警告: 标准化 {split}.{feature_type}[{i}] 失败: {e}")
                        # 使用零数组作为备用
                        standardized_features.append(np.zeros(target_shape, dtype=np.float32))
                
                # 更新特征列表
                chunk_data[split][feature_type] = standardized_features
    
    return chunk_data


def merge_successful_chunks(successful_chunks, output_path):
    """合并成功加载的分块到完整数据集"""
    if not successful_chunks:
        print(f"[{time.strftime('%H:%M:%S')}] 没有成功加载的分块可合并")
        return None
    
    print(f"[{time.strftime('%H:%M:%S')}] 开始合并 {len(successful_chunks)} 个成功加载的分块...")
    
    # 初始化完整数据集
    full_dataset = {
        'train': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
        'valid': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
        'test': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []}
    }
    
    total_samples = 0
    
    # 合并所有成功分块的数据
    for chunk_info in successful_chunks:
        chunk_data = chunk_info['data']
        chunk_file = chunk_info['file']
        
        # 标准化当前分块的特征形状
        chunk_data = standardize_chunk_features(chunk_data)
        
        chunk_samples = 0
        for split in full_dataset:
            if split in chunk_data:
                for key in full_dataset[split]:
                    if key in chunk_data[split]:
                        full_dataset[split][key].extend(chunk_data[split][key])
                # 计算当前分块的样本数
                if 'id' in chunk_data[split]:
                    chunk_samples += len(chunk_data[split]['id'])
        
        total_samples += chunk_samples
        print(f"[{time.strftime('%H:%M:%S')}] 合并分块 {os.path.basename(chunk_file)}: {chunk_samples} 个样本")
    
    print(f"[{time.strftime('%H:%M:%S')}] 总共合并 {total_samples} 个样本")
    
    # 检查是否有成功合并的样本
    if total_samples == 0:
        print(f"[{time.strftime('%H:%M:%S')}] 警告：没有成功合并任何样本数据")
        return None
    
    # 转换列表为numpy数组
    for split in full_dataset:
        for key in full_dataset[split]:
            if key != 'annotations' and key != 'id' and key != 'raw_text' and key != 'tgt':
                try:
                    if full_dataset[split][key]:  # 确保列表不为空
                        # 检查所有特征形状是否一致
                        first_shape = full_dataset[split][key][0].shape
                        all_same_shape = all(feature.shape == first_shape for feature in full_dataset[split][key])
                        
                        if all_same_shape:
                            full_dataset[split][key] = np.array(full_dataset[split][key])
                            print(f"  {split}.{key} 特征形状: {full_dataset[split][key].shape}")
                        else:
                            print(f"  警告: {split}.{key} 特征形状不一致，保持为列表格式")
                            # 记录形状分布用于调试
                            shapes = {}
                            for feature in full_dataset[split][key]:
                                shape_str = str(feature.shape)
                                shapes[shape_str] = shapes.get(shape_str, 0) + 1
                            print(f"    形状分布: {shapes}")
                    else:
                        print(f"  {split}.{key} 为空列表，跳过numpy转换")
                except Exception as e:
                    print(f"  转换 {split}.{key} 为numpy数组失败: {e}")
                    # 保持为列表格式
    
    # 保存合并后的完整数据集
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 使用临时文件确保写入完整性
    temp_output_path = output_path + '.tmp'
    try:
        with open(temp_output_path, 'wb') as f:
            pickle.dump(full_dataset, f)
        
        # 重命名临时文件为目标文件
        os.rename(temp_output_path, output_path)
        print(f"[{time.strftime('%H:%M:%S')}] 合并后的完整数据集已保存至: {output_path}")
        
        return full_dataset
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 保存合并数据集失败: {e}")
        # 清理临时文件
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return None


def save_progress_info(processed_videos, output_path, current_chunk):
    """保存进度信息到JSON文件"""
    progress_file = output_path + '.progress.json'
    progress_data = {
        'processed_videos': list(processed_videos),
        'current_chunk': current_chunk,
        'timestamp': time.time()
    }
    
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        return True
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 保存进度信息失败: {e}")
        return False


def load_progress_info(output_path):
    """从JSON文件加载进度信息"""
    progress_file = output_path + '.progress.json'
    
    if not os.path.exists(progress_file):
        return set(), 0
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        processed_videos = set(progress_data.get('processed_videos', []))
        current_chunk = progress_data.get('current_chunk', 0)
        
        print(f"[{time.strftime('%H:%M:%S')}] 从进度文件恢复: {len(processed_videos)} 个已处理视频, 当前分块: {current_chunk}")
        return processed_videos, current_chunk
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 加载进度信息失败: {e}")
        return set(), 0


def process_videos_and_create_dataset(video_folder, output_path, model_path=None, num_workers=None, gpu_ids=None, batch_size=4):
    """
    处理所有视频并创建LDDU_MMER兼容的数据集，支持断点续传和增量保存
    
    Args:
        video_folder: 视频文件夹路径
        output_path: 输出pkl文件路径
        model_path: 模型权重路径
        num_workers: 并行工作进程数，默认为CPU核心数的一半
        gpu_ids: 指定使用的GPU卡号，如"0"或"0,1"（可选）
        batch_size: 批量处理的视频数量，默认为4
    """
    # 获取所有视频文件
    video_files = get_video_files(video_folder)
    print(f"[{time.strftime('%H:%M:%S')}] 找到 {len(video_files)} 个视频文件")
    
    # 尝试从分块加载现有数据
    successful_chunks, chunk_stats = load_dataset_chunks(output_path)
    
    # 从JSON进度文件加载进度信息
    processed_videos, current_chunk = load_progress_info(output_path)
    
    # 如果从成功分块恢复，提取已处理的视频
    if successful_chunks:
        # 构建key_id到视频路径的映射
        key_id_to_video_path = {}
        for video_path in video_files:
            key_id = extract_key_id_from_path(video_path)
            key_id_to_video_path[key_id] = video_path
        
        # 从成功分块中提取已处理的视频路径
        recovered_count = 0
        for chunk_info in successful_chunks:
            chunk_data = chunk_info['data']
            for split in chunk_data:
                if 'id' in chunk_data[split]:
                    for key_id in chunk_data[split]['id']:
                        if key_id in key_id_to_video_path:
                            video_path = key_id_to_video_path[key_id]
                            processed_videos.add(video_path)
                            recovered_count += 1
        
        print(f"[{time.strftime('%H:%M:%S')}] 从成功分块恢复 {recovered_count} 个已处理视频")
        
        # 立即合并成功分块到新文件，避免重复处理
        merged_dataset = merge_successful_chunks(successful_chunks, output_path)
        if merged_dataset:
            print(f"[{time.strftime('%H:%M:%S')}] 成功分块已合并到新文件")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 警告：成功分块合并失败")
    
    # 筛选出未处理的视频
    unprocessed_videos = [vf for vf in video_files if vf not in processed_videos]
    print(f"[{time.strftime('%H:%M:%S')}] 需要处理 {len(unprocessed_videos)} 个视频，使用batch_size={batch_size}")
    
    if not unprocessed_videos:
        print(f"[{time.strftime('%H:%M:%S')}] 所有视频已处理完成")
        return
    
    # 准备情感相关的视觉提示
    emotion_visual_prompts = [
        "Identify emotions from facial expressions and body language",
        "Analyze the emotional context of the scene",
        "Determine the emotional tone of the video content"
    ]
    
    # 初始化全局模型缓存
    models = get_global_models(model_path, gpu_ids)
    
    # 读取标签文件
    label_csv_path = "/data/jianghong/lddu_mmer/dataset/dataset1/label.csv"
    key_id_to_label = {}
    
    if os.path.exists(label_csv_path):
        label_data = pd.read_csv(label_csv_path, header=0)
        print(f"成功加载label.csv，共包含 {len(label_data)} 行数据")
        
        label_dict = {row['key_id']: row for _, row in label_data.iterrows()}
        
        for key_id, row in label_dict.items():
            mode = row.get('mode', 'test')
            
            tgt_value = row.get('tgt', [0])
            if isinstance(tgt_value, list):
                pass
            elif isinstance(tgt_value, np.ndarray):
                tgt_value = tgt_value.tolist()
            elif isinstance(tgt_value, (int, float, str)):
                tgt_value = [tgt_value]
            else:
                tgt_value = [0]
            
            key_id_to_label[key_id] = {
                'mode': mode,
                'raw_text': row.get('raw_text', ""),
                'id': key_id,
                'classification_labels': row.get('classification_labels', 0.0),
                'regression_labels': row.get('regression_label', 0.0),
                'annotations': row.get('annotations', []),
                'tgt': tgt_value
            }
        print(f"[{time.strftime('%H:%M:%S')}] 成功加载 {len(key_id_to_label)} 个标签")
    
    # 确定并行进程数
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    print(f"[{time.strftime('%H:%M:%S')}] 使用 {num_workers} 个并行进程处理视频")
    
    # 初始化统计计数器
    stats = {
        'total_videos': len(video_files),
        'successfully_processed': len(processed_videos),
        'error_skipped': 0
    }
    
    # 分块大小
    chunk_size = 3000
    current_chunk_data = {
        'train': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
        'valid': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
        'test': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                  'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []}
    }
    
    start_time = time.time()
    
    # 使用线程池并行处理视频
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_video = {}
        
        # 批量处理视频，提高处理效率
        for i in range(0, len(unprocessed_videos), batch_size):
            batch_videos = unprocessed_videos[i:i+batch_size]
            for video_file in batch_videos:
                future_to_video[executor.submit(
                    process_single_video, 
                    video_file, models, model_path, emotion_visual_prompts, key_id_to_label, gpu_ids
                )] = video_file
        
        # 处理完成的任务
        for i, future in enumerate(concurrent.futures.as_completed(future_to_video)):
            video_file = future_to_video[future]
            try:
                result = future.result()
                
                if result["status"] == "success":
                    # 提取特征和信息
                    features = result["features"]
                    key_id = result["key_id"]
                    split = result["split"]
                    label_info = result["label_info"]
                    
                    # 添加多模态特征数据到当前分块
                    # 特征已经在process_single_video中标准化了形状
                    current_chunk_data[split]['vision'].append(features['visual_features'])
                    current_chunk_data[split]['audio'].append(features['audio_features'])
                    # current_chunk_data[split]['text'].append(features['text_features'])
                    
                    # 添加其他字段
                    if label_info:
                        current_chunk_data[split]['raw_text'].append(label_info.get('raw_text', features.get('text_transcription', "")))
                        current_chunk_data[split]['id'].append(label_info.get('id', key_id))
                        current_chunk_data[split]['annotations'].append(label_info.get('annotations', []))
                        current_chunk_data[split]['tgt'].append(label_info.get('tgt', []))
                        
                        try:
                            cls_label = float(label_info.get('classification_labels', 0.0))
                        except (ValueError, TypeError):
                            cls_label = 0.0
                        try:
                            reg_label = float(label_info.get('regression_labels', 0.0))
                        except (ValueError, TypeError):
                            reg_label = 0.0
                        current_chunk_data[split]['classification_labels'].append(cls_label)
                        current_chunk_data[split]['regression_labels'].append(reg_label)
                    else:
                        current_chunk_data[split]['raw_text'].append(features.get('text_transcription', ""))
                        current_chunk_data[split]['id'].append(key_id)
                        current_chunk_data[split]['annotations'].append([])
                        current_chunk_data[split]['classification_labels'].append(0.0)
                        current_chunk_data[split]['regression_labels'].append(0.0)
                        current_chunk_data[split]['tgt'].append([])
                    
                    # 记录已处理的视频（使用完整路径）
                    processed_videos.add(video_file)
                    stats['successfully_processed'] += 1
                    
                    # 每处理chunk_size个视频或最后一个视频时，保存当前分块
                    current_processed_in_chunk = len(current_chunk_data['train']['id']) + len(current_chunk_data['valid']['id']) + len(current_chunk_data['test']['id'])
                    
                    if current_processed_in_chunk >= chunk_size or (i + 1) == len(unprocessed_videos):
                        # 保存当前分块
                        chunk_dir = os.path.dirname(output_path)
                        base_name = os.path.splitext(os.path.basename(output_path))[0]
                        chunk_file = os.path.join(chunk_dir, f"{base_name}_chunk_{current_chunk:04d}.pkl")
                        
                        if save_dataset_chunk(current_chunk_data, chunk_file):
                            # 更新进度信息
                            save_progress_info(processed_videos, output_path, current_chunk)
                            
                            # 重置当前分块数据
                            current_chunk_data = {
                                'train': {'vision': [], 'audio': [], 'text': [],'raw_text': [], 'id': [], 
                                          'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
                                'valid': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                                          'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []},
                                'test': {'vision': [], 'audio': [], 'text': [], 'raw_text': [], 'id': [], 
                                          'annotations': [], 'classification_labels': [], 'regression_labels': [], 'tgt': []}
                            }
                            current_chunk += 1
                
                elif result["status"] == "error":
                    stats['error_skipped'] += 1
                
                # 定期打印进度
                if ((i + 1) % 1000 == 0 or (i + 1) == len(unprocessed_videos)):
                    elapsed_time = time.time() - start_time
                    avg_time_per_video = elapsed_time / (i + 1) if (i + 1) > 0 else 0
                    remaining_time = avg_time_per_video * (len(unprocessed_videos) - (i + 1))
                    
                    print(f"[{time.strftime('%H:%M:%S')}] 已处理 {i+1}/{len(unprocessed_videos)} 个视频，成功 {stats['successfully_processed']} 个，错误 {stats['error_skipped']} 个")
                    print(f"  平均每视频耗时: {avg_time_per_video:.2f}秒, 预计剩余时间: {remaining_time/60:.2f}分钟")
                    print(f"  当前批量大小: {batch_size}")
                    
                    # 定期保存进度信息
                    save_progress_info(processed_videos, output_path, current_chunk)
                    
            except Exception as e:
                stats['error_skipped'] += 1
                sys.stderr.write(f"获取 {video_file} 结果时出错: {e}\n")
    
    # 处理完成后，重新加载所有分块（包括新生成的和原有的），合并成功的分块
    print(f"[{time.strftime('%H:%M:%S')}] 开始合并所有成功分块数据...")
    
    # 重新加载所有分块，只合并成功的
    successful_chunks_final, final_chunk_stats = load_dataset_chunks(output_path)
    
    if successful_chunks_final:
        # 合并所有成功分块到最终数据集
        final_dataset = merge_successful_chunks(successful_chunks_final, output_path)
        
        if final_dataset is None:
            print(f"[{time.strftime('%H:%M:%S')}] 警告：最终数据集合并失败")
            return
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 警告：没有找到任何成功分块")
        return
    
    # 清理临时文件
    progress_files = [
        output_path + '.progress.json',
    ]
    
    # 删除进度文件
    for file_path in progress_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"[{time.strftime('%H:%M:%S')}] 删除临时文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 删除临时文件 {os.path.basename(file_path)} 失败: {e}")
    
    # 保留分块文件，不进行删除操作
    print(f"[{time.strftime('%H:%M:%S')}] 已保留分块文件，未进行删除操作")
    
    print(f"完整数据集已保存至 {output_path}")
    
    # 打印数据集统计信息
    print(f"训练集大小: {len(final_dataset['train']['id'])}")
    print(f"验证集大小: {len(final_dataset['valid']['id'])}")
    print(f"测试集大小: {len(final_dataset['test']['id'])}")
    
    # 打印处理统计信息
    total_time = time.time() - start_time
    print(f"\n=== 处理统计信息 ===")
    print(f"总视频数: {stats['total_videos']}")
    print(f"成功处理: {stats['successfully_processed']}")
    print(f"错误跳过: {stats['error_skipped']}")
    print(f"成功率: {(stats['successfully_processed'] / stats['total_videos'] * 100):.2f}%")
    print(f"总耗时: {total_time/60:.2f}分钟, 平均每视频耗时: {total_time/len(video_files):.2f}秒")
    print(f"并行进程数: {num_workers}")
    print(f"批量大小: {batch_size}")
    
    # 打印分块统计信息
    if final_chunk_stats:
        print(f"\n=== 分块统计信息 ===")
        print(f"总分块数: {final_chunk_stats['total_chunks']}")
        print(f"成功分块: {final_chunk_stats['successful_chunks']}")
        print(f"失败分块: {final_chunk_stats['failed_chunks']}")
        if final_chunk_stats['failed_files']:
            print(f"失败分块文件: {', '.join([os.path.basename(f) for f in final_chunk_stats['failed_files']])}")


def main():
    parser = argparse.ArgumentParser(description="处理HumanOmni视频特征并创建LDDU_MMER兼容的数据集")
    parser.add_argument("--video_folder", type=str, 
                        default="/data/jianghong/lddu_mmer/dataset/dataset1/Raw", 
                        help="包含视频文件的文件夹")
    parser.add_argument("--output_path", type=str, 
                        default="/data/jianghong/lddu_mmer/dataset/dataset1/features_full.pkl", 
                        help="输出pkl文件的路径")
    parser.add_argument("--model_path", type=str, default=None, help="HumanOmni模型权重路径")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="并行处理的进程数，默认为CPU核心数的一半")
    parser.add_argument("--gpu_ids", type=str, default=None, 
                        help="指定使用的GPU卡号，如'0'或'0,1'，不指定则自动选择")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="批量处理的视频数量，默认为4")
    
    args = parser.parse_args()
    
    # 如果指定了GPU设备，在初始化前设置环境变量
    if args.gpu_ids:
        print(f"设置GPU设备: {args.gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    process_videos_and_create_dataset(
        args.video_folder, 
        args.output_path, 
        args.model_path,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()