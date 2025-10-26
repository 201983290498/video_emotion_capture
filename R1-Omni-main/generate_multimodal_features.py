import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import time
import argparse
import os
from tqdm import tqdm
import logging

# 配置日志
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 设置环境变量以禁用wandb
os.environ['WANDB_MODE'] = 'dryrun'
os.environ['WANDB_SILENT'] = 'true'

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从extract_features.py导入提取特征的函数
from extract_features import extract_features

# 设置环境变量
def setup_environment():
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用tokenizers并行处理以避免死锁
    # 不再强制设置CUDA_VISIBLE_DEVICES，避免与系统冲突
    # 让extract_features.py中的get_device函数自动处理设备选择
    logger.info("环境设置完成")

def generate_key_id(video_path):
    """从视频路径生成key_id，格式为：文件夹名$_$视频名"""
    # 提取文件夹名和视频名
    parts = video_path.split(os.sep)
    folder_name = parts[-2]  # _1nvuNk7EFY
    video_name = os.path.splitext(parts[-1])[0]  # 5
    key_id = f"{folder_name}$_${video_name}"
    return key_id, folder_name

def main():
    setup_environment()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成多模态特征')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--video_path', type=str, help='单个视频路径')
    parser.add_argument('--device_index', type=int, default=0, help='指定CUDA设备索引，默认为0')
    args = parser.parse_args()
    
    # 设置路径
    if os.path.exists('/data/jianghong'):
        # 在华为昇腾910B环境中运行
        base_dir = '/data/jianghong'
        default_model_path = os.path.join(base_dir, 'R1-Omni', 'models', 'R1-Omni-0.5B')
    else:
        # 在本地环境中测试
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(base_dir, 'models', 'R1-Omni-0.5B')
        
    # 如果没有提供模型路径，使用默认路径
    model_path = args.model_path if args.model_path else default_model_path
    
    # 如果提供了单个视频路径，直接处理该视频
    if args.video_path:
        print(f"正在处理单个视频: {args.video_path}")
        # 提取特征
        # 首先检查CUDA状态
        cuda_available = torch.cuda.is_available()
        num_devices = torch.cuda.device_count() if cuda_available else 0
        logger.info(f"系统CUDA状态 - 可用: {cuda_available}, 设备数量: {num_devices}")
        
        # 根据系统状态自动选择最佳设备
        device_type = 'cuda' if cuda_available and num_devices > 0 else 'cpu'
        logger.info(f"选择的设备类型: {device_type}")
        
        # 创建ctx参数以支持CUDA设备和内存优化
        ctx = {
            'device_type': device_type,  # 根据系统状态自动选择设备类型
            'device_index': args.device_index,  # 添加设备索引参数
            'force_cpu_asr': device_type == 'cpu',  # 如果是CPU，强制在CPU上运行ASR
            'reduce_memory_usage': True,  # 启用内存优化
            'max_split_size_mb': 32  # 设置CUDA内存分割大小为32MB
        }
        logger.info(f"使用的上下文参数: {ctx}")
        
        # 提取特征
        try:
            features = extract_features(model_path, args.video_path, ctx=ctx)
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回空特征字典或默认值
            features = {
                'visual_features': np.zeros((1, 768)),
                'audio_features': np.zeros((1, 768)),
                'text_features': np.zeros((1, 768))
            }
        
        # 生成key_id
        key_id, folder_name = generate_key_id(args.video_path)
        
        # 准备结果字典
        result_dict = {
            'test': {
                'raw_text': ['Unknown'],  # 默认值
                'audio': [features['audio_features']],
                'vision': [features['visual_features']],
                'id': [key_id],
                'text': [features['text_features']],
                'annotations': ['Unknown'],
                'classification_labels': [0.0],
                'regression_label': [0.0],
                'tgt': [[0]]
            }
        }
        
        # 保存结果
        output_path = os.path.join(base_dir, 'lddu_mmer/dataset', f'{os.path.basename(args.video_path)}_features.pkl')
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        with open(output_path, 'wb') as f:
            pickle.dump(result_dict, f)
        
        print(f"特征已保存到: {output_path}")
        print(f"视觉特征形状: {features['visual_features'].shape}")
        print(f"音频特征形状: {features['audio_features'].shape}")
        print(f"文本特征形状: {features['text_features'].shape}")
        return
    
    # 默认处理方式
    video_dir = os.path.join(base_dir, 'lddu_mmer/dataset', 'dataset1/Raw')
    label_test_path = os.path.join(base_dir, 'lddu_mmer/dataset', 'dataset1/label.csv')
    output_path = os.path.join(base_dir, 'lddu_mmer/dataset', 'features.pkl')
    # 断点续传记录文件
    processed_log_path = os.path.join(base_dir, 'lddu_mmer/dataset', 'processed_videos.json')
    
    print(f"基础目录: {base_dir}")
    print(f"模型路径: {model_path}")
    print(f"视频目录: {video_dir}")
    print(f"标签文件: {label_test_path}")
    print(f"输出文件: {output_path}")
    
    # 加载label.csv
    print("正在加载label.csv文件...")
    label_df = pd.read_csv(label_test_path)
    print(f"成功加载label.csv，共包含 {len(label_df)} 行数据")
    
    # 创建label字典，key为key_id，value为整行数据
    label_dict = {row['key_id']: row for _, row in label_df.iterrows()}
    
    # 检查是否存在已保存的中间结果
    if os.path.exists(output_path + '.tmp'):
        try:
            print("发现中间结果文件，正在加载...")
            with open(output_path + '.tmp', 'rb') as f:
                result_dict = pickle.load(f)
            print("中间结果加载成功！")
        except Exception as e:
            print(f"加载中间结果失败: {e}")
            # 初始化结果字典，按照mode划分（包含tgt字段以兼容UnAlignedMoseiDataset）
            result_dict = {
                'train': {
                    'raw_text': [],
                    'audio': [],
                    'vision': [],
                    'id': [],
                    'text': [],
                    'annotations': [],
                    'classification_labels': [],
                    'regression_label': [],
                    'tgt': []  
                },
                'valid': {
                    'raw_text': [],
                    'audio': [],
                    'vision': [],
                    'id': [],
                    'text': [],
                    'annotations': [],
                    'classification_labels': [],
                    'regression_label': [],
                    'tgt': []  
                },
                'test': {
                    'raw_text': [],
                    'audio': [],
                    'vision': [],
                    'id': [],
                    'text': [],
                    'annotations': [],
                    'classification_labels': [],
                    'regression_label': [],
                    'tgt': []  
                }
            }
    else:
        # 初始化结果字典，按照mode划分（包含tgt字段以兼容UnAlignedMoseiDataset）
        result_dict = {
            'train': {
                'raw_text': [],
                'audio': [],
                'vision': [],
                'id': [],
                'text': [],
                'annotations': [],
                'classification_labels': [],
                'regression_label': [],
                'tgt': []  
            },
            'valid': {
                'raw_text': [],
                'audio': [],
                'vision': [],
                'id': [],
                'text': [],
                'annotations': [],
                'classification_labels': [],
                'regression_label': [],
                'tgt': []  
            },
            'test': {
                'raw_text': [],
                'audio': [],
                'vision': [],
                'id': [],
                'text': [],
                'annotations': [],
                'classification_labels': [],
                'regression_label': [],
                'tgt': []  
            }
        }
    
    # 加载已处理视频记录
    processed_videos = set()
    if os.path.exists(processed_log_path):
        try:
            print("发现已处理视频记录，正在加载...")
            with open(processed_log_path, 'r', encoding='utf-8') as f:
                processed_videos = set(json.load(f))
            print(f"已加载{len(processed_videos)}个已处理视频记录")
        except Exception as e:
            print(f"加载已处理视频记录失败: {e}")
            processed_videos = set()
    
    # 遍历所有视频文件
    print("开始遍历视频文件并提取特征...")
    processed_count = 0
    found_count = 0
    skipped_count = 0
    error_count = 0
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(video_dir):
        # 只处理一级子文件夹
        if root == video_dir:
            for subdir in tqdm(dirs, desc="处理文件夹"):
                subdir_path = os.path.join(root, subdir)
                
                # 遍历子文件夹中的所有视频文件
                for file in os.listdir(subdir_path):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(subdir_path, file)
                        processed_count += 1
                        
                        try:
                            # 生成key_id
                            key_id, folder_name = generate_key_id(video_path)
                            
                            # 检查该视频是否已经处理过
                            if video_path in processed_videos:
                                print(f"跳过已处理的视频: {video_path}")
                                skipped_count += 1
                                continue
                            
                            # 检查key_id是否在label.csv中
                            if key_id not in label_dict:
                                print(f"警告: key_id {key_id} 在label.csv中未找到")
                                continue
                            
                            found_count += 1
                            
                            # 获取对应的label数据
                            label_row = label_dict[key_id]
                            mode = label_row['mode']
                            
                            # 确保mode是有效的
                            if mode not in result_dict:
                                print(f"警告: 无效的mode值 {mode}，使用'test'作为默认值")
                                mode = 'test'
                            
                            print(f"正在处理视频: {video_path}")
                            print(f"key_id: {key_id}, mode: {mode}")
                            
                            # 安全地从label.csv中提取标签信息，即使键不存在也不会抛出错误
                            regression_label = label_row.get('regression_label', 0.0)
                            classification_labels = label_row.get('classification_labels', 0.0)
                            annotations = label_row.get('annotations', 'Unknown')
                            # 从label_row获取raw_text列数据
                            raw_text = label_row.get('raw_text', 'Unknown')
                            
                            # 使用extract_features函数提取特征
                            start_time = time.time()
                            # 首先检查CUDA状态
                            cuda_available = torch.cuda.is_available()
                            num_devices = torch.cuda.device_count() if cuda_available else 0
                            logger.info(f"系统CUDA状态 - 可用: {cuda_available}, 设备数量: {num_devices}")
                            
                            # 根据系统状态自动选择最佳设备
                            device_type = 'cuda' if cuda_available and num_devices > 0 else 'cpu'
                            logger.info(f"选择的设备类型: {device_type}")
                            
                            # 创建ctx参数以支持CUDA设备和内存优化
                            ctx = {
                                'device_type': device_type,  # 根据系统状态自动选择设备类型
                                'device_index': args.device_index,  # 添加设备索引参数
                                'force_cpu_asr': device_type == 'cpu',  # 如果是CPU，强制在CPU上运行ASR
                                'reduce_memory_usage': True,  # 启用内存优化
                                'max_split_size_mb': 32  # 设置CUDA内存分割大小为32MB
                            }
                            features = extract_features(model_path, video_path, ctx=ctx)
                            end_time = time.time()
                            
                            # 提取多模态特征
                            audio_features = features['audio_features']
                            visual_features = features['visual_features']
                            text_features = features['text_features']
                            
                            # 音频转录文本已从label.csv中获取
                            # raw_text = ''
                            
                            # 获取id（使用key_id作为id）
                            id_value = key_id
                            
                            # 将数据添加到对应mode的列表中
                            result_dict[mode]['raw_text'].append(raw_text)
                            result_dict[mode]['audio'].append(audio_features)
                            result_dict[mode]['vision'].append(visual_features)
                            result_dict[mode]['id'].append(id_value)
                            result_dict[mode]['text'].append(text_features)
                            result_dict[mode]['annotations'].append(annotations)
                            result_dict[mode]['classification_labels'].append(classification_labels)
                            result_dict[mode]['regression_label'].append(regression_label)
                            
                            # 直接从label.csv中获取tgt字段的值
                            # 安全地访问tgt键，即使键不存在也不会抛出错误
                            tgt_value = label_row.get('tgt', [0])
                            # 确保tgt是列表格式
                            if isinstance(tgt_value, list):
                                # 如果已经是列表，直接使用
                                pass  # 不需要额外处理
                            elif isinstance(tgt_value, np.ndarray):
                                # 如果是数组，转换为列表
                                tgt_value = tgt_value.tolist()
                            elif isinstance(tgt_value, (int, float, str)):
                                # 如果是单个数值或字符串，包装成列表
                                tgt_value = [tgt_value]
                            else:
                                # 其他情况，使用[0]作为默认值
                                tgt_value = [0]
                            result_dict[mode]['tgt'].append(tgt_value)
                            
                            print(f"成功提取特征，耗时: {end_time - start_time:.2f}秒")
                            print(f"已处理: {processed_count}, 找到匹配: {found_count}, 跳过: {skipped_count}, 错误: {error_count}")
                            
                            # 将已处理的视频路径添加到记录中
                            processed_videos.add(video_path)
                            
                            # 每处1000个视频更新一次已处理记录和保存中间结果
                            if len(processed_videos) % 1000 == 0:
                                print("更新已处理记录和保存中间结果...")
                                # 保存已处理视频记录
                                with open(processed_log_path, 'w', encoding='utf-8') as f:
                                    json.dump(list(processed_videos), f, ensure_ascii=False, indent=2)
                                # 保存中间结果
                                with open(output_path + '.tmp', 'wb') as f:
                                    pickle.dump(result_dict, f)
                            
                        except Exception as e:
                            print(f"处理视频 {video_path} 时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            error_count += 1
                            continue
    
    # 确保所有列表的长度一致
    for mode in result_dict:
        data = result_dict[mode]
        lengths = {k: len(v) for k, v in data.items()}
        print(f"{mode}模式下各字段长度: {lengths}")
    
    # 更新已处理记录和保存最终结果
    print("更新已处理记录...")
    with open(processed_log_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed_videos), f, ensure_ascii=False, indent=2)
    
    print("保存最终结果...")
    with open(output_path, 'wb') as f:
        pickle.dump(result_dict, f)
    
    print("特征提取和保存完成！")
    print(f"总处理视频数: {processed_count}")
    print(f"找到匹配的视频数: {found_count}")
    print(f"跳过已处理的视频数: {skipped_count}")
    print(f"处理错误的视频数: {error_count}")
    print(f"结果文件路径: {output_path}")
    print(f"已处理视频记录路径: {processed_log_path}")
    
    # 打印各mode的样本数量
    for mode in result_dict:
        print(f"{mode}模式样本数: {len(result_dict[mode]['id'])}")

if __name__ == "__main__":
    main()