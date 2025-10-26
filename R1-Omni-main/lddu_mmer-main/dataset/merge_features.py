# merge_features.py
import pickle
import os
import numpy as np
from collections import defaultdict

def load_pickle_file(file_path):
    """加载pickle文件"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载文件: {file_path}")
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None

def merge_feature_files(file1_path, file2_path, output_path):
    """
    合并两个特征文件
    
    Args:
        file1_path: 第一个特征文件路径
        file2_path: 第二个特征文件路径
        output_path: 合并后的输出文件路径
    """
    # 加载两个文件
    data1 = load_pickle_file(file1_path)
    data2 = load_pickle_file(file2_path)
    
    if data1 is None or data2 is None:
        print("无法加载文件，合并失败")
        return None
    
    # 检查数据结构
    print("\n文件1结构:")
    for mode in ['train', 'valid', 'test']:
        if mode in data1:
            print(f"  {mode}: {len(data1[mode]['id'])} 个样本")
    
    print("\n文件2结构:")
    for mode in ['train', 'valid', 'test']:
        if mode in data2:
            print(f"  {mode}: {len(data2[mode]['id'])} 个样本")
    
    # 创建合并后的数据结构
    merged_data = {
        'train': defaultdict(list),
        'valid': defaultdict(list),
        'test': defaultdict(list)
    }
    
    # 合并数据
    total_samples = 0
    for mode in ['train', 'valid', 'test']:
        if mode in data1 and mode in data2:
            # 合并所有字段
            for key in data1[mode].keys():
                if key in data1[mode] and key in data2[mode]:
                    # 检查是否为张量或列表
                    if isinstance(data1[mode][key], list) and isinstance(data2[mode][key], list):
                        merged_data[mode][key] = data1[mode][key] + data2[mode][key]
                    else:
                        print(f"警告: {mode}.{key} 不是列表类型，跳过合并")
                        merged_data[mode][key] = data1[mode][key]  # 保留第一个文件的值
                elif key in data1[mode]:
                    merged_data[mode][key] = data1[mode][key]
                elif key in data2[mode]:
                    merged_data[mode][key] = data2[mode][key]
            
            sample_count = len(merged_data[mode]['id'])
            total_samples += sample_count
            print(f"合并后 {mode}: {sample_count} 个样本")
        elif mode in data1:
            merged_data[mode] = data1[mode]
            sample_count = len(merged_data[mode]['id'])
            total_samples += sample_count
            print(f"文件1 {mode}: {sample_count} 个样本")
        elif mode in data2:
            merged_data[mode] = data2[mode]
            sample_count = len(merged_data[mode]['id'])
            total_samples += sample_count
            print(f"文件2 {mode}: {sample_count} 个样本")
    
    # 转换为普通字典
    final_merged_data = {}
    for mode in ['train', 'valid', 'test']:
        final_merged_data[mode] = dict(merged_data[mode])
    
    # 保存合并后的文件
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(final_merged_data, f)
        print(f"\n合并完成! 总样本数: {total_samples}")
        print(f"合并后的文件已保存到: {output_path}")
        return final_merged_data
    except Exception as e:
        print(f"保存合并文件时出错: {str(e)}")
        return None

def verify_merged_file(file_path):
    """验证合并后的文件"""
    print(f"\n验证合并文件: {file_path}")
    try:
        data = load_pickle_file(file_path)
        if data is None:
            return False
        
        print("文件结构验证:")
        total_samples = 0
        for mode in ['train', 'valid', 'test']:
            if mode in data:
                mode_data = data[mode]
                sample_count = len(mode_data['id'])
                total_samples += sample_count
                print(f"  {mode}: {sample_count} 个样本")
                
                # 检查关键字段
                for key in ['vision', 'audio', 'text', 'id', 'raw_text']:
                    if key in mode_data:
                        print(f"    {key}: {len(mode_data[key])} 项")
                        # 检查特征维度
                        if key in ['vision', 'audio', 'text'] and len(mode_data[key]) > 0:
                            sample_feature = mode_data[key][0]
                            if sample_feature is not None:
                                print(f"      特征维度: {sample_feature.shape}")
                            else:
                                print(f"      特征: None")
                    else:
                        print(f"    {key}: 缺失")
        
        print(f"\n总计: {total_samples} 个样本")
        return True
        
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False

def check_for_duplicates(file_path):
    """检查重复的样本ID"""
    print(f"\n检查重复样本ID: {file_path}")
    try:
        data = load_pickle_file(file_path)
        if data is None:
            return
        
        duplicate_count = 0
        for mode in ['train', 'valid', 'test']:
            if mode in data and 'id' in data[mode]:
                ids = data[mode]['id']
                unique_ids = set(ids)
                if len(ids) != len(unique_ids):
                    duplicates = len(ids) - len(unique_ids)
                    duplicate_count += duplicates
                    print(f"  {mode}模式发现 {duplicates} 个重复ID")
                else:
                    print(f"  {mode}模式无重复ID")
        
        if duplicate_count > 0:
            print(f"警告: 总共发现 {duplicate_count} 个重复样本ID")
        else:
            print("所有样本ID都是唯一的")
            
    except Exception as e:
        print(f"检查重复ID时出错: {str(e)}")

if __name__ == "__main__":
    # 文件路径
    file1 = "/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/dataset/dataset1/aggregated_features_merged.pkl"
    file2 = "/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/dataset/dataset1/aggregated_features_3.pkl"  
    output = "/data/testmllm/project/video_capture/R1-Omni-main/lddu_mmer-main/dataset/dataset1/aggregated_features.pkl"
    
    print("开始合并特征文件...")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print(f"输出文件: {output}")
    
    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 文件 {file1} 不存在")
        exit(1)
    if not os.path.exists(file2):
        print(f"错误: 文件 {file2} 不存在")
        exit(1)
    
    # 合并文件
    merged_data = merge_feature_files(file1, file2, output)
    
    if merged_data is not None:
        # 验证合并结果
        if verify_merged_file(output):
            print("\n✅ 合并验证成功!")
        else:
            print("\n❌ 合并验证失败!")
        
        # 检查重复样本
        # check_for_duplicates(output)
        
        print(f"\n合并过程完成!")
        print(f"合并后的文件: {output}")
    else:
        print("\n❌ 合并失败!")