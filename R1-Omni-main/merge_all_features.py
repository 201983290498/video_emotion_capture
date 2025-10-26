import os
import pickle
import numpy as np

def load_pickle_file(file_path):
    """加载pickle文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def merge_features_to_main():
    """
    将所有指定的特征文件合并到all_features.pkl中，并在合并后删除源文件
    需要合并的文件：all_features.pkl、all_features_1.pkl、all_features_2.pkl、all_features_3.pkl、all_features_4.pkl
    """
    # 指定需要处理的文件列表
    required_files = [
        'all_features.pkl',
        'all_features_1.pkl', 
        'all_features_2.pkl',
        'all_features_3.pkl',
        'all_features_4.pkl'
    ]
    
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 检查哪些文件存在
    existing_files = []
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    if not existing_files:
        print("错误：未找到任何需要合并的特征文件")
        return
    
    print(f"找到 {len(existing_files)} 个存在的特征文件：")
    for file in existing_files:
        print(f"  - {os.path.basename(file)}")
    
    # 确保主文件存在
    main_file = os.path.join(current_dir, 'all_features.pkl')
    if main_file not in existing_files:
        print("错误：主文件 all_features.pkl 不存在")
        return
    
    # 加载主文件作为基础数据
    try:
        main_data = load_pickle_file(main_file)
        print("成功加载主文件 all_features.pkl")
    except Exception as e:
        print(f"错误：无法加载主文件 all_features.pkl: {str(e)}")
        return
    
    # 检查文件结构是否符合预期（根据get_pkl.py的分析，应该是嵌套字典结构）
    if not isinstance(main_data, dict):
        print("错误：文件结构不符合预期，应该是字典格式")
        return
    
    # 合并其他文件到主文件
    files_to_delete = []
    for file_path in existing_files:
        # 跳过主文件本身
        if file_path == main_file:
            continue
        
        try:
            print(f"正在处理文件：{os.path.basename(file_path)}")
            # 加载当前文件数据
            current_data = load_pickle_file(file_path)
            
            # 验证数据结构
            if not isinstance(current_data, dict):
                print(f"警告：文件 {os.path.basename(file_path)} 不是字典格式，跳过")
                continue
            
            # 遍历顶层键（如train、test等）
            for top_key in current_data:
                if top_key not in main_data:
                    # 如果主数据中没有该键，直接添加
                    main_data[top_key] = current_data[top_key]
                    print(f"添加新的顶层键：{top_key}")
                else:
                    # 确保两边都是字典结构
                    if isinstance(main_data[top_key], dict) and isinstance(current_data[top_key], dict):
                        # 遍历第二层键（如id、visual、text等）
                        for sub_key in current_data[top_key]:
                            if sub_key not in main_data[top_key]:
                                # 如果主数据中没有该子键，直接添加
                                main_data[top_key][sub_key] = current_data[top_key][sub_key]
                                print(f"  在 {top_key} 下添加新的子键：{sub_key}")
                            else:
                                # 根据数据类型进行合并
                                main_value = main_data[top_key][sub_key]
                                current_value = current_data[top_key][sub_key]
                                
                                if isinstance(main_value, list) and isinstance(current_value, list):
                                    # 合并列表
                                    original_length = len(main_value)
                                    main_value.extend(current_value)
                                    print(f"  合并 {top_key}.{sub_key}: {original_length} + {len(current_value)} = {len(main_value)} 条数据")
                                elif isinstance(main_value, np.ndarray) and isinstance(current_value, np.ndarray):
                                    # 合并numpy数组
                                    try:
                                        main_data[top_key][sub_key] = np.concatenate([main_value, current_value])
                                        print(f"  合并 {top_key}.{sub_key}: 成功合并numpy数组")
                                    except ValueError as e:
                                        print(f"  警告：合并 {top_key}.{sub_key} 的numpy数组时形状不匹配: {str(e)}")
                                else:
                                    print(f"  警告：无法合并 {top_key}.{sub_key}，数据类型不兼容")
                    else:
                        print(f"  警告：{top_key} 的数据类型不兼容，跳过")
            
            # 将处理完成的文件添加到待删除列表
            files_to_delete.append(file_path)
            print(f"文件 {os.path.basename(file_path)} 处理完成，将被删除")
            
        except Exception as e:
            print(f"错误：处理文件 {os.path.basename(file_path)} 时发生异常: {str(e)}")
            continue
    
    # 保存更新后的主文件
    try:
        with open(main_file, 'wb') as f:
            pickle.dump(main_data, f)
        print(f"\n成功保存更新后的主文件: {main_file}")
    except Exception as e:
        print(f"错误：保存更新后的主文件时发生异常: {str(e)}")
        # 如果保存失败，不删除源文件
        print("保存失败，取消删除源文件")
        return
    
    # 删除已合并的文件
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"已删除文件: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"警告：删除文件 {os.path.basename(file_path)} 时发生错误: {str(e)}")
    
    # 打印合并后的统计信息
    print("\n合并后的数据集统计信息：")
    for top_key, top_value in main_data.items():
        if isinstance(top_value, dict):
            print(f"\n{top_key}:")
            for sub_key, sub_value in top_value.items():
                if isinstance(sub_value, list):
                    print(f"  - {sub_key}: {len(sub_value)} 条数据")
                elif isinstance(sub_value, np.ndarray):
                    print(f"  - {sub_key}: numpy数组，形状 {sub_value.shape}")
                else:
                    print(f"  - {sub_key}: 类型 {type(sub_value).__name__}")
        else:
            print(f"- {top_key}: 类型 {type(top_value).__name__}")

if __name__ == "__main__":
    merge_features_to_main()