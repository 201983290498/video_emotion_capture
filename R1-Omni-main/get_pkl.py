import pickle
import sys
import os
import joblib
import numpy as np

file_path = 'E:\\jh\\emotion_analysis\\lddu_mmer\\dataset\\dataset1\\aligned_50.pkl'

print(f"正在读取文件: {file_path}")

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"成功加载数据，类型: {type(data)}")

if isinstance(data, dict):  
    print(f"字典键: {list(data.keys())}")
    print(f"字典长度: {len(data)}")
    if len(data) > 0:
        first_key = list(data.keys())[0]
        print(f"第一个键: {first_key}")
        print(f"第一个键对应的值类型: {type(data[first_key])}")
        
        # 解析train键下的字典信息
        if 'train' in data:
            train_data = data['test']
            if isinstance(train_data, dict):
                print("\ntrain键下的字典信息:")
                print(f"train字典的键: {list(train_data.keys())}")
                print(f"train字典的长度: {len(train_data)}")
                
                # 输出每个键对应的第一个值和维度信息
                print("\ntrain字典各键的第一个值和维度:")
                for key in train_data:
                    value = train_data[key]
                    # 检查值是否可迭代且非空
                    if hasattr(value, '__len__') and len(value) > 0:
                        first_value = value[2]  # 获取第一个元素
                        print(f"- {key} 的第一个值类型: {type(first_value)}")
                        # 打印维度信息
                        if isinstance(first_value, np.ndarray):
                            print(f"- {key} 的第一个值维度: {first_value.shape}")
                        elif hasattr(first_value, '__len__') and not isinstance(first_value, str):
                            try:
                                print(f"- {key} 的第一个值长度: {len(first_value)}")
                                if hasattr(first_value[0], '__len__') and not isinstance(first_value[0], str):
                                    print(f"- {key} 的第一个值内部维度: {len(first_value[0])}")
                            except:
                                pass
                        print(f"- {key} 的第一个值示例: {first_value}")
                    else:
                        print(f"- {key} 的值为空或不可迭代")
            else:
                print("\ntrain键对应的值不是字典类型")
    