# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import json
import random
import time
import pickle
import csv


"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes

label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (500, 768)
visual: (500, 896)
audio: (500, 896)    
"""

emotion_dict = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}
class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, max_frames=500):
        self.data_path = data_path
        self.data_type = data_type
        self.max_frames = max_frames  # 存储max_frames参数
        # 新增：目录索引式按需加载支持
        self.use_indexed = os.path.isdir(self.data_path)
        if self.use_indexed:
            self.records, self._label_map = self._build_index(self.data_path, self.data_type)
            # 兼容旧属性引用
            self.visual, self.audio, self.text, self.labels = None, None, None, None
        else:
            self.visual, self.audio, \
                self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        if self.data_path.endswith("pt"):
            data = torch.load(self.data_path)
        else:
            data = pickle.load(open(self.data_path, 'rb'))
        data = data[data_type]
        if 'src-visual' in data.keys():
            visual = data['src-visual']
        else:
            visual = data['vision']
        if 'src-audio' in data.keys():
            audio = data['src-audio']
        else:
            audio = data['audio']
        if 'src-text' in data.keys():
            text = data['src-text']
        else:
            text = data['text']
        labels = data['tgt']      
        return visual, audio, text, labels
    
    # 新增：构建索引（扫描目录中的npz特征并加载标签映射）
    def _build_index(self, base_path, split):
        # 优先使用 split 子目录，否则使用 base
        split_dir = os.path.join(base_path, split)
        search_dir = split_dir if os.path.isdir(split_dir) else base_path
        npz_files = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))
        # 读取可能存在的label.csv
        label_map = {}
        for cand in [os.path.join(base_path, 'label.csv'), os.path.join(os.path.dirname(base_path), 'label.csv')]:
            if os.path.exists(cand):
                try:
                    with open(cand, 'r', encoding='utf-8') as cf:
                        reader = csv.reader(cf)
                        header = next(reader, None)
                        for row in reader:
                            if not row:
                                continue
                            vid = row[0]
                            vals = row[1:]
                            if len(vals) == 6:
                                label_map[vid] = [float(x) for x in vals]
                            elif len(vals) == 1:
                                label_map[vid] = [vals[0]]
                except Exception:
                    pass
        records = []
        for p in npz_files:
            try:
                with np.load(p, mmap_mode='r') as z:
                    # 解析视频ID
                    vid = None
                    if 'video_id' in z:
                        v = z['video_id']
                        try:
                            vid = v.item() if hasattr(v, 'item') else str(v)
                        except Exception:
                            vid = str(v)
                    else:
                        bn = os.path.basename(p)
                        vid = os.path.splitext(bn)[0].replace('_features','')
                    # 标签来源优先级：npz内置 -> csv映射 -> 全0
                    lbl = None
                    if 'labels' in z:
                        lbl = z['labels']
                        lbl = lbl.tolist() if hasattr(lbl, 'tolist') else lbl
                    elif 'classification_labels' in z:
                        lbl = z['classification_labels']
                        lbl = lbl.tolist() if hasattr(lbl, 'tolist') else lbl
                    elif vid in label_map:
                        lbl = label_map[vid]
                    else:
                        lbl = [0]*6
                records.append({'file': p, 'id': vid, 'label': lbl})
            except Exception:
                # 跳过异常文件
                continue
        if not records:
            raise RuntimeError(f"未在目录 {search_dir} 中找到 {split} 集的 .npz 特征文件；请检查目录结构或生成索引。")
        return records, label_map

    def _parse_label(self, label_list):
        label = np.zeros(6, dtype=np.float32)
        try:
            # 6元素直接赋值（统一为二值 one-hot：>0 视为正类）
            if isinstance(label_list, (list, tuple)) and len(label_list) == 6 and all(isinstance(x, (int, float)) for x in label_list):
                for i in range(6):
                    v = float(label_list[i])
                    label[i] = 1.0 if v > 0 else 0.0
            # 字符串列表，如 ['[2, 4, 9, 3]']
            elif isinstance(label_list, (list, tuple)) and len(label_list) == 1 and isinstance(label_list[0], str):
                str_content = label_list[0].strip("[]'\"")
                nums = [float(num.strip()) for num in str_content.split(',') if num.strip()]
                for num in nums:
                    if num in emotion_dict:
                        label[emotion_dict[num]] = 1.0
            else:
                # 旧格式：直接遍历整数/浮点的情感索引
                for emo in label_list:
                    if isinstance(emo, (int, float)) and emo in emotion_dict:
                        label[emotion_dict[emo]] = 1.0
        except Exception:
            label = np.zeros(6, dtype=np.float32)
        return label

    def _load_sample_from_file(self, rec):
        with np.load(rec['file'], mmap_mode='r') as z:
            # 取键的兼容性
            visual = z['visual'] if 'visual' in z else (z['vision'] if 'vision' in z else None)
            audio = z['audio'] if 'audio' in z else None
            text = z['text'] if 'text' in z else None
            if visual is None or audio is None:
                raise RuntimeError(f"文件{rec['file']}缺少必要键(visual/audio)")
            # 转换为float32
            visual = np.array(visual, dtype=np.float32)
            audio = np.array(audio, dtype=np.float32)
            if text is None:
                text = np.zeros((min(self.max_frames, visual.shape[0]), 768), dtype=np.float32)
            else:
                text = np.array(text, dtype=np.float32)
        # 处理-Inf
        audio[audio == -np.inf] = 0
        # 截断到max_frames
        if visual.shape[0] > self.max_frames:
            visual = visual[:self.max_frames]
        if audio.shape[0] > self.max_frames:
            audio = audio[:self.max_frames]
        if text.shape[0] > self.max_frames:
            text = text[:self.max_frames]
        # 掩码
        text_mask = np.ones(text.shape[0], dtype=np.int64)
        visual_mask = np.ones(visual.shape[0], dtype=np.int64)
        audio_mask = np.ones(audio.shape[0], dtype=np.int64)
        # 标签解析
        label = self._parse_label(rec['label'])
        return text, text_mask, visual, visual_mask, audio, audio_mask, label
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        # 使用类属性max_frames来控制视觉特征长度，默认为500
        max_frames = getattr(self, 'max_frames', 500)
        
        # 处理视觉特征长度
        if visual.shape[0] > max_frames:
            # 截断过长的特征
            visual = visual[:max_frames]
        
        # 创建与视觉特征长度匹配的掩码
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        
        # 添加详细的调试信息，打印标签列表的内容和类型
        print(f"标签列表类型: {type(label_list)}, 内容: {label_list}, 长度: {len(label_list) if hasattr(label_list, '__len__') else '无'}")
        
        # 尝试处理不同格式的标签
        try:
            # 检查是否是6元素的列表（直接对应6种情感）
            if isinstance(label_list, list) and len(label_list) == 6:
                # 直接使用这些值作为标签
                for i in range(6):
                    label[i] = float(label_list[i])
                print(f"处理为6元素列表: {label}")
            # 尝试处理字符串列表格式的标签 (如 ['[2, 4, 9, 3]'])
            elif isinstance(label_list, list) and len(label_list) == 1 and isinstance(label_list[0], str):
                # 尝试解析字符串中的数字列表
                try:
                    # 移除字符串两端的括号和引号
                    str_content = label_list[0].strip("[]'\"")
                    # 分割字符串获取数字
                    nums = [float(num.strip()) for num in str_content.split(',')]
                    # 将数字映射到对应的情感标签位置
                    for num in nums:
                        if num in emotion_dict:
                            label[emotion_dict[num]] = 1
                    print(f"处理为字符串列表格式: {label}")
                except Exception as e:
                    print(f"解析字符串标签出错: {e}")
            # 尝试旧的处理方式作为备选
            else:
                filter_label = label_list[1:-1] if len(label_list) > 2 else label_list
                for emo in filter_label:
                    if isinstance(emo, (int, float)) and emo in emotion_dict:
                        label[emotion_dict[emo]] = 1
                print(f"使用旧的处理方式: {label}")
        except Exception as e:
            # 如果出现错误，记录并返回全0标签
            print(f"处理标签时出错: {e}, 标签内容: {label_list}")
            label = np.zeros(6, dtype=np.float32)
        
        # 添加调试信息，显示最终的标签
        print(f"最终标签: {label}, 非零元素数量: {np.count_nonzero(label)}")
        
        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    def __len__(self):
        # 新增：按需加载时返回索引长度
        if getattr(self, 'use_indexed', False):
            return len(self.records)
        return len(self.labels)
    
    def __getitem__(self, index):
        # 新增：按记录从文件加载样本
        if getattr(self, 'use_indexed', False):
            text, text_mask, visual, visual_mask, audio, audio_mask, label = self._load_sample_from_file(self.records[index])
            return text, text_mask, visual, visual_mask, \
                audio, audio_mask, label, index
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label, index


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, max_frames=500):
        self.data_path = data_path
        self.data_type = data_type
        self.max_frames = max_frames  # 存储max_frames参数
        # 新增：目录索引式按需加载支持
        self.use_indexed = os.path.isdir(self.data_path)
        if self.use_indexed:
            self.records, self._label_map = self._build_index(self.data_path, self.data_type)
            self.visual, self.audio, self.text, self.labels = None, None, None, None
        else:
            self.visual, self.audio, \
                self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        visual = data['vision']
        audio = data['audio']
        text = data['text']
        audio = np.array(audio)
        labels = data['tgt']     
        return visual, audio, text, labels

    # 复用与Aligned一致的索引构建/解析逻辑
    _build_index = AlignedMoseiDataset._build_index
    _parse_label = AlignedMoseiDataset._parse_label
    _load_sample_from_file = AlignedMoseiDataset._load_sample_from_file
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        # 使用类属性max_frames来控制视觉特征长度，默认为500
        max_frames = getattr(self, 'max_frames', 500)
        
        # 处理视觉特征长度
        if visual.shape[0] > max_frames:
            # 截断过长的特征
            visual = visual[:max_frames]
        
        # 创建与视觉特征长度匹配的掩码
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        # 统一走解析函数，确保输出为二值 one-hot
        return self._parse_label(self.labels[index])

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    def __len__(self):
        # 新增：按需加载时返回索引长度
        if getattr(self, 'use_indexed', False):
            return len(self.records)
        return len(self.labels)
    
    def __getitem__(self, index):
        # 新增：按记录从文件加载样本
        if getattr(self, 'use_indexed', False):
            text, text_mask, visual, visual_mask, audio, audio_mask, label = self._load_sample_from_file(self.records[index])
            return text, text_mask, visual, visual_mask, \
                audio, audio_mask, label, index
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label, index