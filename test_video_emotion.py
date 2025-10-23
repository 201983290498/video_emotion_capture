#!/usr/bin/env python3
"""
Qwen3-Omni 视频情感分析测试脚本
使用Thinker模型分析视频中的情感表达
"""

import os
import sys
import logging
import torch
from pathlib import Path
import glob

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_video_emotion_analysis():
    """测试Qwen3-Omni对视频的情感分析能力"""
    
    try:
        # 步骤1: 检查环境
        logger.info("步骤1: 检查环境...")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
        
        # 步骤2: 导入模块
        logger.info("步骤2: 导入Qwen3-Omni模块...")
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        logger.info("✓ 成功导入Qwen3-Omni模块")
        
        # 步骤3: 设置路径
        model_path = os.path.expanduser("~/models/Qwen/Qwen3-Omni-30B-A3B-Instruct")
        video_dir = "/data/testmllm/project/verl/video_capture/videos"
        
        # 获取第一个视频文件进行测试
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"在 {video_dir} 中没有找到视频文件")
        
        test_video = video_files[0]  # 使用第一个视频文件
        logger.info(f"测试视频: {test_video}")
        
        # 步骤4: 加载处理器
        logger.info("步骤4: 加载处理器...")
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        logger.info("✓ 处理器加载成功")
        
        # 步骤5: 加载模型
        logger.info("步骤5: 加载Thinker模型到GPU...")
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("✓ 模型加载成功")
        logger.info(f"模型设备: {next(model.parameters()).device}")
        
        # 步骤6: 准备视频情感分析对话
        logger.info("步骤6: 准备视频情感分析...")
        
        conversations = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. You are an expert at analyzing emotions and facial expressions in videos."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": test_video},
                    {"type": "text", "text": "请分析这个视频中人物的情感表达。请详细描述你观察到的面部表情、情绪状态，以及可能的情感变化。请用中文回答。"},
                ],
            },
        ]
        
        # 步骤7: 处理输入
        logger.info("步骤7: 处理多模态输入...")
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        
        logger.info("✓ 输入处理完成")
        logger.info(f"输入形状: {inputs['input_ids'].shape}")
        
        # 步骤8: 生成情感分析结果
        logger.info("步骤8: 开始视频情感分析...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        # 步骤9: 解码结果
        logger.info("步骤9: 解码分析结果...")
        
        # 提取新生成的部分
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[:, input_length:]
        emotion_analysis = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        logger.info("=" * 60)
        logger.info("🎭 视频情感分析结果:")
        logger.info("=" * 60)
        logger.info(f"测试视频: {os.path.basename(test_video)}")
        logger.info(f"分析结果:\n{emotion_analysis}")
        logger.info("=" * 60)
        
        # 步骤10: 测试多个视频（可选）
        if len(video_files) > 1:
            logger.info(f"步骤10: 测试更多视频文件 (共{len(video_files)}个)...")
            
            # 测试前3个视频
            for i, video_file in enumerate(video_files[1:4], 1):
                logger.info(f"分析第{i+1}个视频: {os.path.basename(video_file)}")
                
                conversations_multi = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are an expert at analyzing emotions in videos. Please provide concise emotion analysis."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_file},
                            {"type": "text", "text": "请简要分析这个视频中的主要情感表达，用1~3个情感词语概括。"},
                        ],
                    },
                ]
                
                inputs_multi = processor.apply_chat_template(
                    conversations_multi,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)
                
                with torch.no_grad():
                    outputs_multi = model.generate(
                        **inputs_multi,
                        do_sample=True,
                        max_new_tokens=150,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                
                input_length_multi = inputs_multi['input_ids'].shape[1]
                new_tokens_multi = outputs_multi[:, input_length_multi:]
                result_multi = processor.batch_decode(new_tokens_multi, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                logger.info(f"视频{i+1}分析: {result_multi.strip()}")
        
        logger.info("🎉 视频情感分析测试成功完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_video_emotion_analysis()
    sys.exit(0 if success else 1)