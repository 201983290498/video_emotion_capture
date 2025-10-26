#!/usr/bin/env python3
"""
情感识别推理速度测试脚本
支持LDDU和Qwen模型的性能基准测试
"""

import os
import sys
import time
import json
import statistics
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.lddu_service import LDDUService
from modules.qwen_service import QwenModelService
from config import Config


class EmotionSpeedTester:
    """情感识别速度测试器"""
    
    def __init__(self):
        self.config = Config()
        self.lddu_service = None
        self.qwen_service = None
        self.test_results = {
            'lddu': [],
            'qwen': []
        }
        self.logger: Optional[logging.Logger] = None
        self.log_file: Optional[str] = None
    
    def set_log_file(self, log_file: str):
        """设置日志文件并初始化记录器"""
        self.log_file = log_file
        logger = logging.getLogger('EmotionSpeedTester')
        logger.setLevel(logging.INFO)
        # 防止重复添加处理器
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_file) for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        self.logger = logger
        self._log(f"日志文件设置为: {log_file}")
    
    def _log(self, message: str):
        """同时输出到控制台和日志文件"""
        print(message)
        if self.logger:
            self.logger.info(message)
        
    def initialize_services(self, test_lddu: bool = True, test_qwen: bool = True):
        """初始化测试服务"""
        self._log("正在初始化测试服务...")
        
        if test_lddu:
            try:
                self._log("初始化LDDU服务...")
                self.lddu_service = LDDUService()
                loaded = self.lddu_service.load_model(
                    humanomni_model_path=self.config.HUMANOMNI_MODEL_PATH,
                    model_dir=self.config.LDDU_MODEL_DIR,
                    init_model_path=self.config.LDDU_INIT_CHECKPOINT,
                    emotion_labels=self.config.EMOTION_LABELS
                )
                if loaded and self.lddu_service.is_loaded:
                    self._log("✓ LDDU服务初始化并加载模型成功")
                else:
                    self._log(f"✗ LDDU服务初始化失败: {getattr(self.lddu_service, 'load_error', '未知错误')}")
                    self.lddu_service = None
            except Exception as e:
                self._log(f"✗ LDDU服务初始化异常: {e}")
                self.lddu_service = None
        
        if test_qwen:
            try:
                self._log("初始化Qwen服务...")
                self.qwen_service = QwenModelService()
                loaded_q = self.qwen_service.load_model(
                    model_path=self.config.QWEN_MODEL_PATH,
                    device="auto",
                    max_gpus=self.config.MAX_GPUS
                )
                if loaded_q and self.qwen_service.is_model_ready():
                    self._log("✓ Qwen服务初始化并加载模型成功")
                else:
                    status = self.qwen_service.get_status() if self.qwen_service else {}
                    self._log(f"✗ Qwen服务初始化失败: {status.get('load_error') or '模型未就绪'}")
                    self.qwen_service = None
            except Exception as e:
                self._log(f"✗ Qwen服务初始化异常: {e}")
                self.qwen_service = None
    
    def get_test_videos(self, test_dir: str = "videos") -> List[str]:
        """获取测试视频文件列表"""
        video_dir = Path(test_dir)
        if not video_dir.exists():
            self._log(f"测试视频目录不存在: {test_dir}")
            return []
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        
        for file_path in video_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(str(file_path))
        
        return sorted(video_files)
    
    def test_lddu_speed(self, video_path: str, rounds: int = 3) -> Dict[str, Any]:
        """测试LDDU模型推理速度"""
        if not self.lddu_service or not self.lddu_service.is_loaded:
            return {'error': 'LDDU服务不可用'}
        
        times = []
        results = []
        
        for i in range(rounds):
            try:
                start_time = time.time()
                result = self.lddu_service.analyze_video_emotion(video_path)
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                results.append(result)
                
                self._log(f"  LDDU轮次 {i+1}/{rounds}: {inference_time:.3f}s")
            
            except Exception as e:
                self._log(f"  LDDU轮次 {i+1}/{rounds} 失败: {e}")
                continue
        
        if not times:
            return {'error': '所有测试轮次都失败'}
        
        return {
            'model': 'LDDU',
            'video_path': video_path,
            'rounds': len(times),
            'times': times,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'results': results
        }
    
    def test_qwen_speed(self, video_path: str, rounds: int = 3) -> Dict[str, Any]:
        """测试Qwen模型推理速度"""
        if not self.qwen_service or not self.qwen_service.is_model_ready():
            return {'error': 'Qwen服务不可用'}
        
        times = []
        results = []
        
        for i in range(rounds):
            try:
                start_time = time.time()
                result = self.qwen_service.analyze_video_emotion(video_path)
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                results.append(result)
                
                self._log(f"  Qwen轮次 {i+1}/{rounds}: {inference_time:.3f}s")
                
            except Exception as e:
                self._log(f"  Qwen轮次 {i+1}/{rounds} 失败: {e}")
                continue
        
        if not times:
            return {'error': '所有测试轮次都失败'}
        
        return {
            'model': 'Qwen',
            'video_path': video_path,
            'rounds': len(times),
            'times': times,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'results': results
        }
    
    def run_benchmark(self, video_files: List[str], rounds: int = 3, 
                     test_lddu: bool = True, test_qwen: bool = True):
        """运行基准测试"""
        self._log(f"\n开始基准测试 - 视频数量: {len(video_files)}, 每个视频测试轮次: {rounds}")
        self._log("=" * 60)
        
        for i, video_path in enumerate(video_files, 1):
            video_name = Path(video_path).name
            self._log(f"\n[{i}/{len(video_files)}] 测试视频: {video_name}")
            
            # 测试LDDU
            if test_lddu and self.lddu_service:
                self._log("测试LDDU模型:")
                lddu_result = self.test_lddu_speed(video_path, rounds)
                self.test_results['lddu'].append(lddu_result)
            
            # 测试Qwen
            if test_qwen and self.qwen_service:
                self._log("测试Qwen模型:")
                qwen_result = self.test_qwen_speed(video_path, rounds)
                self.test_results['qwen'].append(qwen_result)
    
    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            'details': self.test_results
        }
        
        # 生成LDDU统计
        if self.test_results['lddu']:
            valid_lddu = [r for r in self.test_results['lddu'] if 'error' not in r]
            if valid_lddu:
                all_times = []
                for result in valid_lddu:
                    all_times.extend(result['times'])
                
                report['summary']['lddu'] = {
                    'total_tests': len(valid_lddu),
                    'total_inferences': len(all_times),
                    'avg_time': statistics.mean(all_times),
                    'min_time': min(all_times),
                    'max_time': max(all_times),
                    'std_dev': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                    'fps': 1.0 / statistics.mean(all_times) if all_times else 0
                }
        
        # 生成Qwen统计
        if self.test_results['qwen']:
            valid_qwen = [r for r in self.test_results['qwen'] if 'error' not in r]
            if valid_qwen:
                all_times = []
                for result in valid_qwen:
                    all_times.extend(result['times'])
                
                report['summary']['qwen'] = {
                    'total_tests': len(valid_qwen),
                    'total_inferences': len(all_times),
                    'avg_time': statistics.mean(all_times),
                    'min_time': min(all_times),
                    'max_time': max(all_times),
                    'std_dev': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                    'fps': 1.0 / statistics.mean(all_times) if all_times else 0
                }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印测试报告（同时写入日志）"""
        self._log("\n" + "=" * 60)
        self._log("情感识别推理速度测试报告")
        self._log("=" * 60)
        self._log(f"测试时间: {report['timestamp']}")
        
        for model_name, stats in report['summary'].items():
            self._log(f"\n{model_name.upper()} 模型性能统计:")
            self._log(f"  总测试数: {stats['total_tests']}")
            self._log(f"  总推理次数: {stats['total_inferences']}")
            self._log(f"  平均推理时间: {stats['avg_time']:.3f}s")
            self._log(f"  最快推理时间: {stats['min_time']:.3f}s")
            self._log(f"  最慢推理时间: {stats['max_time']:.3f}s")
            self._log(f"  标准差: {stats['std_dev']:.3f}s")
            self._log(f"  理论FPS: {stats['fps']:.2f}")
    
    def save_report(self, report: Dict[str, Any], output_file: str = "emotion_speed_report.json"):
        """保存测试报告到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._log(f"\n测试报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='情感识别推理速度测试')
    parser.add_argument('--video-dir', default='videos', help='测试视频目录')
    parser.add_argument('--rounds', type=int, default=3, help='每个视频的测试轮次')
    parser.add_argument('--no-lddu', action='store_true', help='跳过LDDU测试')
    parser.add_argument('--no-qwen', action='store_true', help='跳过Qwen测试')
    parser.add_argument('--output', default='emotion_speed_report.json', help='报告输出文件')
    parser.add_argument('--log', default='emotion_speed.log', help='日志文件路径')
    parser.add_argument('--no-json', action='store_true', help='不保存JSON报告，仅写入日志')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = EmotionSpeedTester()
    tester.set_log_file(args.log)
    
    # 初始化服务
    tester.initialize_services(
        test_lddu=not args.no_lddu,
        test_qwen=not args.no_qwen
    )
    
    # 获取测试视频
    video_files = tester.get_test_videos(args.video_dir)
    if not video_files:
        tester._log(f"在目录 {args.video_dir} 中未找到测试视频文件")
        return
    
    tester._log(f"找到 {len(video_files)} 个测试视频文件")
    
    # 运行基准测试
    tester.run_benchmark(
        video_files=video_files,
        rounds=args.rounds,
        test_lddu=not args.no_lddu,
        test_qwen=not args.no_qwen
    )
    
    # 生成并显示报告（同时写入日志）
    report = tester.generate_report()
    tester.print_report(report)
    if not args.no_json:
        tester.save_report(report, args.output)


if __name__ == '__main__':
    main()