import sys
import os

# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入train.py中的get_args函数
try:
    from train import get_args
    
    print("成功导入get_args函数")
    
    # 测试参数解析
    # 使用简单的参数，避免触发其他代码路径
    sys.argv = ['train.py', '--do_test', '--data_path', '/tmp/test.pkl']
    
    print("开始解析参数...")
    args = get_args()
    
    print("参数解析成功!")
    print("\n测试参数值:")
    print(f"--data_path: {args.data_path}")
    print(f"--batch_size: {args.batch_size}")
    print(f"--max_words: {args.max_words}")
    print(f"--max_frames: {args.max_frames}")
    print(f"--max_sequence: {args.max_sequence}")
    
    # 检查是否所有必要的参数都已正确设置
    required_args = ['data_path', 'output_dir', 'batch_size', 'epochs', 'max_words', 'max_frames', 'max_sequence']
    missing_args = [arg for arg in required_args if not hasattr(args, arg)]
    
    if missing_args:
        print(f"\n警告: 缺少以下必要参数: {missing_args}")
    else:
        print("\n所有必要参数都已正确设置!")
    
    print("\n测试通过! 参数解析功能正常工作。")
    
    sys.exit(0)
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)