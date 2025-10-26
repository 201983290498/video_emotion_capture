import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('delete_videos.log'),
        logging.StreamHandler()
    ]
)

# 日志文件路径
LOG_FILE_PATH = "/data/jianghong/R1-Omni/processed_videos.log"

def delete_videos_from_log(log_file_path):
    """
    从日志文件中读取视频路径并删除对应的视频文件
    
    Args:
        log_file_path: 包含视频路径的日志文件路径
    """
    if not os.path.exists(log_file_path):
        logging.error(f"日志文件不存在: {log_file_path}")
        return
    
    deleted_count = 0
    failed_count = 0
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            video_paths = f.readlines()
        
        total_videos = len(video_paths)
        logging.info(f"找到 {total_videos} 个视频路径")
        
        for i, line in enumerate(video_paths, 1):
            # 去除行尾的换行符和空白字符
            video_path = line.strip()
            
            if not video_path:
                continue
            
            logging.info(f"处理 ({i}/{total_videos}): {video_path}")
            
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logging.info(f"已删除: {video_path}")
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"删除失败 {video_path}: {str(e)}")
                    failed_count += 1
            else:
                logging.warning(f"文件不存在: {video_path}")
                failed_count += 1
    
    except Exception as e:
        logging.error(f"处理日志文件时出错: {str(e)}")
    
    logging.info(f"删除完成: 成功 {deleted_count}, 失败 {failed_count}")

def main():
    """
    主函数
    """
    log_file = LOG_FILE_PATH
    
    # 如果命令行参数提供了日志文件路径，则使用命令行参数
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    logging.info(f"开始从日志文件删除视频: {log_file}")
    delete_videos_from_log(log_file)

if __name__ == "__main__":
    main()