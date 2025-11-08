import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import json
import signal
import sys
import argparse

# 配置
TSV_FILE = 'open-images-dataset-train0.tsv'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # 下载图片保存的根目录
BATCH_SIZE = 1000  # 每个文件夹存储的图片数
MAX_WORKERS = 64  # 并行下载的线程数
TIMEOUT = 30  # 下载超时时间（秒）
RETRY_COUNT = 3  # 下载失败重试次数
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'download_progress.json')  # 进度记录文件

def download_image(url, save_path):
    """下载单张图片并保存到指定路径"""
    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True)
            response.raise_for_status()  # 检查是否成功
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 写入文件
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(1)  # 重试前等待
                continue
            print(f"下载失败 {url}: {str(e)}")
            return False

def extract_image_id(url):
    """从URL中提取图片ID作为文件名"""
    return url.split('/')[-1]

def process_batch(batch_data, batch_index):
    """处理一批图片下载任务"""
    folder_name = str(batch_index)
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    
    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)
    
    successful = 0
    failed = 0
    
    # 使用线程池并行下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, (url, _) in enumerate(batch_data):
            image_id = extract_image_id(url)
            image_name = f"{i:04d}_{image_id}"
            save_path = os.path.join(folder_path, image_name)
            futures.append(executor.submit(download_image, url, save_path))
        
        # 等待并统计结果
        for future in tqdm(futures, desc=f"Batch {batch_index}", total=len(futures)):
            if future.result():
                successful += 1
            else:
                failed += 1
    
    return successful, failed

def main():
    print(f"开始处理: {TSV_FILE}")
    
    # 检测TSV文件格式
    tsv_format = detect_tsv_format()
    print(f"检测到TSV文件格式: {tsv_format}")
    
    # 加载进度信息
    downloaded_urls = set()
    last_processed_row = -1
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
                downloaded_urls = set(progress_data.get('downloaded_urls', []))
                last_processed_row = progress_data.get('last_processed_row', -1)
                print(f"找到已有进度记录: 已处理 {last_processed_row+1} 行, 已下载 {len(downloaded_urls)} 张图片")
        except Exception as e:
            print(f"读取进度文件失败: {e}, 将从头开始下载")
    
    # 更新全局进度变量
    current_progress['downloaded_urls'] = downloaded_urls
    current_progress['last_processed_row'] = last_processed_row
    
    # 统计总行数以进行进度显示
    total_rows = 0
    with open(TSV_FILE, 'r', encoding='utf-8') as f:
        for _ in f:
            total_rows += 1
    print(f"共发现 {total_rows} 行数据")
    
    batch_data = []
    batch_index = last_processed_row // BATCH_SIZE
    total_successful = len(downloaded_urls)
    total_failed = 0
    
    with open(TSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            # 跳过已处理的行
            if i <= last_processed_row:
                continue
                
            if not row:
                continue  # 跳过空行
                
            # 根据检测到的格式获取URL
            url = get_url_from_row(row, tsv_format)
            if not url:
                continue  # 跳过无效行
            
            # 跳过已下载的URL
            if url in downloaded_urls:
                continue
                
            batch_data.append((url, row))
            
            # 达到批次大小或处理完最后一批
            if len(batch_data) >= BATCH_SIZE or i == total_rows - 1:
                if batch_data:  # 确保有数据才处理
                    batch_index += 1
                    print(f"\n处理第 {batch_index} 批 ({len(batch_data)} 张图片)...")
                    successful, failed = process_batch(batch_data, batch_index)
                    total_successful += successful
                    total_failed += failed
                    
                    # 更新进度记录
                    for processed_url, _ in batch_data:
                        downloaded_urls.add(processed_url)
                        current_progress['downloaded_urls'].add(processed_url)
                    
                    current_progress['last_processed_row'] = i
                    
                    progress_data = {
                        'last_processed_row': i,
                        'downloaded_urls': list(downloaded_urls)
                    }
                    
                    with open(PROGRESS_FILE, 'w') as f:
                        json.dump(progress_data, f)
                    
                    print(f"批次完成: 成功 {successful}, 失败 {failed}")
                    batch_data = []  # 重置批次数据
    
    print(f"\n全部处理完成! 共 {total_successful} 张下载成功, {total_failed} 张下载失败")

def detect_tsv_format():
    """检测TSV文件的格式，返回URL所在的列索引"""
    try:
        with open(TSV_FILE, 'r', encoding='utf-8') as f:
            for _ in range(5):  # 检查前5行
                line = f.readline().strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) > 0:
                    # 如果第一列看起来像URL
                    if parts[0].startswith('http'):
                        return 'url_first'
                    # 如果有其他列看起来像URL
                    for i, part in enumerate(parts):
                        if part.startswith('http'):
                            return f'url_at_{i}'
                            
    except Exception as e:
        print(f"检测TSV格式时出错: {e}")
    
    # 默认假设URL在第一列
    return 'url_first'

def get_url_from_row(row, format_type):
    """根据检测到的格式从行中提取URL"""
    if not row:
        return None
        
    if format_type == 'url_first':
        return row[0] if row[0].startswith('http') else None
    elif format_type.startswith('url_at_'):
        try:
            index = int(format_type.split('_')[-1])
            return row[index] if len(row) > index and row[index].startswith('http') else None
        except:
            pass
            
    # 尝试在任何列中找URL
    for cell in row:
        if cell.startswith('http'):
            return cell
            
    return None

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量下载图片工具')
    parser.add_argument('--tsv', type=str, help='指定TSV文件路径')
    parser.add_argument('--output', type=str, help='指定输出目录')
    parser.add_argument('--batch-size', type=int, help='每个文件夹的图片数量')
    parser.add_argument('--workers', type=int, help='并行下载的线程数')
    args = parser.parse_args()
    
    # 根据命令行参数更新配置
    if args.tsv:
        TSV_FILE = args.tsv
        print(f"使用指定的TSV文件: {TSV_FILE}")
    
    if args.output:
        OUTPUT_DIR = args.output
        print(f"使用指定的输出目录: {OUTPUT_DIR}")
        
    if args.batch_size:
        BATCH_SIZE = args.batch_size
        print(f"每个文件夹的图片数量: {BATCH_SIZE}")
        
    if args.workers:
        MAX_WORKERS = args.workers
        print(f"并行下载的线程数: {MAX_WORKERS}")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 全局变量用于信号处理
    current_progress = {
        'last_processed_row': -1,
        'downloaded_urls': set()
    }
    
    # 处理Ctrl+C信号
    def signal_handler(sig, frame):
        print("\n程序被中断，正在保存进度...")
        progress_data = {
            'last_processed_row': current_progress['last_processed_row'],
            'downloaded_urls': list(current_progress['downloaded_urls'])
        }
        
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_data, f)
            print(f"进度已保存到 {PROGRESS_FILE}")
        except Exception as e:
            print(f"保存进度失败: {e}")
            
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 更新进度文件路径
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'download_progress.json')
    
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"总耗时: {elapsed:.2f} 秒")
