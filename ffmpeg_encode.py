import os
import subprocess
import time
import glob
import re
import threading
from typing import Callable, Iterator, Dict
from client import THStreamClient
from THStreamData import THStreamDataPayload

def read_image_sequence_to_binary(
    image_folder: str,
    pattern: str = "render_*.png",
    framerate: int = 30,
    codec: str = "libx264",
    preset: str = "ultrafast",
    crf: int = 23,
    output_format: str = "mpegts"
) -> bytes:
    """
    使用ffmpeg将图片序列读取并编码为二进制数据
    """
    # 直接处理时间戳格式
    glob_pattern = os.path.join(image_folder, pattern)
    images = sorted(glob.glob(glob_pattern), 
                     key=lambda f: int(re.search(r'_(\d+)\.', f).group(1)) 
                     if re.search(r'_(\d+)\.', f) else f)
    
    if not images:
        raise ValueError(f"未找到匹配 '{pattern}' 的图片")
    
    # 创建临时文件列表
    temp_list_path = os.path.join("/tmp", f"img_list_{int(time.time())}.txt")
    with open(temp_list_path, 'w') as f:
        for img in images:
            f.write(f"file '{img}'\n")
    
    try:
        # 构建ffmpeg命令，使用文件列表
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_list_path,
            "-r", str(framerate),
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-f", output_format,
            "-"
        ]
        
        # 执行ffmpeg命令，捕获二进制输出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 检查是否成功
        if result.returncode != 0:
            raise Exception(f"FFmpeg转换失败: {result.stderr.decode()}")
        
        return result.stdout
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)

def stream_image_sequence_to_binary(
    image_folder: str,
    pattern: str = "render_*.png",
    framerate: int = 30,
    codec: str = "libx264",
    preset: str = "ultrafast",
    crf: int = 23,
    output_format: str = "mpegts",
    chunk_size: int = 1024
) -> Iterator[bytes]:
    """
    使用ffmpeg流式处理图片序列并编码为二进制数据块
    """
    # 直接处理时间戳格式
    glob_pattern = os.path.join(image_folder, pattern)
    images = sorted(glob.glob(glob_pattern), 
                     key=lambda f: int(re.search(r'_(\d+)\.', f).group(1)) 
                     if re.search(r'_(\d+)\.', f) else f)
    
    if not images:
        raise ValueError(f"未找到匹配 '{pattern}' 的图片")
    
    # 创建临时文件列表
    temp_list_path = os.path.join("/tmp", f"img_list_{int(time.time())}.txt")
    with open(temp_list_path, 'w') as f:
        for img in images:
            f.write(f"file '{img}'\n")
    
    # 构建ffmpeg命令，使用文件列表
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", temp_list_path,
        "-r", str(framerate),
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-f", output_format,
        "-"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 启动ffmpeg进程
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # 流式读取输出
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
        
        # 检查进程是否成功完成
        process.wait()
        if process.returncode != 0:
            error_message = process.stderr.read().decode()
            raise Exception(f"FFmpeg转换失败: {error_message}")
    finally:
        # 确保清理临时文件
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)

def default_send_function(data: bytes, timestamp_ms: int, client: THStreamClient) -> None:
    """
    默认的数据发送函数实现 - 向客户端缓冲区添加数据
    """
    # 往缓冲区放入数据
    payload_send = THStreamDataPayload(
        rgb_data=data, 
        point_data=b'\x02', 
        face_data=b'\x03', 
        limb_data=b'\x04',    
        ext_data=b'\x05', 
        ext_desc=f"{str(timestamp_ms)}"
    )
    client.send_data_buffer.add_item(payload_send)

def run_client(client):
    """运行客户端的线程函数"""
    client.run()

def process_and_send_realtime(
    image_folder: str,
    client: THStreamClient,
    pattern: str = "render_*.png",
    framerate: int = 30,
    codec: str = "libx264",
    preset: str = "ultrafast",
    crf: int = 23,
    output_format: str = "mpegts",
    chunk_size: int = 1024,
    max_buffer_size: int = 10
) -> Dict[str, int]:
    """
    实时处理和发送图片序列，控制缓冲区大小
    """
    chunk_count = 0
    total_bytes = 0
    start_time = time.time()
    
    # 创建迭代器处理数据流
    for chunk in stream_image_sequence_to_binary(
        image_folder, pattern, framerate, codec, 
        preset, crf, output_format, chunk_size
    ):
        # 缓冲区满了就等待
        buffer_size = client.send_data_buffer.get_size()
        while buffer_size >= max_buffer_size:
            time.sleep(0.1)
            buffer_size = client.send_data_buffer.get_size()
            
        # 计算当前时间戳
        timestamp_ms = int(time.time() * 1000)
        
        # 发送数据
        default_send_function(chunk, timestamp_ms, client)
        
        chunk_count += 1
        total_bytes += len(chunk)
        
        elapsed = time.time() - start_time
        rate = chunk_count / elapsed if elapsed > 0 else 0
        
        print(f"已发送块 #{chunk_count}, 大小 {len(chunk)} 字节, 总计 {total_bytes} 字节, 速率 {rate:.2f} 块/秒")
    
    return {
        "chunk_count": chunk_count,
        "total_bytes": total_bytes,
        "elapsed_seconds": time.time() - start_time
    }

def main(image_folder="/home/ztw/HVCCS/res/render_res", 
         pattern="render_*.png", 
         framerate=30, 
         chunk_size=1024, 
         server_addr='127.0.0.1', 
         port_num=50051):
    """
    主函数：创建客户端线程并处理图片序列发送
    """
    # 创建并启动客户端线程
    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.daemon = True  # 设置为守护线程，主线程结束时自动退出
    client_thread.start()
    
    try:
        # 流式处理并发送
        print("开始处理图片序列...")
        stats = process_and_send_realtime(
            image_folder=image_folder,
            client=client,
            pattern=pattern,
            framerate=framerate,
            chunk_size=chunk_size,
            max_buffer_size=10  # 设置最大缓冲区大小与face_blendshape示例一致
        )
        print(f"发送完成: {stats['chunk_count']} 块，共 {stats['total_bytes']} 字节，耗时 {stats['elapsed_seconds']:.2f} 秒")
        
        # 等待缓冲区清空
        while client.send_data_buffer.get_size() > 0:
            time.sleep(0.5)
            print(f"等待缓冲区清空，当前大小: {client.send_data_buffer.get_size()}")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 等待一些时间让最后的数据发送完成
        time.sleep(2)
        client.stop()  # 停止客户端
        client_thread.join(timeout=5)  # 等待客户端线程结束

if __name__ == "__main__":
    main(server_addr='101.6.65.237', port_num=50051)  # 使用与face_blendshape示例相同的服务器