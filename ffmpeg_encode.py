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
            # 确保使用绝对路径
            abs_path = os.path.abspath(img)
            # 使用转义的路径
            f.write(f"file '{abs_path.replace(os.sep, '/')}'\n")

    # 检查文件列表内容
    print(f"临时文件列表路径: {temp_list_path}")
    with open(temp_list_path, 'r') as f:
        list_content = f.read()
        # 只打印前3行和总行数
        lines = list_content.splitlines()
        print(f"文件列表包含 {len(lines)} 行")
        for i, line in enumerate(lines[:3]):
            print(f"  行 {i+1}: {line}")
        if len(lines) > 3:
            print(f"  ... 还有 {len(lines) - 3} 行未显示")
    
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
    chunk_size: int = 1024,
    save_temp_file: bool = True  # 新增参数用于控制是否保存临时文件
) -> Iterator[bytes]:
    """
    使用ffmpeg流式处理图片序列并编码为二进制数据块
    """
    # 直接处理时间戳格式
    glob_pattern = os.path.join(image_folder, pattern)
    all_files = glob.glob(glob_pattern)
    print(f"glob.glob找到 {len(all_files)} 个文件")
    print(f"前5个文件: {all_files[:5]}")
    
    # 检查正则表达式匹配
    match_failures = 0
    for file in all_files[:10]:  # 检查前10个文件
        match = re.search(r'_(\d+)\.', file)
        if not match:
            match_failures += 1
            print(f"警告: 文件 {os.path.basename(file)} 无法提取时间戳")
    
    if match_failures > 0:
        print(f"警告: {match_failures} 个文件无法提取时间戳，将改用更通用的模式")
    
    # 尝试更健壮的排序方式
    try:
        images = sorted(all_files, 
                 key=lambda f: int(re.search(r'_(\d+)\.', f).group(1)) 
                 if re.search(r'_(\d+)\.', f) else f)
    except Exception as e:
        print(f"排序时出错: {e}，将使用简单文件名排序")
        images = sorted(all_files)
    
    if not images:
        raise ValueError(f"未找到匹配 '{pattern}' 的图片")
    
    print(f"找到 {len(images)} 个匹配的图片文件")
    print(f"排序后前5个文件: {[os.path.basename(f) for f in images[:5]]}")
    
    # 创建临时文件列表
    temp_list_path = os.path.join("/tmp", f"img_list_{int(time.time())}.txt")
    with open(temp_list_path, 'w') as f:
        for img in images:
            # 确保使用绝对路径
            abs_path = os.path.abspath(img)
            # 使用转义的路径
            f.write(f"file '{abs_path.replace(os.sep, '/')}'\n")

    # 检查文件列表内容
    print(f"临时文件列表路径: {temp_list_path}")
    with open(temp_list_path, 'r') as f:
        list_content = f.read()
        # 只打印前3行和总行数
        lines = list_content.splitlines()
        print(f"文件列表包含 {len(lines)} 行")
        for i, line in enumerate(lines[:3]):
            print(f"  行 {i+1}: {line}")
        if len(lines) > 3:
            print(f"  ... 还有 {len(lines) - 3} 行未显示")
    
    # 创建临时文件用于保存编码后的视频流
    temp_video_path = None
    temp_file = None
    if save_temp_file:
        timestamp = int(time.time())
        temp_video_path = os.path.join("/tmp", f"encoded_video_{timestamp}.ts")
        temp_file = open(temp_video_path, 'wb')
        print(f"编码后的视频流将保存到: {temp_video_path}")
    
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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

    # 启动另一个线程读取标准错误
    stderr_log = []
    def read_stderr():
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            if line:
                log_line = line.decode('utf-8', errors='replace').strip()
                stderr_log.append(log_line)
                print(f"FFmpeg: {log_line}")

    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.daemon = True
    stderr_thread.start()
    
    try:
        # 流式读取输出
        total_bytes = 0
        chunk_count = 0
        
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
                
            # 如果启用了保存临时文件，写入数据
            if temp_file:
                temp_file.write(chunk)
                temp_file.flush()
            
            total_bytes += len(chunk)
            chunk_count += 1
            
            # 每10个块输出一次调试信息
            if chunk_count % 10 == 0:
                print(f"已编码 {chunk_count} 个块，共 {total_bytes} 字节")
                
            yield chunk
        
        print(f"编码完成，共 {chunk_count} 个块，{total_bytes} 字节")
        
        # 检查进程是否成功完成
        process.wait()
        if process.returncode != 0:
            error_message = process.stderr.read().decode()
            raise Exception(f"FFmpeg转换失败: {error_message}")
            
    finally:
        # 关闭临时文件
        if temp_file:
            temp_file.close()
            print(f"临时编码文件已保存: {temp_video_path}")
            
            # 分析临时文件中的视频帧
            try:
                cmd_analyze = [
                    "ffprobe", 
                    "-v", "error", 
                    "-count_frames",
                    "-select_streams", "v:0", 
                    "-show_entries", "stream=nb_read_frames,duration", 
                    "-of", "default=noprint_wrappers=1",
                    temp_video_path
                ]
                subprocess.run(cmd_analyze)
                print(f"临时视频文件分析完成")
            except Exception as e:
                print(f"分析文件失败: {e}")
                
        # 清理临时文件列表
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
    max_buffer_size: int = 10,
    save_temp_file: bool = True
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
        preset, crf, output_format, chunk_size,
        save_temp_file=save_temp_file
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
    # 移除守护线程设置
    # client_thread.daemon = True
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
            max_buffer_size=10,
            save_temp_file=True  # 添加此参数
        )
        print(f"发送完成: {stats['chunk_count']} 块，共 {stats['total_bytes']} 字节，耗时 {stats['elapsed_seconds']:.2f} 秒")
            
    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("关闭客户端...")

if __name__ == "__main__":
    main(server_addr='127.0.0.1', port_num=50051)  # 使用与face_blendshape示例相同的服务器