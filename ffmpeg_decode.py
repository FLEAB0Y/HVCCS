import os
import time
import shutil
import threading
import subprocess
import io
from datetime import datetime
import grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import data_stream_pb2_grpc
from server import THStreamServiceServicer, serve


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def decode_vid(servicer, decode_output_path, save_frames=True, save_video=True):
    """
    从接收缓冲区获取视频数据并解码
    
    参数:
        servicer: 服务器实例，包含接收缓冲区
        decode_output_path: 解码后视频/帧的保存路径
        save_frames: 是否保存为单独的图片帧
        save_video: 是否保存为完整视频文件
    """
    # 确保输出目录存在
    if not os.path.exists(decode_output_path):
        os.makedirs(decode_output_path)
    
    # 创建帧目录
    frames_dir = os.path.join(decode_output_path, "frames")
    if save_frames and not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # 创建临时文件用于存储收到的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_path = os.path.join(decode_output_path, f"temp_stream_{timestamp}.ts")
    video_output_path = os.path.join(decode_output_path, f"decoded_video_{timestamp}.mp4")
    
    # 打开临时文件用于写入
    with open(temp_file_path, 'wb') as temp_file:
        received_data_size = 0
        start_time = time.time()
        last_data_time = time.time()  # 添加此行初始化变量
        frame_count = 0
        
        print(f"开始接收视频数据，临时文件: {temp_file_path}")
        
        try:
            # 循环获取和处理数据
            while True:
                # 缓冲区空了就等待
                buffer_size = servicer.receive_data_buffer.get_size()
                if buffer_size < 1:
                    time.sleep(0.1)
                    # 检查是否已经超过10秒没有接收到新数据
                    if received_data_size > 0 and (time.time() - last_data_time) > 10:
                        print("超过10秒未接收到数据，认为传输完成")
                        break
                    continue  # 继续下一次循环
                
                # 循环获取缓冲区中的所有数据
                while buffer_size > 0:
                    payload_rec = servicer.receive_data_buffer.get_items()
                    buffer_size = servicer.receive_data_buffer.get_size()
                    if not payload_rec:
                        break
                        
                    try:
                        rgb_data_bytes = payload_rec.rgbData
                        if rgb_data_bytes:
                            # 将数据写入临时文件
                            temp_file.write(rgb_data_bytes)
                            temp_file.flush()
                            
                            # 更新统计信息
                            received_data_size += len(rgb_data_bytes)
                            last_data_time = time.time()
                            frame_count += 1
                            
                            # 输出进度
                            elapsed = time.time() - start_time
                            rate = received_data_size / elapsed / 1024 if elapsed > 0 else 0
                            print(f"接收数据块 #{frame_count}, 大小: {len(rgb_data_bytes)} 字节, "
                                  f"总计: {received_data_size} 字节, 速率: {rate:.2f} KB/s")
                            
                    except AttributeError as e:
                        print(f"AttributeError: {e}")
                    
        except KeyboardInterrupt:
            print("用户中断接收过程")
        
        print(f"数据接收完成，共 {received_data_size} 字节")
                
    # 如果没有接收到数据，删除临时文件并返回
    if received_data_size == 0:
        print("未接收到任何数据")
        os.remove(temp_file_path)
        return
    
    # 解码视频
    if save_video:
        print(f"开始解码视频到: {video_output_path}")
        try:
            # 解码为视频文件
            cmd_video = [
                "ffmpeg", "-y",
                "-i", temp_file_path,
                "-c:v", "libx264",
                "-preset", "medium",
                "-vf", "format=yuv420p",  # 确保兼容性
                "-movflags", "+faststart",  # 优化流式播放
                video_output_path
            ]
            subprocess.run(cmd_video, check=True)
            print(f"视频解码完成: {video_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"视频解码失败: {e}")
    
    # 提取帧
    if save_frames:
        print(f"开始提取帧到: {frames_dir}")
        try:
            # 解码为单独的图片帧
            cmd_frames = [
                "ffmpeg", "-y",
                "-i", temp_file_path,
                "-vsync", "0",
                "-vf", "fps=30",  # 强制指定帧率
                os.path.join(frames_dir, "frame_%04d.png")
            ]
            subprocess.run(cmd_frames, check=True)
            print(f"帧提取完成，保存在: {frames_dir}")
        except subprocess.CalledProcessError as e:
            print(f"帧提取失败: {e}")
    
    # 保留临时文件用于调试
    # os.remove(temp_file_path)

def main(decode_output_path, save_frames=True, save_video=True):
    """
    主函数
    
    参数:
        decode_output_path: 解码输出路径
        save_frames: 是否保存单独的帧图片
        save_video: 是否保存为视频文件
    """
    # 确保输出目录存在
    if not os.path.exists(decode_output_path):
        os.makedirs(decode_output_path)
    else:
        # 询问用户是否清空目录
        print(f"输出目录 {decode_output_path} 已存在")
        choice = input("是否清空该目录? (y/n): ").strip().lower()
        if choice == 'y':
            clear_folder(decode_output_path)
            print(f"已清空目录: {decode_output_path}")
    
    # 开启服务器线程
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer,))
    server_thread.start()
    
    print("服务器已启动，等待接收视频数据...")
    
    try:
        decode_vid(servicer, decode_output_path, save_frames, save_video)
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("程序结束")

if __name__ == '__main__':
    main(
        decode_output_path="/home/ztw/HVCCS/res/decode_res",
        save_frames=True,  # 保存为单独的帧
        save_video=True    # 保存为视频文件
    )

