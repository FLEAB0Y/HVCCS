# 整合姿势和面部表情识别的发送器
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
import threading
import shutil
from client import THStreamClient
from THStreamData import THStreamDataPayload, THDataWarehouse
import os
import json

class FrameDataManager:
    """管理帧数据，整合同一帧的姿势和面部特征"""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_data = {}  # 使用时间戳作为键
        self.max_frames = 10  # 最大缓存帧数
    
    def update_pose_data(self, timestamp_ms, pose_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None}
            self.frame_data[timestamp_ms]["pose"] = pose_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()
    
    def update_face_data(self, timestamp_ms, face_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None}
            self.frame_data[timestamp_ms]["face"] = face_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()
    
    def _try_send_complete_frame(self, timestamp_ms):
        # 当一帧的姿势和面部数据都收集完成时调用此方法发送
        frame = self.frame_data.get(timestamp_ms)
        if frame and frame["pose"] is not None and frame["face"] is not None:
            # 这里需要访问client，通过全局变量或作为参数传入
            if hasattr(self, 'client'):
                payload_send = THStreamDataPayload(
                    rgb_data=b'\x01', 
                    point_data=b'\x02',
                    face_data=frame["face"],
                    limb_data=frame["pose"],    
                    ext_data=b'\x05', 
                    ext_desc=f"{str(timestamp_ms)}"
                )
                self.client.send_data_buffer.add_item(payload_send)
                # 发送后可以删除此帧数据
                del self.frame_data[timestamp_ms]
    
    def _cleanup_old_frames(self):
        # 清理旧帧防止内存溢出
        if len(self.frame_data) > self.max_frames:
            oldest_timestamp = min(self.frame_data.keys())
            del self.frame_data[oldest_timestamp]
    
    def set_client(self, client):
        self.client = client

class DataQuantizer:
    """数据量化工具类，用于减少浮点数据的大小"""
    
    @staticmethod
    def quantize_float_to_int(value, min_val, max_val, bits=8):
        """将浮点数量化为指定位数的整数
        
        Args:
            value: 要量化的浮点数值
            min_val: 浮点数范围最小值
            max_val: 浮点数范围最大值
            bits: 量化位数 (8, 16等)
            
        Returns:
            量化后的整数值
        """
        # 计算量化范围
        quant_range = 2**bits - 1
        # 归一化并量化
        normalized = (value - min_val) / (max_val - min_val)
        quantized = int(round(normalized * quant_range))
        # 确保在有效范围内
        return max(0, min(quantized, quant_range))
    
    @staticmethod
    def dequantize_int_to_float(quantized, min_val, max_val, bits=8):
        """将量化的整数还原为浮点数
        
        Args:
            quantized: 量化后的整数值
            min_val: 浮点数范围最小值
            max_val: 浮点数范围最大值
            bits: 量化位数
            
        Returns:
            还原的浮点数值
        """
        quant_range = 2**bits - 1
        normalized = quantized / quant_range
        return min_val + normalized * (max_val - min_val)
    
    @staticmethod
    def pack_quantized_ints(quantized_values, bits=8):
        """将量化后的整数打包为字节数组
        
        Args:
            quantized_values: 量化后的整数列表
            bits: 每个值的位数
            
        Returns:
            打包后的字节数组
        """
        import struct
        
        if bits == 8:
            # 8位量化直接打包为无符号字节
            return bytes(quantized_values)
        elif bits == 16:
            # 16位量化打包为无符号短整型
            return struct.pack(f'<{len(quantized_values)}H', *quantized_values)
        else:
            raise ValueError(f"不支持的量化位数: {bits}")

class NoiseGenerator:
    """为数据添加噪声以模拟通信过程"""
    
    @staticmethod
    def add_gaussian_noise(data, noise_level=0.0, bits=8):
        """为量化后的数据添加高斯噪声
        
        Args:
            data: 量化后的整数列表
            noise_level: 噪声强度 (0.0-1.0)
            bits: 量化位数
            
        Returns:
            添加噪声后的数据列表
        """
        if noise_level <= 0:
            return data  # 如果噪声级别为0或负数，不添加噪声
            
        import numpy as np
        
        # 转换为numpy数组以便批量处理
        data_array = np.array(data)
        
        # 计算量化范围
        max_value = 2**bits - 1
        
        # 计算噪声标准差（相对于数据范围）
        std_dev = max_value * noise_level
        
        # 生成高斯噪声
        noise = np.random.normal(0, std_dev, size=data_array.shape)
        
        # 添加噪声
        noisy_data = data_array + noise.astype(int)
        
        # 确保值在有效范围内
        noisy_data = np.clip(noisy_data, 0, max_value)
        
        return noisy_data.astype(int).tolist()

def run_client(client):
    """运行客户端线程"""
    client.run()

def process_pose_result(result, timestamp_ms, frame_data_manager, frame_width, frame_height, 
                        debug=False, res_path=None, quantize_bits=16, noise_level=0.0):
    """处理姿势检测结果并更新帧数据管理器，支持数据量化和噪声添加"""
    if result.pose_landmarks:
        # 处理姿势数据
        for pose_landmark in result.pose_landmarks:
            # 转换为像素坐标并处理
            landmarks_data = []
            for landmark in pose_landmark:
                landmarks_data.append((
                    landmark.x * frame_width,
                    frame_height - landmark.y * frame_height,
                    landmark.z * frame_width
                ))
            
            # 量化姿势数据
            if quantize_bits > 0:
                # 确定坐标范围用于量化
                quantized_landmarks = []
                for x, y, z in landmarks_data:
                    # 量化x坐标
                    qx = DataQuantizer.quantize_float_to_int(x, 0, frame_width, quantize_bits)
                    # 量化y坐标
                    qy = DataQuantizer.quantize_float_to_int(y, 0, frame_height, quantize_bits)
                    # 量化z坐标
                    qz = DataQuantizer.quantize_float_to_int(z, -frame_width/2, frame_width/2, quantize_bits)
                    quantized_landmarks.extend([qx, qy, qz])
                
                # 添加高斯噪声
                if noise_level > 0:
                    quantized_landmarks = NoiseGenerator.add_gaussian_noise(
                        quantized_landmarks, noise_level, quantize_bits)
                    
                # 打包量化数据
                landmarks_data_bytes = DataQuantizer.pack_quantized_ints(quantized_landmarks, quantize_bits)
                # 添加量化元数据头 (位数、坐标数量等)
                metadata = struct.pack('<BB', quantize_bits, len(landmarks_data) * 3)
                landmarks_data_bytes = metadata + landmarks_data_bytes
            else:
                # 不量化，使用原来的方式
                flat_landmarks = [coord for landmark in landmarks_data for coord in landmark]
                landmarks_str = ','.join(map(str, flat_landmarks))
                landmarks_data_bytes = landmarks_str.encode('utf-8')
            
            # 更新帧数据管理器
            frame_data_manager.update_pose_data(timestamp_ms, landmarks_data_bytes)
        
        if debug:
            print(f"姿势数据大小: {len(landmarks_data_bytes)} 字节")
            if quantize_bits > 0:
                print(f"姿势数据使用 {quantize_bits} 位量化")
            if noise_level > 0:
                print(f"姿势数据添加了 {noise_level:.2f} 级噪声")
            current_timestamp_ms = int(time.time() * 1000)
            print(f"姿势处理延迟: {current_timestamp_ms - timestamp_ms} ms")

def process_face_result(result, timestamp_ms, frame_data_manager, debug=False, 
                       res_path=None, quantize_bits=8, noise_level=0.0):
    """处理面部表情检测结果并更新帧数据管理器，支持数据量化和噪声添加"""
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes:
            # 提取所有分数值
            blendshape_data = [category.score for category in blendshape]
            
            # 量化面部表情数据
            if quantize_bits > 0:
                # 面部表情分数范围为0到1
                quantized_blendshapes = []
                for score in blendshape_data:
                    q_score = DataQuantizer.quantize_float_to_int(score, 0, 1, quantize_bits)
                    quantized_blendshapes.append(q_score)
                
                # 添加高斯噪声
                if noise_level > 0:
                    quantized_blendshapes = NoiseGenerator.add_gaussian_noise(
                        quantized_blendshapes, noise_level, quantize_bits)
                
                # 打包量化数据
                blendshape_data_bytes = DataQuantizer.pack_quantized_ints(quantized_blendshapes, quantize_bits)
                # 添加量化元数据头 (位数、表情数量)
                metadata = struct.pack('<BB', quantize_bits, len(blendshape_data))
                blendshape_data_bytes = metadata + blendshape_data_bytes
            else:
                # 不量化，使用原来的方式
                blendshape_str = ','.join(map(str, blendshape_data))
                blendshape_data_bytes = blendshape_str.encode('utf-8')

            # 更新帧数据管理器
            frame_data_manager.update_face_data(timestamp_ms, blendshape_data_bytes)
            
            if debug:
                print(f"面部表情数据大小: {len(blendshape_data_bytes)} 字节")
                if quantize_bits > 0:
                    print(f"面部表情数据使用 {quantize_bits} 位量化")
                if noise_level > 0:
                    print(f"面部表情数据添加了 {noise_level:.2f} 级噪声")
                current_timestamp_ms = int(time.time() * 1000)
                print(f"面部处理延迟: {current_timestamp_ms - timestamp_ms} ms")
                
                print("===== 检测到的面部表情数据 =====")
                for i, score in enumerate(blendshape_data):
                    print(f"表情 {i}: 强度={score:.4f}")
                print("===============================")

def load_video_settings(config_path):
    """加载视频处理配置文件
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        包含全局设置和视频特定设置的字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # 确保配置文件包含必要的字段
        if 'global_settings' not in settings:
            settings['global_settings'] = {
                "pose_quantize_bits": 16,
                "face_quantize_bits": 8,
                "pose_noise_level": 0.05,
                "face_noise_level": 0.02
            }
        if 'video_settings' not in settings:
            settings['video_settings'] = {}
            
        return settings
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        # 返回默认设置
        return {
            'global_settings': {
                "pose_quantize_bits": 16,
                "face_quantize_bits": 8,
                "pose_noise_level": 0.05,
                "face_noise_level": 0.02
            },
            'video_settings': {}
        }

def get_video_settings(video_path, settings):
    """获取特定视频的处理参数
    
    Args:
        video_path: 视频文件路径
        settings: 配置设置字典
        
    Returns:
        视频的处理参数字典
    """
    video_filename = os.path.basename(video_path)
    global_settings = settings['global_settings']
    video_specific_settings = settings['video_settings'].get(video_filename, {})
    
    # 将特定视频设置与全局设置合并
    video_params = {
        "pose_quantize_bits": video_specific_settings.get('pose_quantize_bits', global_settings.get('pose_quantize_bits', 16)),
        "face_quantize_bits": video_specific_settings.get('face_quantize_bits', global_settings.get('face_quantize_bits', 8)),
        "pose_noise_level": video_specific_settings.get('pose_noise_level', global_settings.get('pose_noise_level', 0.05)),
        "face_noise_level": video_specific_settings.get('face_noise_level', global_settings.get('face_noise_level', 0.02))
    }
    
    return video_params

def main(server_addr='127.0.0.1', port_num=50051, 
         pose_model_path=None, face_model_path=None, 
         debug=False, res_path=None, video_path=None, video_folder=None,
         config_path=None):
    """主函数：设置并运行姿势和面部表情检测，支持每个视频的自定义参数"""
    if not video_path and not video_folder:
        raise ValueError("必须提供视频文件路径或视频文件夹路径")
    
    # 加载视频处理配置
    if config_path:
        settings = load_video_settings(config_path)
    else:
        settings = {
            'global_settings': {
                "pose_quantize_bits": 16,
                "face_quantize_bits": 8,
                "pose_noise_level": 0.05,
                "face_noise_level": 0.02
            },
            'video_settings': {}
        }
    
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # 创建客户端
    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()
    
    # 创建帧数据管理器
    frame_data_manager = FrameDataManager()
    frame_data_manager.set_client(client)
    
    # 获取要处理的视频文件列表
    video_files = []
    if video_folder:
        # 支持的视频文件扩展名
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
        # 获取文件夹中所有视频文件
        for file in os.listdir(video_folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(video_folder, file))
        if not video_files:
            raise ValueError(f"在文件夹 {video_folder} 中未找到视频文件")
        print(f"找到 {len(video_files)} 个视频文件待处理")
    else:
        # 单个视频文件模式
        video_files = [video_path]
    
    # 创建姿势检测器
    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options, 
        running_mode=VisionRunningMode.VIDEO, 
        output_segmentation_masks=False,
        # 先使用默认回调函数，稍后会更新
        result_callback=lambda result, _, timestamp_ms: None
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    # 创建面部表情检测器
    face_base_options = python.BaseOptions(model_asset_path=face_model_path)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options, 
        running_mode=VisionRunningMode.VIDEO, 
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        # 先使用默认回调函数，稍后会更新
        result_callback=lambda result, _, timestamp_ms: None
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # 处理每个视频文件
    for index, video_file in enumerate(video_files):
        # 获取特定视频的参数
        video_params = get_video_settings(video_file, settings)
        
        print(f"开始处理视频 {index+1}/{len(video_files)}: {os.path.basename(video_file)}")
        print(f"使用参数: 姿势量化={video_params['pose_quantize_bits']}位, "
              f"面部量化={video_params['face_quantize_bits']}位, "
              f"姿势噪声={video_params['pose_noise_level']:.2f}, "
              f"面部噪声={video_params['face_noise_level']:.2f}")
        
        # 为每个视频处理使用其特定的参数
        process_video(
            video_file, 
            frame_data_manager, 
            pose_detector, 
            face_detector, 
            client, 
            debug=debug,
            pose_quantize_bits=video_params['pose_quantize_bits'],
            face_quantize_bits=video_params['face_quantize_bits'],
            pose_noise_level=video_params['pose_noise_level'],
            face_noise_level=video_params['face_noise_level']
        )
        
        # 在两个视频之间暂停30秒（最后一个视频除外）
        if index < len(video_files) - 1:
            print(f"视频 {os.path.basename(video_file)} 处理完成，等待30秒后继续...")
            for i in range(30, 0, -1):
                print(f"\r等待继续: {i}秒", end="")
                time.sleep(1)
            print("\n开始处理下一个视频...")

def process_video(video_path, frame_data_manager, pose_detector, face_detector, client, 
                 debug=False, pose_noise_level=0.0, face_noise_level=0.0, pose_quantize_bits=16, face_quantize_bits=8):
    """处理单个视频文件"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告: 无法打开视频文件: {video_path}，跳过此文件")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 更新处理姿势结果的回调函数中的帧尺寸和噪声设置
    pose_detector.result_callback = lambda result, _, timestamp_ms: process_pose_result(
        result, timestamp_ms, frame_data_manager, frame_width, frame_height, 
        debug=debug, quantize_bits=pose_quantize_bits, noise_level=pose_noise_level
    )
    
    # 更新面部表情处理回调函数
    face_detector.result_callback = lambda result, _, timestamp_ms: process_face_result(
        result, timestamp_ms, frame_data_manager, 
        debug=debug, quantize_bits=face_quantize_bits, noise_level=face_noise_level
    )
    
    # 使用基于帧的相对时间戳
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            # 从视频文件读取一帧
            ret, frame = cap.read()
            if not ret:
                print("视频处理完成")
                break
            
            # 计算当前帧的时间戳（毫秒）
            frame_timestamp_ms = int(frame_count * (1000 / fps))
            
            # 缓冲区满了就等待
            buffer_size = client.send_data_buffer.get_size()
            while buffer_size >= 5:
                time.sleep(0.01)
                buffer_size = client.send_data_buffer.get_size()
    
            # 将图像从BGR颜色空间转换为RGB颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 为两个检测器分别异步检测
            pose_detector.detect_async(mp_image, frame_timestamp_ms)
            face_detector.detect_async(mp_image, frame_timestamp_ms)
            
            if debug:
                print(f"帧索引: {frame_count}, 时间戳: {frame_timestamp_ms}ms")
            
            frame_count += 1
            
            # 模拟视频实际播放速度
            time.sleep(1. / fps)
            
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()
            print('视频资源已释放')

if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建模型路径
    pose_model_path = os.path.join(script_dir, "..", "data", "pose_landmarker_full.task")
    face_model_path = os.path.join(script_dir, "..", "data", "face_landmarker_v2_with_blendshapes.task")
    
    # 视频文件夹路径
    video_folder = os.path.join(script_dir, "..", "data", "videos")
    
    # 配置文件路径
    config_path = os.path.join(script_dir, "..", "data", "video_settings.json")
    
    main(
        server_addr='127.0.0.1',
        port_num=50051, 
        pose_model_path=pose_model_path,
        face_model_path=face_model_path,
        debug=False,
        video_folder=video_folder,
        config_path=config_path  # 添加配置文件路径
    )