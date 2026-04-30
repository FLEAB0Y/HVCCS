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
import argparse  # 添加此行
import json

class FrameDataManager:
    """管理帧数据，整合同一帧的姿势和面部特征"""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_data = {}  # 使用时间戳作为键
        self.max_frames = 10  # 最大缓存帧数
        self.last_valid_face_data = None  # 存储最近一次有效的面部数据
        self.last_two_pose_frames = []  # 存储最近两帧的姿势数据用于平滑处理
        self.frame_count = 0  # 已处理的帧计数
        self.last_processed_landmarks = None  # 存储上一帧处理后的关键点数据
    
    def update_pose_data(self, timestamp_ms, pose_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None, "t_encode": None}
            self.frame_data[timestamp_ms]["pose"] = pose_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()
    
    def update_face_data(self, timestamp_ms, face_data):
        with self.lock:
            if timestamp_ms not in self.frame_data:
                self.frame_data[timestamp_ms] = {"pose": None, "face": None, "t_encode": None}
            self.frame_data[timestamp_ms]["face"] = face_data
            self._try_send_complete_frame(timestamp_ms)
            self._cleanup_old_frames()
    
    def _try_send_complete_frame(self, timestamp_ms):
        # 当一帧的姿势和面部数据都收集完成时调用此方法发送
        frame = self.frame_data.get(timestamp_ms)
        if frame and frame["pose"] is not None and frame["face"] is not None:
            # 这里需要访问client，通过全局变量或作为参数传入
            if hasattr(self, 'client'):
                # 平滑处理姿势数据
                smoothed_pose = self._smooth_pose_data(frame["pose"])
                # t_encode: 即将放入发送缓冲区的时刻
                t_encode_ms = int(time.time() * 1000)
                timing_meta = {
                    "t_begin": int(timestamp_ms),
                    "t_encode": t_encode_ms
                }
                
                payload_send = THStreamDataPayload(
                    rgb_data=b'\x00', 
                    point_data=b'\x00',
                    face_data=frame["face"],
                    limb_data=smoothed_pose,    
                    ext_data=json.dumps(timing_meta).encode('utf-8'), 
                    ext_desc=f"{str(timestamp_ms)}"
                )
                self.client.send_data_buffer.add_item(payload_send)
                # 发送后可以删除此帧数据
                del self.frame_data[timestamp_ms]
    
    def _smooth_pose_data(self, current_pose_data):
        """对姿势数据进行平滑处理"""
        self.frame_count += 1
        
        # 前两帧不做平滑处理
        if self.frame_count <= 2:
            # 保存当前帧数据以供后续平滑使用
            self.last_two_pose_frames.append(current_pose_data)
            if len(self.last_two_pose_frames) > 2:
                self.last_two_pose_frames.pop(0)  # 保持最多两帧
            return current_pose_data
            
        # 第三帧及以后的帧进行平滑处理
        try:
            # 解析当前帧和历史帧的姿势数据
            current_pose_values = [float(x) for x in current_pose_data.decode('utf-8').split(',')]
            
            # 解析历史帧数据
            prev_frames_values = []
            for frame_data in self.last_two_pose_frames:
                prev_frames_values.append([float(x) for x in frame_data.decode('utf-8').split(',')])
            
            # 确保所有帧的数据长度一致
            if all(len(prev_values) == len(current_pose_values) for prev_values in prev_frames_values):
                # 计算平均值
                smoothed_values = []
                for i in range(len(current_pose_values)):
                    # 当前帧和历史帧的权重可以调整，这里使用简单平均
                    avg_value = (current_pose_values[i] + 
                                prev_frames_values[0][i] + 
                                prev_frames_values[1][i]) / 3.0
                    smoothed_values.append(avg_value)
                
                # 更新历史帧缓存
                self.last_two_pose_frames.pop(0)
                self.last_two_pose_frames.append(current_pose_data)
                
                # 转换回字符串格式
                smoothed_str = ','.join(map(str, smoothed_values))
                return smoothed_str.encode('utf-8')
            
        except Exception as e:
            print(f"平滑处理时出错: {e}")
        
        # 如果出现任何问题，返回原始数据
        self.last_two_pose_frames.append(current_pose_data)
        if len(self.last_two_pose_frames) > 2:
            self.last_two_pose_frames.pop(0)
        return current_pose_data
    
    def _cleanup_old_frames(self):
        # 清理旧帧防止内存溢出
        if len(self.frame_data) > self.max_frames:
            oldest_timestamp = min(self.frame_data.keys())
            del self.frame_data[oldest_timestamp]
    
    def set_client(self, client):
        self.client = client

def run_client(client):
    """运行客户端线程"""
    client.run()

def process_pose_result(result, timestamp_ms, frame_data_manager, frame_width, frame_height, debug=False, res_path=None):
    """处理姿势检测结果并更新帧数据管理器"""
    if result.pose_landmarks:
        # 处理姿势数据
        for pose_landmark in result.pose_landmarks:
            # 转换为像素坐标并处理
            landmarks_data = []
            
            # 如果存在上一帧数据，检查当前帧每个点的可见性
            prev_landmarks = frame_data_manager.last_processed_landmarks
            
            for i, landmark in enumerate(pose_landmark):
                # 检查可见性，如果小于0.5且有上一帧数据，则使用上一帧的数据
                if landmark.visibility < 0.5 and prev_landmarks and i < len(prev_landmarks):
                    # 使用上一帧的数据
                    landmarks_data.append(prev_landmarks[i])
                    if debug:
                        print(f"关键点 {i} 可见性低 ({landmark.visibility:.2f})，使用上一帧数据")
                else:
                    # 使用当前帧数据
                    landmarks_data.append((
                        landmark.x * frame_width,  # 转换为像素坐标
                        frame_height - landmark.y * frame_height,  # 转换为像素坐标并反转 y 轴方向
                        landmark.z * frame_width  # 深度信息也乘以帧宽度
                    ))
            
            # 保存处理后的关键点数据，供下一帧使用
            frame_data_manager.last_processed_landmarks = landmarks_data.copy()
            
            # 展平为列表
            flat_landmarks = [coord for landmark in landmarks_data for coord in landmark]
            
            # 转换为逗号分隔的字符串
            landmarks_str = ','.join(map(str, flat_landmarks))
            landmarks_data_bytes = landmarks_str.encode('utf-8')
            
            # 更新帧数据管理器
            frame_data_manager.update_pose_data(timestamp_ms, landmarks_data_bytes)
        
        if debug:
            print(f"姿势数据大小: {len(landmarks_data_bytes)}")
            current_timestamp_ms = int(time.time() * 1000)
            print(f"姿势处理延迟: {current_timestamp_ms - timestamp_ms} ms")

            print("===== 检测到的姿势关键点 (像素坐标) =====")
            for i, (x, y, z) in enumerate(landmarks_data):
                print(f"关键点 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            print("=======================================")

def process_face_result(result, timestamp_ms, frame_data_manager, debug=False, res_path=None):
    """处理面部表情检测结果并更新帧数据管理器"""
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes:
            # 提取所有分数值
            blendshape_data = [category.score for category in blendshape]
            
            # 转换为逗号分隔的字符串
            blendshape_str = ','.join(map(str, blendshape_data))
            blendshape_data_bytes = blendshape_str.encode('utf-8')

            # 保存最近一次有效的面部数据
            frame_data_manager.last_valid_face_data = blendshape_data_bytes

            # 更新帧数据管理器
            frame_data_manager.update_face_data(timestamp_ms, blendshape_data_bytes)
            
            if debug:
                print(f"面部表情数据大小: {len(blendshape_data_bytes)}")
                current_timestamp_ms = int(time.time() * 1000)
                print(f"面部处理延迟: {current_timestamp_ms - timestamp_ms} ms")
                
                print("===== 检测到的面部表情数据 =====")
                for i, score in enumerate(blendshape_data):
                    print(f"表情 {i}: 强度={score:.4f}")
                print("===============================")
    else:
        # 未检测到面部，使用最近一次有效的面部数据
        if frame_data_manager.last_valid_face_data is not None:
            if debug:
                print("未检测到面部，使用上一次有效的面部数据")
            frame_data_manager.update_face_data(timestamp_ms, frame_data_manager.last_valid_face_data)
        else:
            # 新增：从未检测到面部，使用默认值（全0数据）
            if debug:
                print("未检测到面部，且无历史数据，使用默认值")
            # 假设面部表情有52个特征（根据实际模型调整）
            default_face_data = ','.join(['0.0'] * 52)
            default_face_bytes = default_face_data.encode('utf-8')
            frame_data_manager.update_face_data(timestamp_ms, default_face_bytes)

def main(server_addr='127.0.0.1', port_num=50051, 
         pose_model_path="E:\\ztw\\HVCCS\\data\\pose_landmarker_full.task", face_model_path="E:\\ztw\\HVCCS\\data\\face_landmarker_v2_with_blendshapes.task", 
         debug=False, res_path=None):
    """主函数：设置并运行姿势和面部表情检测"""
    VisionRunningMode = mp.tasks.vision.RunningMode

    # 当命令行未传模型路径时，回退到项目内默认路径。
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_pose_model = os.path.join(project_root, 'data', 'pose_landmarker_full.task')
    default_face_model = os.path.join(project_root, 'data', 'face_landmarker_v2_with_blendshapes.task')

    pose_model_path = pose_model_path or default_pose_model
    face_model_path = face_model_path or default_face_model

    if not os.path.exists(pose_model_path):
        raise FileNotFoundError(f"姿势模型文件不存在: {pose_model_path}")
    if not os.path.exists(face_model_path):
        raise FileNotFoundError(f"面部模型文件不存在: {face_model_path}")
    
    # 创建客户端
    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()
    
    # 创建帧数据管理器
    frame_data_manager = FrameDataManager()
    frame_data_manager.set_client(client)
    
    # 创建姿势检测器
    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options, 
        running_mode=VisionRunningMode.LIVE_STREAM, 
        output_segmentation_masks=False,
        result_callback=lambda result, _, timestamp_ms: process_pose_result(
            result, timestamp_ms, frame_data_manager, frame_width, frame_height, debug=debug
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    # 创建面部表情检测器
    face_base_options = python.BaseOptions(model_asset_path=face_model_path)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options, 
        running_mode=VisionRunningMode.LIVE_STREAM, 
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        result_callback=lambda result, _, timestamp_ms: process_face_result(
            result, timestamp_ms, frame_data_manager, debug=debug
        )
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_timestamp_ms = int(time.time() * 1000)
    
    try:
        while cap.isOpened():
            # 从相机捕获一帧图片
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在当前第k帧捕获后、输入姿势估计器前打上t_begin时间戳
            frame_timestamp_ms = int(time.time() * 1000)
            
            # 缓冲区满了就等待
            buffer_size = client.send_data_buffer.get_size()
            while buffer_size >= 5:
                time.sleep(0.01)
                buffer_size = client.send_data_buffer.get_size()
    
            # 将图像从BGR颜色空间转换为RGB颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 为两个检测器分别异步检测
            pose_detector.detect_async(mp_image, int(frame_timestamp_ms))
            face_detector.detect_async(mp_image, int(frame_timestamp_ms))
            
            if debug:
                print(f"帧时间戳: {frame_timestamp_ms}")
            
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('资源已释放')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行姿势和面部表情检测发送器")
    parser.add_argument("--server_addr", type=str, default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port_num", type=int, default=50051, help="端口号")
    parser.add_argument("--pose_model_path", type=str, help="姿势模型路径")
    parser.add_argument("--face_model_path", type=str, help="面部模型路径")
    parser.add_argument("--debug", action='store_true', help="调试模式")
    parser.add_argument("--res_path", type=str, help="结果路径")
    
    args = parser.parse_args()
    
    main(
        server_addr=args.server_addr,
        port_num=args.port_num,
        pose_model_path=args.pose_model_path,
        face_model_path=args.face_model_path,
        debug=args.debug,
        res_path=args.res_path
    )