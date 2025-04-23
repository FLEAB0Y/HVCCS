# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import time
import json
import threading
import shutil
from client import THStreamClient
from THStreamData import THStreamDataPayload, THDataWarehouse
import os

def run_client(client):
    client.run()

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def print_pose_result_info(result: mp.tasks.vision.PoseLandmarkerResult):
    """调试函数：打印姿势检测结果的详细信息"""
    # 1. 输出 pose_landmarks 信息（2D 姿势关键点）
    print("\n===== pose_landmarks (2D 姿势关键点) =====")
    if result.pose_landmarks:
        print(f"类型: {type(result.pose_landmarks)}")
        print(f"列表长度: {len(result.pose_landmarks)} (检测到的人数)")
        
        # 第一个检测到的人
        if len(result.pose_landmarks) > 0:
            print(f"每人关键点数量: {len(result.pose_landmarks[0])}")
            print("关键点格式: 包含 x, y, z (相对深度), visibility, presence 属性")
            print("示例关键点(第一个人的第一个点):")
            landmark = result.pose_landmarks[0][0]
            print(f"  x: {landmark.x} (归一化坐标: 0~1)")
            print(f"  y: {landmark.y} (归一化坐标: 0~1)")
            print(f"  z: {landmark.z} (相对深度)")
            print(f"  visibility: {landmark.visibility} (可见度: 0~1)")
            print(f"  presence: {landmark.presence} (存在概率: 0~1)")
    else:
        print("未检测到姿势关键点")
    
    # 2. 输出 pose_world_landmarks 信息（3D 世界坐标系中的姿势关键点）
    print("\n===== pose_world_landmarks (3D 世界坐标系姿势关键点) =====")
    if result.pose_world_landmarks:
        print(f"类型: {type(result.pose_world_landmarks)}")
        print(f"列表长度: {len(result.pose_world_landmarks)} (检测到的人数)")
        
        # 第一个检测到的人
        if len(result.pose_world_landmarks) > 0:
            print(f"每人关键点数量: {len(result.pose_world_landmarks[0])}")
            print("关键点格式: 包含 x, y, z (米为单位), visibility, presence 属性")
            print("示例关键点(第一个人的第一个点):")
            landmark = result.pose_world_landmarks[0][0]
            print(f"  x: {landmark.x} (米)")
            print(f"  y: {landmark.y} (米)")
            print(f"  z: {landmark.z} (米)")
            print(f"  visibility: {landmark.visibility} (可见度: 0~1)")
            print(f"  presence: {landmark.presence} (存在概率: 0~1)")
    else:
        print("未检测到3D世界坐标系姿势关键点")
    
    # 3. 输出 segmentation_masks 信息（分割蒙版）
    print("\n===== segmentation_masks (分割蒙版) =====")
    if result.segmentation_masks:
        print(f"类型: {type(result.segmentation_masks)}")
        print(f"蒙版数量: {len(result.segmentation_masks)}")
        
        # 第一个蒙版
        if len(result.segmentation_masks) > 0:
            mask = result.segmentation_masks[0].numpy_view()
            print(f"蒙版形状: {mask.shape} (高度 x 宽度)")
            print(f"数据类型: {mask.dtype}")
            print(f"值范围: {mask.min()} ~ {mask.max()} (通常0~1，表示像素属于人体的概率)")
    else:
        print("未生成分割蒙版")

def detect_result_proc(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int, client: THStreamClient, debug=False, res_path=None):
    """处理姿势检测结果并发送数据"""
    
    # 绘制关键点
    if result.pose_landmarks:
        
        if debug:
            # print_pose_result_info(result)
            # 绘制姿势关键点
            annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
        
            # 保存图像到指定目录
            image_path = os.path.join(res_path, f"pose_{timestamp_ms}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"已保存姿势检测图像: {image_path}")
        
        # 处理并发送数据
        for pose_landmark in result.pose_landmarks:
            landmarks_data = [(landmark.x, landmark.y, landmark.z, landmark.visibility) 
                             for landmark in pose_landmark]
            
            landmarks_data_json = json.dumps(landmarks_data)
            landmarks_data_bytes = landmarks_data_json.encode('utf-8')
            
            payload_send = THStreamDataPayload(
                rgb_data=b'\x01', 
                point_data=b'\x02',
                face_data=b'\x03',
                limb_data=landmarks_data_bytes,    
                ext_data=b'\x05', 
                ext_desc=f"{str(timestamp_ms)}"
            )
            
            client.send_data_buffer.add_item(payload_send)

def main(server_addr='127.0.0.1', port_num=50051, 
         model_path='/HVCCS/data/pose_landmarker_heavy.task', 
         debug=False,
         res_path='../res/imgs'):
    # 创建姿势检测器
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, 
        running_mode=VisionRunningMode.LIVE_STREAM, 
        output_segmentation_masks=False,
        result_callback=lambda result, output_image, timestamp_ms: detect_result_proc(
            result, output_image, timestamp_ms, client, debug=debug, res_path=res_path
        )
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = int(time.time() * 1000)

    # 创建并启动客户端线程
    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()
    
    try:
        while cap.isOpened():
            # 从相机从捕获一帧图片
            ret, frame = cap.read()
            if not ret:
                break
            
            # 缓冲区满了就等待
            buffer_size = client.send_data_buffer.get_size()
            while buffer_size >= 10:
                time.sleep(0.1)
                buffer_size = client.send_data_buffer.get_size()
    
            # 将图像从BGR颜色空间转换为RGB颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # 异步检测
            detector.detect_async(mp_image, int(frame_timestamp_ms))
            
            print(f"frame_timestamp_ms: {frame_timestamp_ms}")
            frame_timestamp_ms = int(time.time() * 1000)
            
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('资源已释放')

if __name__ == "__main__":
    # 4090 server ip addr = 183.173.48.193
    # A100 server ip addr = 101.6.65.237
    # laptop ip addr = 183.173.115.89
    # self ip addr = 127.0.0.1
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径 - 上一级目录的data文件夹
    model_path = os.path.join(script_dir, "..", "data", "pose_landmarker_heavy.task")
    res_path = os.path.join(script_dir, "..", "res", "imgs")
    # 确保输出目录存在
    os.makedirs(res_path, exist_ok=True)
    clear_folder(res_path)
    
    main(server_addr='127.0.0.1', 
         port_num=50051, 
         model_path = model_path, 
         debug=False,
         res_path=res_path)
