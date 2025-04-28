# 整合姿势和面部表情识别的发送器
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
    """运行客户端线程"""
    client.run()

def clear_folder(folder_path):
    """清空并重新创建文件夹"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def draw_landmarks_on_image(rgb_image, detection_result):
    """在图像上绘制姿势关键点"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # 遍历检测到的姿势并可视化
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # 绘制姿势关键点
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

def process_pose_result(result, output_image, timestamp_ms, client, debug=False, res_path=None):
    """处理姿势检测结果并发送数据"""
    if result.pose_landmarks:
        # 处理并发送姿势数据
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
        
        if debug:
            print(f"姿势数据大小: {len(landmarks_data_bytes)}")
            current_timestamp_ms = int(time.time() * 1000)
            print(f"姿势处理延迟: {current_timestamp_ms - timestamp_ms} ms")

            print("===== 检测到的姿势关键点 =====")
            for i, landmark in enumerate(landmarks_data):
                print(f"关键点 {i}: x={landmark[0]}, y={landmark[1]}, z={landmark[2]}, 可见性={landmark[3]}")
            print("=============================")
            
            # 如果需要生成可视化图像
            if res_path:
                annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
                image_path = os.path.join(res_path, f"pose_{timestamp_ms}.jpg")
                cv2.imwrite(image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                print(f"已保存姿势检测图像: {image_path}")

def process_face_result(result, output_image, timestamp_ms, client, debug=False):
    """处理面部表情检测结果并发送数据"""
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes:
            blendshape_data = [(category.index, category.score) for category in blendshape]
            
            blendshape_data_json = json.dumps(blendshape_data)
            blendshape_data_bytes = blendshape_data_json.encode('utf-8')

            payload_send = THStreamDataPayload(
                rgb_data=b'\x01', 
                point_data=b'\x02', 
                face_data=blendshape_data_bytes, 
                limb_data=b'\x04',    
                ext_data=b'\x05', 
                ext_desc=f"{str(timestamp_ms)}"
            )
            
            client.send_data_buffer.add_item(payload_send)
            
            if debug:
                print(f"面部表情数据大小: {len(blendshape_data_bytes)}")
                current_timestamp_ms = int(time.time() * 1000)
                print(f"面部处理延迟: {current_timestamp_ms - timestamp_ms} ms")

def main(server_addr='127.0.0.1', port_num=50051, 
         pose_model_path=None, face_model_path=None, 
         debug=False, res_path=None):
    """主函数：设置并运行姿势和面部表情检测"""
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # 创建客户端
    client = THStreamClient(host=server_addr, port=port_num)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()
    
    # 创建姿势检测器
    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options, 
        running_mode=VisionRunningMode.LIVE_STREAM, 
        output_segmentation_masks=False,
        result_callback=lambda result, output_image, timestamp_ms: process_pose_result(
            result, output_image, timestamp_ms, client, debug=debug, res_path=res_path
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
        result_callback=lambda result, output_image, timestamp_ms: process_face_result(
            result, output_image, timestamp_ms, client, debug=debug
        )
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = int(time.time() * 1000)
    
    try:
        while cap.isOpened():
            # 从相机捕获一帧图片
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
            
            # 为两个检测器分别异步检测
            pose_detector.detect_async(mp_image, int(frame_timestamp_ms))
            face_detector.detect_async(mp_image, int(frame_timestamp_ms))
            
            if debug:
                print(f"帧时间戳: {frame_timestamp_ms}")
            frame_timestamp_ms = int(time.time() * 1000)
            
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('资源已释放')

if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建模型路径
    pose_model_path = os.path.join(script_dir, "..", "data", "pose_landmarker_full.task")
    face_model_path = os.path.join(script_dir, "..", "data", "face_landmarker_v2_with_blendshapes.task")
    
    # 构建结果输出路径
    res_path = os.path.join(script_dir, "..", "res", "avatar_detec_res")
    
    # 确保输出目录存在
    os.makedirs(res_path, exist_ok=True)
    clear_folder(res_path)
    
    main(
        server_addr='127.0.0.1',
        port_num=50051, 
        pose_model_path=pose_model_path,
        face_model_path=face_model_path,
        debug=True,
        res_path=res_path
    )