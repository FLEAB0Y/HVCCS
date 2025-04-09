# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import socket  # 只使用socket模块

def send_blendshape_data(data_list):
    """使用socket直接发送blendshape数据"""
    # 只格式化索引和值，不添加其他文字
    data_str = ";".join([f"{idx},{val}" for idx, val in data_list])
    # print(f"发送数据: {data_str}")
    
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 8889))
    
    # 发送数据
    client.send(data_str.encode('utf-8'))
    
    # 关闭连接
    client.close()

def detect_result_proc(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('face landmarker result: {}'.format(result))
    # print('face blendshape result: {}'.format(result.face_blendshapes))
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes:
            # test 输出category_name和index的一一对应关系
            # for category in blendshape:
            #     print(f"Category Name: {category.category_name}, Index: {category.index}")

            blendshape_data = [(category.index, category.score) for category in blendshape] # 获取blendshape的index和score组成列表
            
            # 使用socket发送方法
            try:
                send_blendshape_data(blendshape_data)
            except Exception as e:
                print(f"发送数据失败: {e}")
            
            # test
            # print(f"blendshape_data: {blendshape_data}")

def main(model_path='HVCCS/data/face_landmarker_v2_with_blendshapes.task'):
    # 创建人脸检测器
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                          running_mode = VisionRunningMode.LIVE_STREAM, 
                                          output_face_blendshapes = True,
                                          output_facial_transformation_matrixes=True,
                                          result_callback=lambda result, output_image, timestamp_ms: detect_result_proc(result, output_image, timestamp_ms)
                                          )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = int(time.time() * 1000)
    
    try:
        while cap.isOpened():
            # 从相机从捕获一帧图片
            ret, frame = cap.read()
            if not ret:
                break
            
            # 将图像从BGR颜色空间转换为RGB颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # 异步检测
            detector.detect_async(mp_image, int(frame_timestamp_ms))
            
            # print(f"frame_timestamp_ms: {frame_timestamp_ms}")
            frame_timestamp_ms = int(time.time() * 1000)
            
    except KeyboardInterrupt:
        print('程序被用户中断')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('资源已释放')

if __name__ == "__main__":
    main(model_path='D:/HVCCS/data/face_landmarker_v2_with_blendshapes.task')
