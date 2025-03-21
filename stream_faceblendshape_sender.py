# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
import json
import threading
from client import THStreamClient
from THStreamData import THStreamDataPayload, THDataWarehouse

def run_client(client):
    client.run()

def detect_result_proc(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int, client: THStreamClient):
    # print('face landmarker result: {}'.format(result))
    # print('face blendshape result: {}'.format(result.face_blendshapes))
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes:
            # 输出category_name和index的一一对应关系
            # for category in blendshape:
            #     print(f"Category Name: {category.category_name}, Index: {category.index}")
            blendshape_data = [(category.index, category.score) for category in blendshape] # 获取blendshape的index和score组成列表
            
            # face blendshape 编码 （TBD）
            
            # 推一路摄像机视频流做对比

            # 将blendshape_data转换为字节
            blendshape_data_json = json.dumps(blendshape_data)
            blendshape_data_bytes = blendshape_data_json.encode('utf-8')

            # test
            print(f"blendshape_data: {blendshape_data}")

            # 往缓冲区放入数据
            payload_send = THStreamDataPayload(
                rgb_data=b'\x01', 
                point_data=b'\x02', 
                face_data=blendshape_data_bytes, 
                limb_data=b'\x04',    
                ext_data=b'\x05', 
                ext_desc=f"{str(timestamp_ms)}"
            )
            
            client.send_data_buffer.add_item(payload_send)

def main(server_addr='127.0.0.1', port_num =50051, model_path='/home/ztw/HVCCS/data/face_landmarker_v2_with_blendshapes.task'):
    # 创建人脸检测器
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                           running_mode = VisionRunningMode.LIVE_STREAM, 
                                           output_face_blendshapes = True,
                                           output_facial_transformation_matrixes=True,
                                           result_callback=lambda result, output_image, timestamp_ms: detect_result_proc(result, output_image, timestamp_ms, client)
                                           )
    detector = vision.FaceLandmarker.create_from_options(options)

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
    # self ip addr = 127.0.0.1
    main(server_addr='127.0.0.1', port_num=50051, model_path='/home/ztw/HVCCS/data/face_landmarker_v2_with_blendshapes.task')
