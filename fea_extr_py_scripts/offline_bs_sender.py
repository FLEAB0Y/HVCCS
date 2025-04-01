# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
import os
import shutil
import json
import threading
from client import THStreamClient
from THStreamData import THStreamDataPayload, THDataWarehouse

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def images_to_video(image_folder, video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # 按名称排序

    if not images:
        print("No images found in the folder.")
        return

    # 获取第一张图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved at {video_path}")

def run_client(client):
    client.run()

if __name__ == "__main__":

    # 设置文件路径
    input_video_path = "../data/intro.MOV"
    face_landmarks_output_path = "../res/detec_res"
    clear_folder(face_landmarks_output_path)

    # 创建FaceLandmarker对象.
    base_options = python.BaseOptions(model_asset_path='../data/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(running_mode=vision.RunningMode.VIDEO,
                                           base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # 加载视频
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) # 计算帧率

    # 从输入视频中检测facelandmarks
    i = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建并启动客户端线程
    client = THStreamClient(host='127.0.0.1', port=50051)
    client_thread = threading.Thread(target=run_client, args=(client,))
    client_thread.start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 缓冲区满了就等待
        buffer_size = client.send_data_buffer.get_size()
        while buffer_size >= 10:
            time.sleep(0.1)
            buffer_size = client.send_data_buffer.get_size()

        start_time = time.time()
        # Convert the frame to a MediaPipe Image object.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Calculate the timestamp for the current frame.
        frame_timestamp_ms = int(i * (1000 / fps))
        i = i + 1
        # Detect face landmarks from the frame.
        detection_result = detector.detect_for_video(mp_image,frame_timestamp_ms)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # 将facelandmarks绘制到图像上
        # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        # converted_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(face_landmarks_output_path + "/detect_output_" + str(f"{i:05d}") + ".png", converted_image)
        # 发送blendshapes
        if detection_result.face_blendshapes:
            for blendshape in detection_result.face_blendshapes:
                # test 输出category_name和index的一一对应关系
                # for category in blendshape:
                
                #     print(f"Category Name: {category.category_name}, Index: {category.index}")
                blendshape_data = [(category.index, category.score) for category in blendshape] # 获取blendshape的index和score组成列表
                # test
                # print(f"Blendshape Data: {blendshape_data}")
                # face blendshape 编码 （TBD）
                
                # 将blendshape_data转换为字节
                blendshape_data_json = json.dumps(blendshape_data)
                blendshape_data_bytes = blendshape_data_json.encode('utf-8')
                # 往缓冲区放入数据
                payload_send = THStreamDataPayload(
                    rgb_data=b'\x01', 
                    point_data=b'\x02', 
                    face_data=blendshape_data_bytes, 
                    limb_data=b'\x04',    
                    ext_data=b'\x05', 
                    ext_desc=f"{str(i)}"
                )
                
                client.send_data_buffer.add_item(payload_send)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()