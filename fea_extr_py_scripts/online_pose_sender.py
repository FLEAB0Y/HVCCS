#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
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
import os

def run_client(client):
    client.run()

def cv2_imshow(image):
  """Display an image using OpenCV."""
  cv2.imshow('Image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

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

if __name__ == '__main__':
  

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='/Users/twz/demo_sys_user/HVCCS/data/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(running_mode=vision.RunningMode.IMAGE,
                                           base_options=base_options,
                                           output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)   

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("/Users/twz/demo_sys_user/HVCCS/data/pose.jpg")  

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)   

    # STEP 5: Process the detection result. In this case, visualize it.
    # 将 RGB 图像转换为 BGR 格式
    bgr_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    # 传入 BGR 格式图像
    annotated_image = draw_landmarks_on_image(bgr_image, detection_result)
    # 直接显示，无需再次转换
    cv2_imshow(annotated_image)

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2_imshow(visualized_mask)