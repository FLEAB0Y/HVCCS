# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
import shutil
from tqdm import tqdm

def clear_folder(folder_path):
    """清空并重新创建指定文件夹"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def create_face_landmarker(model_path, running_mode=vision.RunningMode.VIDEO, num_faces=1):
    """创建并返回FaceLandmarker对象"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        running_mode=running_mode,
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=num_faces
    )
    return vision.FaceLandmarker.create_from_options(options)

def create_pose_landmarker(model_path, running_mode=vision.RunningMode.VIDEO):
    """创建并返回PoseLandmarker对象"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)

def init_output_file(output_path):
    """初始化输出文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        pass  # 仅创建空文件

def save_blendshapes(detection_result, output_path):
    """将blendshapes数据保存到文件"""
    if detection_result.face_blendshapes:
        for blendshape in detection_result.face_blendshapes:
            # 创建一个包含52个元素的列表，初始化为0
            blendshape_values = [0.0] * 52
            
            # 填充获取到的blendshape值
            for category in blendshape:
                blendshape_values[category.index] = category.score
            
            # 将数据写入文件，每行一帧，逗号分隔，行末也有逗号
            with open(output_path, 'a') as f:
                f.write(','.join([f"{value:.6f}" for value in blendshape_values]) + ',\n')
            
            # 只处理第一个面部的blendshapes（因为num_faces=1）
            break

def save_pose_landmarks(detection_result, output_path, frame_width, frame_height):
    """将姿势关键点数据保存到文件"""
    if detection_result.pose_landmarks:
        for pose_landmark in detection_result.pose_landmarks:
            # 处理关键点数据
            landmarks_data = []
            
            for landmark in pose_landmark:
                # 转换为像素坐标并保持一致的坐标系
                landmarks_data.append((
                    landmark.x * frame_width,  # 转换为像素坐标
                    frame_height - landmark.y * frame_height,  # 转换为像素坐标并反转y轴方向
                    landmark.z * frame_width  # 深度信息也乘以帧宽度
                ))
            
            # 展平为一维列表
            flat_landmarks = [coord for landmark in landmarks_data for coord in landmark]
            
            # 将数据写入文件，每行一帧，逗号分隔
            with open(output_path, 'a') as f:
                f.write(','.join([f"{value:.6f}" for value in flat_landmarks]) + ',\n')
            
            # 只处理第一个检测到的姿势
            break

def save_combined_features(face_result, pose_result, output_path, frame_width, frame_height):
    """将面部表情和姿势关键点数据合并保存到文件"""
    # 默认所有特征都是0
    blendshape_values = [0.0] * 52  # 面部表情特征
    pose_values = [0.0] * 99  # 姿势特征 (33个关键点 x 3个坐标)
    
    # 提取面部表情特征
    if face_result.face_blendshapes:
        for blendshape in face_result.face_blendshapes:
            for category in blendshape:
                blendshape_values[category.index] = category.score
            break  # 只处理第一个面部
    
    # 提取姿势特征
    if pose_result.pose_landmarks:
        for pose_landmark in pose_result.pose_landmarks:
            landmarks_data = []
            
            for landmark in pose_landmark:
                # 转换为像素坐标并保持一致的坐标系
                landmarks_data.append((
                    landmark.x * frame_width,  # 转换为像素坐标
                    frame_height - landmark.y * frame_height,  # 转换为像素坐标并反转y轴方向
                    landmark.z * frame_width  # 深度信息也乘以帧宽度
                ))
            
            # 展平为一维列表
            flat_landmarks = [coord for landmark in landmarks_data for coord in landmark]
            
            # 更新姿势数据
            pose_values = flat_landmarks
            break  # 只处理第一个姿势
    
    # 合并特征并写入文件
    combined_values = blendshape_values + pose_values
    with open(output_path, 'a') as f:
        f.write(','.join([f"{value:.6f}" for value in combined_values]) + ',\n')

def process_video(video_path, face_detector, pose_detector, output_path):
    """处理视频并提取合并的面部和姿势特征"""
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 计算帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_index = 0
    # 跟踪上一个时间戳，确保递增
    last_timestamp_ms = -1
    
    # 创建帧处理进度条
    pbar = tqdm(total=total_frames, desc="处理帧", unit="帧")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为MediaPipe Image对象
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 计算当前帧的时间戳，确保严格递增
        frame_timestamp_ms = int(frame_index * (1000 / fps))
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1  # 确保比上一个时间戳大
        
        last_timestamp_ms = frame_timestamp_ms
        frame_index += 1
        
        # 从帧中检测面部特征
        face_result = face_detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        # 从帧中检测姿势特征
        pose_result = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        # 保存合并的特征数据到文件
        save_combined_features(face_result, pose_result, output_path, frame_width, frame_height)
        
        # 更新进度条
        pbar.update(1)
                
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # 关闭进度条
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"处理完成: {frame_index}/{total_frames} 帧")

if __name__ == "__main__":
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 基于脚本目录设置路径
    videos_dir = os.path.join(script_dir, "..", "videos")
    face_model_path = os.path.join(script_dir, "..", "..", "data", "face_landmarker_v2_with_blendshapes.task")
    pose_model_path = os.path.join(script_dir, "..", "..", "data", "pose_landmarker_full.task")
    
    # 确保路径为绝对路径
    videos_dir = os.path.abspath(videos_dir)
    face_model_path = os.path.abspath(face_model_path)
    pose_model_path = os.path.abspath(pose_model_path)
    
    print(f"面部模型路径: {face_model_path}")
    print(f"姿势模型路径: {pose_model_path}")
    print(f"视频目录: {videos_dir}")
    
    # 创建检测器对象
    face_detector = create_face_landmarker(face_model_path)
    pose_detector = create_pose_landmarker(pose_model_path)
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.mkv', '.wmv']
    
    # 获取videos目录下所有视频文件
    video_files = [f for f in os.listdir(videos_dir) 
                  if os.path.isfile(os.path.join(videos_dir, f)) 
                  and os.path.splitext(f)[1].lower() in [ext.lower() for ext in video_extensions]]
    
    if not video_files:
        print("未找到视频文件！")
        exit()
    
    print(f"找到{len(video_files)}个视频文件，开始处理...")
    
    # 创建特征输出目录
    features_dir = os.path.join(script_dir, "..", "ori_features")
    
    os.makedirs(features_dir, exist_ok=True)
    
    # 使用tqdm添加视频处理进度条
    for video_file in tqdm(video_files, desc="处理视频", unit="个"):
        input_video_path = os.path.join(videos_dir, video_file)
        
        # 获取视频文件名（不带扩展名）用于生成输出文件名
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        
        # 使用绝对路径创建输出路径
        output_path = os.path.join(features_dir, f"{video_name}.txt")
        output_path = os.path.abspath(output_path)
        
        print(f"正在处理视频: {video_file}")
        
        # 初始化输出文件
        init_output_file(output_path)
        
        # 为每个视频重新创建检测器对象，确保内部状态被重置
        face_detector = create_face_landmarker(face_model_path)
        pose_detector = create_pose_landmarker(pose_model_path)
        
        # 处理视频并保存结果
        process_video(input_video_path, face_detector, pose_detector, output_path)
        
        print(f"视频 {video_file} 的特征数据已保存:")
        print(f"合并特征: {output_path}")
    
    print("所有视频处理完成！")