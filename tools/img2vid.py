import os
import cv2
import shutil

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
        print("image_name:", image)
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved at {video_path}")


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


if __name__ == "__main__":
    output_fps = 15

    if output_fps <= 0:
        raise ValueError("fps must be greater than 0")

    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # 上一级目录 (HVCCS)
    
    output_video_path = os.path.join(base_dir, 'res', 'video_res')
    clear_folder(output_video_path)
    
    input_path1 = os.path.join(base_dir, 'res', 'splines_encoder')
    input_path2 = os.path.join(base_dir, 'res', 'splines_decoder')
    

    images_to_video(input_path1, os.path.join(output_video_path, "splines_encoder.mp4"), fps=output_fps)
    images_to_video(input_path2, os.path.join(output_video_path, "splines_decoder.mp4"), fps=output_fps)
