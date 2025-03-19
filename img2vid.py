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
    output_video_path = '/home/ztw/HVCCS/res/video_res'
    clear_folder(output_video_path)
    face_landmarks_path = "/home/ztw/HVCCS/res/detec_res"
    clear_folder(face_landmarks_path)
    render_res_path = "res/render_res"
    # 将facelandmarks图像拼接为视频
    images_to_video(face_landmarks_path, output_video_path + "/detect_output.mp4")
    # 将render res拼接为视频
    images_to_video(render_res_path, output_video_path + "/render_output.mp4")
