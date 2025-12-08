import cv2
import numpy as np

def main():
    video_path = "videos/id08.mp4"
    frame_num1 = 38
    frame_num2 = 39
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 读取第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num1)
    ret1, frame1 = cap.read()
    if not ret1:
        print("无法读取帧", frame_num1)
        cap.release()
        return
    
    # 读取第二帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num2)
    ret2, frame2 = cap.read()
    if not ret2:
        print("无法读取帧", frame_num2)
        cap.release()
        return
    
    cap.release()
    
    # 计算残差图
    residual = cv2.absdiff(frame1, frame2)
    
    # 保存图像
    cv2.imwrite('frame1.png', frame1)
    cv2.imwrite('frame2.png', frame2)
    cv2.imwrite('residual.png', residual)
    
    print("图像已保存: frame1.png, frame2.png, residual.png")

if __name__ == "__main__":
    main()