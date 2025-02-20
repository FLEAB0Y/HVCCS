import cv2
import time
from tqdm import tqdm
import bpy
import os
import shutil
import json
import threading
import grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import data_stream_pb2_grpc
from server import THStreamServiceServicer, serve

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

def setup_render_environment(render_output_path):
    clear_folder(render_output_path)

    # 删除所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 加载 .blend 文件
    bpy.ops.wm.open_mainfile(filepath="/home/ztw/MediaPipe/face_detec/data/boy52blendshapes.blend")
    # 获取导入的对象
    render = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            render = obj
            break

    if render is None:
        raise ValueError("没有找到 MESH 类型的对象")

    # 添加一个相机
    bpy.ops.object.camera_add(location=(0, 2, 1.4))
    camera = bpy.context.active_object
    camera.rotation_euler = (-1.5708, 3.1415926, 0)  # 旋转相机，使其面向模型

    # 设置相机为活动相机
    bpy.context.scene.camera = camera

    # 添加光照
    bpy.ops.object.light_add(type='POINT', location=(0, -3, 3))
    light = bpy.context.active_object
    light.data.energy = 1000  # 设置光照强度

    # 设置渲染引擎为 Eevee
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # 设置渲染采样率
    bpy.context.scene.eevee.taa_render_samples = 1  # 设置 Eevee 的采样率为 1

    return render

def process_face_data(servicer, render, render_output_path, index_to_category_name):
    while True:
        # 缓冲区空了就等待
        buffer_size = servicer.receive_data_buffer.get_size()
        while buffer_size < 1:
            time.sleep(0.1)
            buffer_size = servicer.receive_data_buffer.get_size()
        # 从缓冲区获取数据
        payload_rec = servicer.receive_data_buffer.get_items()
        if payload_rec:
            try:
                face_data_bytes = payload_rec.faceData
                data_list = json.loads(face_data_bytes.decode('utf-8'))  # 将接收到的 JSON 数据转换为列表
            except AttributeError as e:
                print(f"AttributeError: {e}")
            
            j = 0 # 计数器，检查 blendshape 数量齐全
            for index, score in data_list:
                category_name = index_to_category_name.get(index)
                if category_name and render.data.shape_keys:
                    shape_keys = render.data.shape_keys.key_blocks
                    if category_name in shape_keys:
                        shape_keys[category_name].value = score
                        j += 1
                else:
                    print("该对象没有 shape keys")
            # 检查 blendshape 数量是否齐全
            if j != 52:
                print("blendshape数量缺失")
            else:
                # 渲染当前帧
                bpy.context.scene.render.filepath = os.path.join(render_output_path, f"render_{str(payload_rec.extDesc).zfill(5)}.png")
                bpy.ops.render.render(write_still=True)
        time.sleep(1./30.)

def main():
    # 设置渲染输出文件的路径
    render_output_path = "/home/ztw/MediaPipe/face_detec/res/render_res"
    render = setup_render_environment(render_output_path)

    # 开启服务器线程
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer,))
    server_thread.start()

    # 定义 index 和 category_name 的匹配表
    index_to_category_name = {
        0: "_neutral",
        1: "browDownLeft",
        2: "browDownRight",
        3: "browInnerUp",
        4: "browOuterUpLeft",
        5: "browOuterUpRight",
        6: "cheekPuff",
        7: "cheekSquintLeft",
        8: "cheekSquintRight",
        9: "eyeBlinkLeft",
        10: "eyeBlinkRight",
        11: "eyeLookDownLeft",
        12: "eyeLookDownRight",
        13: "eyeLookInLeft",
        14: "eyeLookInRight",
        15: "eyeLookOutLeft",
        16: "eyeLookOutRight",
        17: "eyeLookUpLeft",
        18: "eyeLookUpRight",
        19: "eyeSquintLeft",
        20: "eyeSquintRight",
        21: "eyeWideLeft",
        22: "eyeWideRight",
        23: "jawForward",
        24: "jawLeft",
        25: "jawOpen",
        26: "jawRight",
        27: "mouthClose",
        28: "mouthDimpleLeft",
        29: "mouthDimpleRight",
        30: "mouthFrownLeft",
        31: "mouthFrownRight",
        32: "mouthFunnel",
        33: "mouthLeft",
        34: "mouthLowerDownLeft",
        35: "mouthLowerDownRight",
        36: "mouthPressLeft",
        37: "mouthPressRight",
        38: "mouthPucker",
        39: "mouthRight",
        40: "mouthRollLower",
        41: "mouthRollUpper",
        42: "mouthShrugLower",
        43: "mouthShrugUpper",
        44: "mouthSmileLeft",
        45: "mouthSmileRight",
        46: "mouthStretchLeft",
        47: "mouthStretchRight",
        48: "mouthUpperUpLeft",
        49: "mouthUpperUpRight",
        50: "noseSneerLeft",
        51: "noseSneerRight"
    }

    process_face_data(servicer, render, render_output_path, index_to_category_name)

if __name__ == "__main__":
    main()

