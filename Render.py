import time
import bpy
import os
import shutil
import json
import threading
import grpc
from THStreamData import THStreamDataPayload, THDataWarehouse
import data_stream_pb2_grpc
from server import THStreamServiceServicer, serve
import math

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def calculate_camera_position(initial_angle, angular_velocity, run_time): # run_time = frames / fps
    # 计算新的角度
    new_angle = initial_angle + angular_velocity * run_time
    
    # 圆环的半径为4
    radius = 4
    
    # 计算新的位置
    x = radius * math.cos(new_angle)
    y = radius * math.sin(new_angle)
    z = 2.6 # 和人物身高对齐
    rx = math.radians(83) # 保持画面滚转角
    ry = 0 # 保持画面俯仰角
    rz = new_angle + math.pi / 2 # 对准（0，0，z）轴
    position = (x, y, z)
    rotation = (rx, ry, rz)
    return position, rotation

def skelton_sim_data_gen(t, rv):
    """
    计算正弦函数值，值域限制在[-0.1,0.1]之间
    
    参数:
        t: 时间
        rv: 角速度
    
    返回:
        值域在[-0.1,0.1]之间的正弦值
    """
    amplitude = 0.05  # 振幅设为0.05，使值域在[-0.05,0.05]
    return amplitude * math.sin(rv * t) + math.radians(-25)

class RenderParameters:
    def __init__(self, camera_location, camera_rotation, light_location, light_energy, render_engine, render_samples):
        self.camera_location = camera_location
        self.camera_rotation = camera_rotation
        self.light_location = light_location
        self.light_energy = light_energy
        self.render_engine = render_engine
        self.render_samples = render_samples

def setup_render_environment(render_output_path, Avatar_path, render_params):
    clear_folder(render_output_path)

    # 加载 .blend 文件
    bpy.ops.wm.open_mainfile(filepath=Avatar_path)

    face_obj = None
    # 打印场景中的所有对象
    print("场景中的所有对象:")
    for obj in bpy.data.objects:
        if obj:
            if obj.type == 'MESH' and obj.data.shape_keys:
                print(f"发现shapekeys在对象 '{obj.name}':")
                for kb in obj.data.shape_keys.key_blocks:
                    print(f"  - {kb.name} (值: {kb.value})")
                face_obj = obj
                break
            else:
                print(f"对象 '{obj.name}' 没有shapekeys或不是Mesh类型")
        else:
            print("场景中未找到对象")
        
    # 寻找骨架对象
    skelton_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            skelton_obj = obj
            print(f"找到骨架: {skelton_obj.name}")
            break
        else:
            print(f"未找到'ARMATURE'对象")
    
    # 找到相机，如果没有相机则创建一个
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    
    if not camera:
        # 创建新相机
        bpy.ops.object.camera_add(location=(0, -4, 2), rotation=(3.14, 0, 0))
        camera = bpy.context.object
    
    # 设置相机为活动相机
    # bpy.context.scene.camera = camera  

    # 添加光照
    bpy.context.view_layer.objects.active = None  # 确保没有活动对象
    bpy.ops.object.light_add(type='POINT', location=render_params.light_location)
    light = bpy.context.active_object
    light.data.energy = render_params.light_energy  # 设置光照强度
                  
    # 设置渲染引擎为 Eevee
    bpy.context.scene.render.engine = render_params.render_engine
    if render_params.render_engine == 'CYCLES':
        bpy.context.scene.cycles.device = 'GPU'  # 使用GPU加速
    # bpy.context.scene.render.image_settings.file_format = 'PNG'  # 设置输出文件格式为 PNG

    # 设置渲染采样率
    bpy.context.scene.eevee.taa_render_samples = render_params.render_samples  # 设置 Eevee 的采样率为 1

    return face_obj, skelton_obj, camera

def process_face_data(servicer, face_obj, skelton_obj, camera, fps,
                      render_output_path, index_to_category_name, 
                      render_params, cam_r0, cam_rv):
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
                #  test 打印接收到的数据列表
                # print(f"接收到的数据列表 (frame {payload_rec.extDesc}):")
                # print(f"数据列表长度: {len(data_list)}")
                # print(data_list[:10] + ['...'] if len(data_list) > 10 else data_list)
            except AttributeError as e:
                print(f"AttributeError: {e}")
            
            j = 0 # 计数器，检查 blendshape 数量齐全
            
            for index, score in data_list:
                category_name = index_to_category_name.get(index)
                if category_name and face_obj.data.shape_keys:
                    shape_keys = face_obj.data.shape_keys.key_blocks
                    if category_name in shape_keys:
                        shape_keys[category_name].value = score
                        j += 1
                else:
                    print("该对象没有 shape keys")
            # 检查 blendshape 数量是否齐全
            if j != 52:
                print(f"blendshape数量缺失: 只应用了 {j}/52")
            
            # test 打印当前所有shapekey的值作为参考
            # print(f"渲染前的shapekey状态 (frame {payload_rec.extDesc}):")
            # if render.data.shape_keys:
            #     for kb in render.data.shape_keys.key_blocks:
            #         print(f"  - {kb.name}: {kb.value:.4f}")
            
            # 计算当前帧的运行时间
            run_time = int(payload_rec.extDesc) / float(fps)
            neck_rotation = (skelton_sim_data_gen(run_time, 1), 0 ,0) # 生成一个正弦函数值

            # 设置骨骼动画
            if skelton_obj:
                # 确保骨架在姿态模式下
                bpy.context.view_layer.objects.active = skelton_obj
                bpy.ops.object.mode_set(mode='POSE')

                # 访问具体的骨骼并修改它们的参数
                pose_bones = skelton_obj.pose.bones

                # 定义需要修改的骨骼路径
                bone_path = ['Ctrl_Master', 'Ctrl_Hips', 'Ctrl_Spine', 'Ctrl_Spine1', 'Ctrl_Spine2', 'Ctrl_Neck']

                for bone_name in bone_path:
                    if bone_name in pose_bones:
                        bone = pose_bones[bone_name]
                        print(f"修改骨骼: {bone_name}")

                        # 根据不同骨骼设置不同的参数
                        if bone_name == 'Ctrl_Master':
                            # 整体位置控制
                            bone.location = (0, 0, 0)
                        elif bone_name == 'Ctrl_Hips':
                            # 臀部位置微调
                            bone.location.y = 0.05
                            bone.rotation_euler = (0.1, 0, 0)
                        elif bone_name == 'Ctrl_Spine':
                            # 脊柱底部旋转
                            bone.rotation_euler = (0.05, 0, 0)
                        elif bone_name == 'Ctrl_Spine1':
                            # 脊柱中部旋转
                            bone.rotation_euler = (0.08, 0, 0)
                        elif bone_name == 'Ctrl_Spine2':
                            # 脊柱上部旋转
                            bone.rotation_euler = (0.1, 0, 0)
                        elif bone_name == 'Ctrl_Neck':
                            # 颈部旋转 - 微微抬头
                            bone.rotation_euler = neck_rotation
                    else:
                        print(f"未找到骨骼: {bone_name}")

                # 返回对象模式
                bpy.ops.object.mode_set(mode='OBJECT')
            else:
                print("场景中未找到骨架对象，无法修改骨骼参数")

            # 修改相机位置到指定坐标
            print(f"run_time: {run_time}")
            camera.location, camera.rotation_euler = calculate_camera_position(cam_r0, cam_rv, run_time) # 丢包模式下修改runtime逻辑（TBD）
            print(f"相机位置: {camera.location}, 旋转: {camera.rotation_euler}")
            # 设置相机为活动相机
            bpy.context.scene.camera = camera
            
            # 计时开始
            start_time = time.time()
            
            # 渲染当前帧
            bpy.context.scene.render.filepath = os.path.join(render_output_path, f"render_{str(payload_rec.extDesc).zfill(5)}.png")
            bpy.ops.render.render(write_still=True)
            
            # 计时结束
            end_time = time.time()
            
            # 计算并打印渲染时间
            render_time = end_time - start_time
            print(f"渲染一帧耗时: {render_time:.4f} 秒")
        # time.sleep(1./30.)

def main(render_output_path, Avatar_path, render_params):

    # 设置渲染输出文件的路径
    face_obj, skelton_obj, camera = setup_render_environment(render_output_path, Avatar_path, render_params)

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
    
    fps = 30 # 帧率
    cam_r0 = math.radians(-110)  # 初始位置-90度，转换为弧度
    cam_rv = math.radians(1)  # 角速度5度每秒，转换为弧度每秒
    process_face_data(servicer, face_obj, skelton_obj, camera, fps,
                      render_output_path, index_to_category_name, 
                      render_params, cam_r0, cam_rv)

if __name__ == "__main__":
    
    # Avatar_path = "../Render_Avatar/female-sports2_52shape_key_rename.blend"
    # render_params = RenderParameters(
    #     camera_location=(0, -3, 1.5),
    #     camera_rotation=(1.57, 0, 0),
    #     light_location=(0, -3, 3),
    #     light_energy=1000,
    #     render_engine='BLENDER_EEVEE',
    #     render_samples=1
    # )
    
    # Avatar_path = "/home/ztw/HVCCS/data/boy52blendshapes.blend"


    Avatar_path = "../Render_Avatar/nezha_with_backgoud.blend"
    render_params = RenderParameters(
        camera_location=(0, -4, 1.8),
        camera_rotation=(1.57, 0, 0),
        light_location=(1, -3, 3),
        light_energy=1500,
        render_engine='CYCLES', # CYCLES or BLENDER_EEVEE
        render_samples=32
    )

    output_path_relative = "res/render_res"

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_output_path = os.path.join(script_dir, output_path_relative)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(render_output_path), exist_ok=True)

    main(render_output_path, Avatar_path, render_params)

