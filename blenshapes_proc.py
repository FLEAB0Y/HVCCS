import bpy
import time

# 删除所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 加载三维文件
bpy.ops.import_scene.fbx(filepath="/home/ztw/MediaPipe/face_detec/data/male_old1.fbx")
# bpy.ops.import_scene.gltf(filepath="/home/ztw/MediaPipe/face_detec/data/VRoid_V110_Male_v1.1.3.glb")
# bpy.ops.wm.open_mainfile(filepath="/home/ztw/MediaPipe/face_detec/data/modified_model.blend")
# 获取导入的对象
render = None
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        render = obj
        break

if render is None:
    raise ValueError("没有找到 MESH 类型的对象")

# 打印 shape keys
if render.data.shape_keys:
    shape_keys = render.data.shape_keys.key_blocks
    print("Shape Keys:")
    for key in shape_keys:
        print(key.name)
else:
    print("该对象没有 shape keys")

# 重命名 Fcl_ALL_Neutral 为 _neutral
if "Fcl_ALL_Neutral" in shape_keys:
    shape_keys["Fcl_ALL_Neutral"].name = "_neutral"

# 保留的 shape keys 列表
keep_shape_keys = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

# 删除不在保留列表中的 shape keys
for key in shape_keys:
    if key.name not in keep_shape_keys:
        render.shape_key_remove(key)

# 打印 shape keys
if render.data.shape_keys:
    shape_keys = render.data.shape_keys.key_blocks
    print("Shape Keys:")
    for key in shape_keys:
        print(key.name)
else:
    print("该对象没有 shape keys")

# 保存修改后的模型
bpy.ops.wm.save_as_mainfile(filepath="/home/ztw/MediaPipe/face_detec/data/boy52blendshapes.blend")


# 添加一个相机
bpy.ops.object.camera_add(location=(0, 3, 1.2))
camera = bpy.context.active_object
camera.rotation_euler = (-1.5708, 3.141592, 0)  # 旋转相机，使其面向模型

# 设置相机为活动相机
bpy.context.scene.camera = camera

# 添加光照
bpy.ops.object.light_add(type='POINT', location=(0, -3, 3))
light = bpy.context.active_object
light.data.energy = 1000  # 设置光照强度

# 设置渲染输出文件的路径
bpy.context.scene.render.filepath = "/home/ztw/MediaPipe/face_detec/res/render_res/render_output.png"

# 设置渲染采样率
bpy.context.scene.cycles.samples = 1  # 设置采样率为1 
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 使用Cycles渲染引擎

# 记录渲染开始时间
start_time = time.time()

# 执行渲染并保存结果图像
bpy.ops.render.render(write_still=True)

# 记录渲染结束时间
end_time = time.time()

# 计算并输出渲染所需的时间
render_time = end_time - start_time
print(f"渲染所需时间: {render_time:.2f} 秒")