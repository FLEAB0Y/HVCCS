import bpy


# 删除所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 加载三维文件
bpy.ops.import_scene.fbx(filepath="/home/ztw/MediaPipe/face_detec/data/woman_young1.fbx")
# bpy.ops.import_scene.gltf(filepath="your_path/charactor_with_blendshapes.glb")
# bpy.ops.wm.open_mainfile(filepath="boy52blendshapes.blend")

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


# 添加一个相机
bpy.ops.object.camera_add(location=(0, 3, 0))
camera = bpy.context.active_object
camera.rotation_euler = (-1.1708, 3.141592, 0)  # 旋转相机，使其面向模型

# 设置相机为活动相机
bpy.context.scene.camera = camera

# 添加光照
bpy.ops.object.light_add(type='POINT', location=(0, -3, 3))
light = bpy.context.active_object
light.data.energy = 1000  # 设置光照强度

# 设置渲染输出文件的路径
bpy.context.scene.render.filepath = "/home/ztw/MediaPipe/face_detec/res/render_res/render_output.png"

# 设置渲染采样率
bpy.context.scene.cycles.samples = 4  # 设置采样率为1 
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 使用Cycles渲染引擎

# 执行渲染并保存结果图像
bpy.ops.render.render(write_still=True)