import bpy

# 删除所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 加载三维文件
# bpy.ops.import_scene.fbx(filepath="data/woman_young1.fbx")
# bpy.ops.import_scene.gltf(filepath="your_path/charactor_with_blendshapes.glb")
bpy.ops.wm.open_mainfile(filepath="data/girl52blendshapes.blend")

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

# 添加简化的光照
bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))  # 使用太阳光替代点光源
light = bpy.context.active_object
light.data.energy = 3  # 太阳光能量不需要太高

# 设置渲染输出文件的路径
bpy.context.scene.render.filepath = "/home/ztw/HVCCS/res/render_res/render_output.png"

# 设置渲染采样率
bpy.context.scene.cycles.samples = 1  # 降低至最低采样率
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 已经使用EEVEE

# EEVEE特定优化
bpy.context.scene.eevee.taa_render_samples = 1  # 降低TAA采样
bpy.context.scene.eevee.use_bloom = False  # 关闭泛光
bpy.context.scene.eevee.use_ssr = False  # 关闭屏幕空间反射
# bpy.context.scene.eevee.use_ssao = False  # 这行有问题，移除或注释掉
bpy.context.scene.eevee.use_gtao = False  # 关闭全局环境光遮蔽
bpy.context.scene.eevee.use_soft_shadows = False  # 关闭软阴影

# 关闭或简化阴影
# for area in bpy.context.screen.areas:
#     if area.type == 'VIEW_3D':
#         for space in area.spaces:
#             if space.type == 'VIEW_3D':
#                 space.shading.light = 'FLAT'  # 使用简单照明

# 禁用抗锯齿
bpy.context.scene.render.filter_size = 0

# 降低渲染分辨率
# bpy.context.scene.render.resolution_x = 640  # 原始可能是1920或更高
# bpy.context.scene.render.resolution_y = 480  # 原始可能是1080或更高
# bpy.context.scene.render.resolution_percentage = 50  # 再降低50%

# 只渲染图像的一部分区域进行测试
# bpy.context.scene.render.use_border = True
# bpy.context.scene.render.border_min_x = 0.25
# bpy.context.scene.render.border_max_x = 0.75
# bpy.context.scene.render.border_min_y = 0.25
# bpy.context.scene.render.border_max_y = 0.75

# 启用多线程
bpy.context.scene.render.threads_mode = 'AUTO'
bpy.context.scene.render.threads = 4  # 使用4个线程
# 执行渲染并保存结果图像
bpy.ops.render.render(write_still=True)

print(dir(bpy.context.scene.eevee))