import bpy
import os
import math


# 打开指定的Blender文件
blend_file_path = "D:/HVCCS/data/girl52blendshapes.blend"
if os.path.exists(blend_file_path):
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    print(f"成功打开Blender文件: {blend_file_path}")
else:
    print(f"错误：无法找到Blender文件: {blend_file_path}")
    exit(1)

# 添加摄像头
def add_camera():
    # 删除现有的摄像头（如果有的话）
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)
    
    # 创建新摄像头
    camera_data = bpy.data.cameras.new("Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_object)
    
    # 设置摄像头位置和旋转
    camera_object.location = (0, -2.5, 1.0)  # 位置：前方2.5米，高1米
    camera_object.rotation_euler = (math.radians(90), 0, 0)  # 向下倾斜70度
    
    # 将此摄像头设为活动摄像头
    bpy.context.scene.camera = camera_object
    
    print("添加了新的摄像头")
    return camera_object

# 添加光源
def add_light():
    # 创建主光源 - 前方高处柔光
    key_light_data = bpy.data.lights.new(name="Key_Light", type='AREA')
    key_light_data.energy = 300  # 亮度
    key_light_data.size = 2.0  # 面积光源大小
    key_light_object = bpy.data.objects.new(name="Key_Light", object_data=key_light_data)
    bpy.context.collection.objects.link(key_light_object)
    key_light_object.location = (2.0, -2.0, 2.0)
    key_light_object.rotation_euler = (math.radians(60), math.radians(15), 0)
    
    # 创建填充光 - 侧面提亮阴影
    fill_light_data = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_light_data.energy = 100  # 较弱
    fill_light_data.size = 1.0
    fill_light_object = bpy.data.objects.new(name="Fill_Light", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light_object)
    fill_light_object.location = (-1.5, -1.0, 0.5)
    fill_light_object.rotation_euler = (math.radians(30), math.radians(-30), 0)
    
    # 创建背光 - 后方勾边光
    back_light_data = bpy.data.lights.new(name="Back_Light", type='SPOT')
    back_light_data.energy = 150
    back_light_object = bpy.data.objects.new(name="Back_Light", object_data=back_light_data)
    bpy.context.collection.objects.link(back_light_object)
    back_light_object.location = (0, 1.0, 2.0)
    back_light_object.rotation_euler = (math.radians(120), 0, 0)
    
    print("添加了三点光照")

# 添加摄像头和光源
camera = add_camera()
add_light()

# 设置渲染引擎
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 或者 'CYCLES'

# 启用合成节点
bpy.context.scene.use_nodes = True

# 获取合成节点树的引用
node_tree = bpy.context.scene.node_tree

# 清除所有现有节点
for node in node_tree.nodes:
    node_tree.nodes.remove(node)

# 添加渲染层节点和合成输出节点
render_layers_node = node_tree.nodes.new(type='CompositorNodeRLayers')
composite_node = node_tree.nodes.new(type='CompositorNodeComposite')

# 链接渲染层的图像输出到合成节点
node_tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])

# 设置渲染分辨率
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

# 添加 Viewer Node
viewer_node = node_tree.nodes.new(type='CompositorNodeViewer')
node_tree.links.new(render_layers_node.outputs['Image'], viewer_node.inputs['Image'])
node_tree.links.new(render_layers_node.outputs['Image'], composite_node.inputs['Image'])

# 执行渲染
bpy.ops.render.render()

# 导入OpenCV和NumPy
import cv2
import numpy as np

# 确保 Viewer Node 图像存在
if "Viewer Node" in bpy.data.images:
    # 获取Viewer Node图像
    viewer_image = bpy.data.images["Viewer Node"]
    
    # 获取图像尺寸
    width, height = viewer_image.size
    
    # 获取像素数据
    pixels = np.array(viewer_image.pixels[:])
    
    # 重塑数组为RGBA格式
    pixels = pixels.reshape((height, width, 4))
    
    # 转换为BGR格式(OpenCV使用BGR而非RGB)
    bgr_image = pixels[:, :, :3][:, :, ::-1]
    
    # 确保像素值在0-255范围内
    bgr_image = (bgr_image * 255).astype(np.uint8)
    
    # 显示图像
    cv2.imshow('Blender 渲染', bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("渲染结果已通过OpenCV显示")
else:
    print("错误: Viewer Node 图像不可用")

# 保留实时预览设置
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        # 设置视图为摄像机视图
        area.spaces[0].region_3d.view_perspective = 'CAMERA'
        # 启用实时渲染预览
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'  # 设置为渲染预览模式
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
        break

print("已切换到摄像机视图并启用实时渲染预览")
