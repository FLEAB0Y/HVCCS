import bpy
import numpy as np
import os

# Load the blend file
blend_file_path = os.path.abspath("data/girl52blendshapes.blend")
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

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

# switch on nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
  
# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)
  
# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')      
rl.location = 185, 285
 
# create output node
v = tree.nodes.new('CompositorNodeViewer')   
v.location = 750, 210
v.use_alpha = False
 
# Links
links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input

# 渲染前打印场景中的对象
for obj in bpy.data.objects:
    print(f"场景中的对象: {obj.name}, 类型: {obj.type}")
 
# render
bpy.ops.render.render()
 
# get viewer pixels
pixels = bpy.data.images['Viewer Node'].pixels
print(len(pixels))  # size is always width * height * 4 (rgba)

# 检查Viewer Node图像尺寸
viewer_img = bpy.data.images['Viewer Node']
print(f"图像尺寸: {viewer_img.size}, 像素数: {len(viewer_img.pixels)}")
 
# copy buffer to numpy array for faster manipulation
arr = np.array(pixels[:])

print(arr.shape)  # (width * height * 4)

# Save the rendered image
output_dir = "/home/abc/ztw_HVCCS/HVCCS/res/render_res"
# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_render.png")
bpy.data.images['Viewer Node'].save_render(output_path)
print(f"Rendered image saved to {output_path}")