import bpy
import os

def setup_and_render(blend_file_path, output_path):
    # 清除当前场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 打开指定的blend文件
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    # 检查mesh人物的shapekeys
    print("检查场景中的mesh人物是否有shapekeys...")
    found_shapekeys = False
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            print(f"检查mesh对象: {obj.name}")
            if obj.data.shape_keys:
                found_shapekeys = True
                print(f"发现shapekeys在对象 '{obj.name}':")
                for kb in obj.data.shape_keys.key_blocks:
                    print(f"  - {kb.name} (值: {kb.value})")
            else:
                print(f"对象 '{obj.name}' 没有shapekeys")
    
    if not found_shapekeys:
        print("场景中没有找到任何带有shapekeys的mesh对象")
    
    # 设置渲染参数
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # 确保使用适当的渲染引擎
    scene.render.engine = 'CYCLES'  # 改为使用Cycles渲染引擎
    
    # Cycles渲染引擎的低质量设置
    scene.cycles.samples = 32       # 降低采样数
    scene.cycles.max_bounces = 4    # 降低最大光线反弹次数
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 2
    scene.cycles.use_denoising = False  # 关闭降噪
    scene.cycles.use_adaptive_sampling = True  # 使用自适应采样
    scene.cycles.adaptive_threshold = 0.1  # 设置自适应采样阈值
    
    # 尝试使用GPU加速渲染
    try:
        scene.cycles.device = 'GPU'
    except:
        print("无法使用GPU渲染，使用CPU")
        scene.cycles.device = 'CPU'
    
    # 降低渲染质量设置
    scene.render.use_high_quality_normals = False
    scene.render.use_motion_blur = False    # 关闭运动模糊
    # # 确保使用适当的渲染引擎
    # scene.render.engine = 'BLENDER_EEVEE'  # 改为使用EEVEE渲染引擎
    
    # # EEVEE渲染引擎的设置 - 优化为最快速度
    # scene.eevee.taa_render_samples = 1      # 最小采样数
    # scene.eevee.use_soft_shadows = False    # 关闭软阴影
    # scene.eevee.use_ssr = False             # 关闭屏幕空间反射
    # scene.eevee.use_ssr_refraction = False  # 关闭屏幕空间折射
    # scene.eevee.use_gtao = False            # 关闭全局环境光遮蔽
    # scene.eevee.use_bloom = False           # 关闭泛光效果
    # scene.eevee.use_volumetric_shadows = False  # 关闭体积阴影
    # scene.eevee.volumetric_samples = 16     # 降低体积采样
    
    # # 降低渲染质量设置
    # scene.render.use_high_quality_normals = False
    # scene.render.use_motion_blur = False    # 关闭运动模糊
    
    # 找到相机，如果没有相机则创建一个
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    
    if not camera:
        # 创建新相机
        bpy.ops.object.camera_add(location=(10, 5, 0), rotation=(0, 0, 0))
        camera = bpy.context.object
    
    # 修改相机位置到指定坐标
    camera.location = (0, -3, 1.5)
    camera.rotation_euler = (1.57, 0, 0)
    print(f"相机位置: {camera.location}, 旋转: {camera.rotation_euler}")
    # 设置当前相机
    scene.camera = camera
    
    # 添加光源（如果需要）
    if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
        light = bpy.context.object
        light.data.energy = 2.0  # 调整亮度
    
    # 执行渲染
    bpy.ops.render.render(write_still=True)
    
    print(f"渲染完成，输出路径: {output_path}")

if __name__ == "__main__":
    blend_file_path = "/home/ztw/Render/female-sports2.blend"
    output_path = "/home/ztw/HVCCS/res/render_res/rendered_image.png"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    setup_and_render(blend_file_path, output_path)