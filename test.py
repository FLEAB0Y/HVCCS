import bpy
import os

def setup_and_render(blend_file_path, output_path):
    # 清除当前场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 删除所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 打开指定的blend文件
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # 打印场景中的所有对象
    print("场景中的所有对象:")
    for obj in bpy.data.objects:
        if obj:
            print(f"找到目标对象: {obj.name}")
            if obj.type == 'MESH' and obj.data.shape_keys:
                print(f"发现shapekeys在对象 '{obj.name}':")
                for kb in obj.data.shape_keys.key_blocks:
                    print(f"  - {kb.name} (值: {kb.value})")
            else:
                print(f"对象 '{obj.name}' 没有shapekeys或不是Mesh类型")
        else:
            print("场景中未找到目标对象")

    
    # # 检查特定mesh人物的shapekeys
    # print("检查HG_Body.001对象的shapekeys...")
    
    # 尝试获取特定对象
    # target_obj = bpy.data.objects.get("HG_Body.001")
    

    
    # 设置渲染参数
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # # 确保使用适当的渲染引擎
    # scene.render.engine = 'CYCLES'  # 改为使用Cycles渲染引擎
    
    # # Cycles渲染引擎的低质量设置
    # scene.cycles.samples = 32       # 降低采样数
    # scene.cycles.max_bounces = 4    # 降低最大光线反弹次数
    # scene.cycles.diffuse_bounces = 2
    # scene.cycles.glossy_bounces = 2
    # scene.cycles.transmission_bounces = 2
    # scene.cycles.volume_bounces = 0
    # scene.cycles.transparent_max_bounces = 2
    # scene.cycles.use_denoising = False  # 关闭降噪
    # scene.cycles.use_adaptive_sampling = True  # 使用自适应采样
    # scene.cycles.adaptive_threshold = 0.1  # 设置自适应采样阈值
    
    # # 尝试使用GPU加速渲染
    # try:
    #     scene.cycles.device = 'GPU'
    # except:
    #     print("无法使用GPU渲染，使用CPU")
    #     scene.cycles.device = 'CPU'
    
    # # 降低渲染质量设置
    # scene.render.use_high_quality_normals = False
    # scene.render.use_motion_blur = False    # 关闭运动模糊
    # 确保使用适当的渲染引擎
    scene.render.engine = 'BLENDER_EEVEE'  # 改为使用EEVEE渲染引擎
    
    # EEVEE渲染引擎的设置 - 优化为最快速度
    scene.eevee.taa_render_samples = 1      # 最小采样数
    scene.eevee.use_soft_shadows = False    # 关闭软阴影
    scene.eevee.use_ssr = False             # 关闭屏幕空间反射
    scene.eevee.use_ssr_refraction = False  # 关闭屏幕空间折射
    scene.eevee.use_gtao = False            # 关闭全局环境光遮蔽
    scene.eevee.use_bloom = False           # 关闭泛光效果
    scene.eevee.use_volumetric_shadows = False  # 关闭体积阴影
    scene.eevee.volumetric_samples = 16     # 降低体积采样
    
    # 降低渲染质量设置
    scene.render.use_high_quality_normals = False
    scene.render.use_motion_blur = False    # 关闭运动模糊
    
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
    
    # 获取场景中的所有光源并修改它们的参数
    light_sources = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    
    if light_sources:
        print(f"找到 {len(light_sources)} 个光源:")
        for i, light in enumerate(light_sources):
            # 根据光源类型设置不同的位置和强度
            if i == 0:  # 第一个光源作为主光源
                light.location = (1, -5, 3)
                light.data.energy = 1500
            else:  # 其他光源作为辅助光源
                light.location = (-2, -3, 2)
                light.data.energy = 500
            
            print(f"光源 '{light.name}' 类型: {light.data.type}, 位置: {light.location}, 强度: {light.data.energy}")
    else:
        # 如果没有光源，创建一个主光源和一个辅助光源
        print("未找到光源，创建新光源")
        
        # 创建主光源 (Sun)
        bpy.ops.object.light_add(type='SUN', location=(3, -2, 4))
        main_light = bpy.context.object
        main_light.name = "Main_Light"
        main_light.data.energy = 2.5
        print(f"创建主光源: {main_light.name}, 位置: {main_light.location}, 强度: {main_light.data.energy}")
        
        # 创建辅助光源 (Point)
        bpy.ops.object.light_add(type='POINT', location=(-2, -3, 2))
        fill_light = bpy.context.object
        fill_light.name = "Fill_Light"
        fill_light.data.energy = 1.5
        print(f"创建辅助光源: {fill_light.name}, 位置: {fill_light.location}, 强度: {fill_light.data.energy}")
    
    # 执行渲染
    bpy.ops.render.render(write_still=True)
    
    print(f"渲染完成，输出路径: {output_path}")

if __name__ == "__main__":
    blend_file_path = "/home/ztw/Render_Avatar/nezha.blend"
    # blend_file_path = "../female-sports2_52shape_key_rename.blend"
    # blend_file_path = "data/boy52blendshapes.blend"
    output_path_relative = "res/render_res/rendered_image.png"
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_path_relative)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    setup_and_render(blend_file_path, output_path)