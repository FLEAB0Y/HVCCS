import maya.cmds as cmds
import maya.standalone
import os

# 初始化Maya独立模式
maya.standalone.initialize()

try:
    # 文件路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "nezha.mb")
    
    # 打开Maya文件
    cmds.file(file_path, open=True, force=True)
    
    # 设置渲染参数
    width = 1920
    height = 1080
    cmds.setAttr("defaultResolution.width", width)
    cmds.setAttr("defaultResolution.height", height)
    cmds.setAttr("defaultResolution.deviceAspectRatio", float(width) / height)
    
    # 设置输出图像格式为PNG
    cmds.setAttr("defaultRenderGlobals.imageFormat", 32)
    cmds.setAttr("defaultRenderGlobals.animation", 0)  # 关闭动画渲染
    
    # 设置输出目录和文件名
    output_dir = os.path.join(current_dir, "renders")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = "nezha_render"
    cmds.workspace(fileRule=["images", output_dir])
    cmds.setAttr("defaultRenderGlobals.imageFilePrefix", output_file, type="string")
    
    # 获取场景中的相机
    cameras = cmds.ls(cameras=True)
    render_cam = None
    
    # 优先使用persp相机
    if "persp" in cameras:
        render_cam = "persp"
    else:
        render_cam = cameras[0]
    
    print(f"使用相机 {render_cam} 进行渲染...")
    
    # 执行渲染
    # 使用Maya Software渲染器
    cmds.setAttr("defaultRenderGlobals.currentRenderer", "mayaSoftware", type="string")
    cmds.render(render_cam)
    
    # 如果使用Arnold渲染器，取消下面这行的注释
    # cmds.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
    # cmds.arnoldRender(cam=render_cam, width=width, height=height)
    
    print(f"渲染完成！图像保存在: {os.path.join(output_dir, output_file)}")

finally:
    # 关闭Maya独立模式
    maya.standalone.uninitialize()