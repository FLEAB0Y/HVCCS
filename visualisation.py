import bpy
import io
from PIL import Image

# 设置渲染引擎
bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 或者 'BLENDER_EEVEE'

# 创建一个BytesIO对象来保存图像数据
image_data = io.BytesIO()

# 设置渲染输出为PNG格式，并将输出重定向到image_data
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = 'D:/HVCCS/res/render_res/temp_image'  # 这个路径不会被实际使用

# 渲染当前帧
bpy.ops.render.render(write_still=True)

# 获取渲染的图像数据并保存到BytesIO对象
render_result = bpy.data.images['Render Result']
render_result.save_render(image_data, 'PNG')

# 将指针重置到开始位置，以便可以读取数据
image_data.seek(0)
print(len(image_data.read()))
# 现在image_data中包含了渲染的图像，可以进一步处理或保存
# 例如，使用PIL打开图像
rendered_image = Image.open(image_data)

# 清理临时渲染结果
bpy.data.images.remove(render_result)

# rendered_image现在可以在内存中使用，例如显示或转换格式
rendered_image.show()
