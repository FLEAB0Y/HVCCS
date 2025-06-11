import os
import shutil
import random  # 导入random模块用于随机排序

def rename_files_with_prefix():
    """为处理后的特征文件添加四位数前缀（随机顺序）"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置要处理的目录
    proc_features_dir = os.path.join(script_dir, "..", "proc_features")
    
    # 确保目录存在
    if not os.path.exists(proc_features_dir):
        print(f"目录不存在: {proc_features_dir}")
        return
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(proc_features_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"在 {proc_features_dir} 中没有找到txt文件")
        return
    
    # 随机打乱文件顺序
    random.shuffle(txt_files)  # 使用随机排序代替字母排序
    
    print(f"找到 {len(txt_files)} 个txt文件，开始随机顺序重命名...")
    
    # 重命名文件
    for i, old_filename in enumerate(txt_files, 1):
        # 创建新文件名：四位数前缀 + 原始文件名
        prefix = f"{i:04d}"
        new_filename = f"{prefix}_{old_filename}"
        
        # 构建完整的文件路径
        old_path = os.path.join(proc_features_dir, old_filename)
        new_path = os.path.join(proc_features_dir, new_filename)
        
        # 重命名文件
        try:
            os.rename(old_path, new_path)
            print(f"已重命名: {old_filename} -> {new_filename}")
        except Exception as e:
            print(f"重命名 {old_filename} 失败: {e}")
    
    print("随机顺序重命名完成！")

if __name__ == "__main__":
    # 设置随机种子，如果需要确保每次运行结果一致可以取消注释下一行
    # random.seed(42)
    rename_files_with_prefix()