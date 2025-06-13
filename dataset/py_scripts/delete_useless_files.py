import os
import glob

def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建目标目录的路径（相对于脚本位置）
    target_dir = os.path.normpath(os.path.join(script_dir, "../proc_features/"))
    
    print(f"正在搜索目录: {target_dir}")
    
    # 检查目标目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误: 目标目录 {target_dir} 不存在!")
        return
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(target_dir, "*.txt"))
    
    deleted_count = 0
    deleted_files = []
    
    # 检查并删除符合条件的文件
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        
        # 移除扩展名并分割文件名
        name_parts = os.path.splitext(file_name)[0].split('_')
        
        # 检查文件名是否有足够的部分，且后两段为q32和l0
        if len(name_parts) >= 3 and name_parts[-2] == "q32" and name_parts[-1] == "l0":
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_files.append(file_name)
                print(f"已删除: {file_name}")
            except Exception as e:
                print(f"删除 {file_name} 时出错: {e}")
    
    # 打印结果统计
    print(f"\n共删除了 {deleted_count} 个无效文件")
    if deleted_count > 0:
        print("删除的文件列表:")
        for file in deleted_files[:10]:  # 只显示前10个文件
            print(f"  - {file}")
        if len(deleted_files) > 10:
            print(f"  ... 以及其他 {len(deleted_files) - 10} 个文件")

if __name__ == "__main__":
    main()