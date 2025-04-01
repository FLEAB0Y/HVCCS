import os
import shutil
import sys

def clear_subdirectories(res_path):
    """清空指定路径下所有子文件夹的内容，但保留文件夹结构"""
    if not os.path.exists(res_path):
        print(f"错误: 路径 '{res_path}' 不存在")
        return False
    
    if not os.path.isdir(res_path):
        print(f"错误: '{res_path}' 不是一个文件夹")
        return False
    
    subdirs = [d for d in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, d))]
    
    if not subdirs:
        print(f"'{res_path}' 下没有子文件夹")
        return False
    
    print(f"将清空以下子文件夹的内容:")
    for subdir in subdirs:
        print(f"- {subdir}")
    
    confirm = input("确认清空以上子文件夹内容? (y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return False
    
    for subdir in subdirs:
        subdir_path = os.path.join(res_path, subdir)
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
            else:
                os.remove(item_path)
                print(f"已删除文件: {item_path}")
    
    print("所有子文件夹内容已清空")
    return True

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录并拼接res路径
    default_res_path = os.path.join(os.path.dirname(script_dir), "res")
    
    if len(sys.argv) > 1:
        res_path = sys.argv[1]
    else:
        res_path = default_res_path
    
    print(f"目标路径: {res_path}")
    clear_subdirectories(res_path)