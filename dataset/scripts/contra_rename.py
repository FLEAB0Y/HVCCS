import os
import shutil

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 定义基于脚本位置的路径
proc_dir = os.path.join(script_dir, "..", "proc_features")
ori_dir = os.path.join(script_dir, "..", "ori_features")
contra_dir = os.path.join(script_dir, "..", "contra_features")

# 确保目标目录存在
os.makedirs(contra_dir, exist_ok=True)

# 遍历proc_features目录中的所有txt文件
for filename in os.listdir(proc_dir):
    if filename.endswith(".txt"):
        # 解析文件名，格式为：0001_name_q4_l0.txt
        parts = filename.split("_")
        if len(parts) >= 4:  # 确保有足够的部分
            number = parts[0]  # 第一段是编号
            name = parts[1]    # 第二段是名称
            
            # 在ori_features目录中查找对应的文件
            ori_file = os.path.join(ori_dir, f"{name}.txt")
            if os.path.exists(ori_file):
                # 复制文件到contra_features目录，并重命名
                new_filename = f"{number}_{name}.txt"
                shutil.copy2(ori_file, os.path.join(contra_dir, new_filename))
                print(f"已复制 {ori_file} 到 {os.path.join(contra_dir, new_filename)}")
            else:
                print(f"警告: 文件 {ori_file} 未找到")
        else:
            print(f"警告: 无法解析文件名 {filename}，部分不足")