import socket
import numpy as np
import time
import sys
import os
from plyfile import PlyData
import struct
import threading

class PointCloudServer:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.running = False
    
    def load_ply(self, file_path):
        """读取PLY文件并返回点云数据"""
        try:
            plydata = PlyData.read(file_path)
            vertex = plydata['vertex']
            
            # 提取XYZ坐标
            x = vertex['x']
            y = vertex['y']
            z = vertex['z']
            
            # 尝试提取RGB值（如果存在）
            try:
                r = vertex['red']
                g = vertex['green']
                b = vertex['blue']
                has_rgb = True
            except ValueError:
                # 如果没有颜色数据，创建默认颜色（白色）
                r = np.ones_like(x) * 255
                g = np.ones_like(x) * 255
                b = np.ones_like(x) * 255
                has_rgb = False
            
            # 将数据组合成XYZRGB格式
            points = np.column_stack((x, y, z, r, g, b))
            
            print(f"加载了 {len(points)} 个点，{'包含' if has_rgb else '不包含'}RGB颜色")
            return points
        except Exception as e:
            print(f"加载PLY文件失败: {e}")
            return None
    
    def load_ply_folder(self, folder_path):
        """读取文件夹中的所有PLY文件并返回点云数据列表"""
        points_list = []
        filenames = []
        
        try:
            # 获取文件夹中所有PLY文件
            ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ply')]
            
            if not ply_files:
                print(f"在 {folder_path} 中没有找到PLY文件")
                return None, None
            
            for ply_file in ply_files:
                file_path = os.path.join(folder_path, ply_file)
                print(f"正在加载: {file_path}")
                points = self.load_ply(file_path)
                if points is not None:
                    points_list.append(points)
                    filenames.append(ply_file)
            
            print(f"总共加载了 {len(points_list)} 个PLY文件")
            return points_list, filenames
        except Exception as e:
            print(f"加载PLY文件夹失败: {e}")
            return None, None
    
    def start_server(self):
        """启动Socket服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"服务器已启动在 {self.host}:{self.port}")
            
            # 启动接受客户端线程
            accept_thread = threading.Thread(target=self.accept_clients)
            accept_thread.daemon = True
            accept_thread.start()
            
            return True
        except Exception as e:
            print(f"启动服务器失败: {e}")
            return False
    
    def accept_clients(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                print(f"客户端已连接: {address}")
                self.clients.append(client_socket)
            except Exception as e:
                if self.running:
                    print(f"接受客户端连接失败: {e}")
                break
    
    def send_point_cloud(self, points):
        """向所有连接的客户端发送点云数据"""
        if len(self.clients) == 0:
            print("没有连接的客户端，等待连接...")
            return False
        
        try:
            # 记录开始发送时间
            start_time = time.time()
            
            # 首先发送点的数量
            num_points = len(points)
            
            # 预先将所有点的数据打包到一个缓冲区中
            # 创建一个字节数组来存储所有点的数据
            point_buffer = bytearray()
            
            # 将点数打包
            point_buffer.extend(struct.pack('!I', num_points))
            
            # 使用NumPy向量化操作一次性处理所有点
            # 提取x,y,z坐标和归一化的r,g,b颜色
            x = points[:, 0].astype(np.float32)
            y = points[:, 1].astype(np.float32)
            z = points[:, 2].astype(np.float32)
            r = (points[:, 3] / 255.0).astype(np.float32)
            g = (points[:, 4] / 255.0).astype(np.float32)
            b = (points[:, 5] / 255.0).astype(np.float32)
            
            # 一次性打包所有点的数据
            for i in range(num_points):
                point_buffer.extend(struct.pack('!ffffff', x[i], y[i], z[i], r[i], g[i], b[i]))
                
            # 向所有客户端发送数据
            for client in self.clients[:]:
                try:
                    # 一次性发送所有数据
                    client.sendall(point_buffer)
                    
                    # 记录完成发送时间
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"已发送 {num_points} 个点到客户端，耗时: {elapsed_time:.4f} 秒")
                    
                except Exception as e:
                    print(f"发送点云数据失败: {e}")
                    self.clients.remove(client)
                    
            return True
        except Exception as e:
            print(f"发送点云数据时出错: {e}")
            return False
    
    def stop_server(self):
        """停止服务器"""
        self.running = False
        
        # 关闭所有客户端连接
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients = []
        
        # 关闭服务器
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("服务器已停止")

def main():
    # 直接在代码中定义参数
    ply_folder = "C:/Users/24150/plydata"  # 指定存放PLY文件的文件夹
    host = '127.0.0.1'
    port = 8888
    interval = 0.  # 0毫秒间隔
    
    # 创建服务器实例
    server = PointCloudServer(host, port)
    
    # 加载文件夹中的所有PLY文件
    points_list, filenames = server.load_ply_folder(ply_folder)
    if points_list is None or len(points_list) == 0:
        return
    
    # 启动服务器
    if not server.start_server():
        return
    
    print("服务器已启动，将以33毫秒间隔自动发送点云数据...")
    print("按Ctrl+C停止程序")
    
    try:
        # 主循环 - 自动发送
        current_index = 0
        while True:
            # 发送当前点云
            print(f"发送: {filenames[current_index]}")
            server.send_point_cloud(points_list[current_index])
            
            # 切换到下一个点云
            current_index = (current_index + 1) % len(points_list)
            
            # 等待指定的时间间隔
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()