import argparse
from server import THStreamServiceServicer, serve
import threading
import time
import socket
import sys
import io
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                           QPushButton, QWidget, QStatusBar, QGroupBox, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
from collections import deque
from plyfile import PlyData
import struct
import numpy as np


class LatencyMonitor(QMainWindow):
    def __init__(self, port_mappings, point_cloud_grpc_port=None):
        super().__init__()
        self.port_mappings = port_mappings
        self.point_cloud_grpc_port = point_cloud_grpc_port
        self.latency_data = {}  # 存储各用户的延迟数据
        self.bandwidth_data = {}  # 存储各用户的带宽数据 (bytes/sec)
        self.packet_stats = {}  # 存储各用户的数据包统计信息
        self.max_points = 300  # 30秒 * 10个点/秒
        
        # 初始化每个用户的数据
        for grpc_port, socket_port in port_mappings:
            self.latency_data[grpc_port] = []
            self.bandwidth_data[grpc_port] = []
            self.packet_stats[grpc_port] = {
                'total_packets': 0,
                'total_bytes': 0,
                'last_second_bytes': 0,
                'bytes_buffer': deque(maxlen=10),  # 存储最近10个100ms的数据量
                'last_update': time.time()
            }
        
        self.init_ui()
        
        # 设置定时器，每100毫秒更新一次图表
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()
    
    def init_ui(self):
        self.setWindowTitle('延迟和带宽监控')
        self.setGeometry(100, 100, 1200, 800)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        central_widget = QWidget()
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        main_layout = QVBoxLayout(central_widget)

        b_users_container = QWidget()
        b_users_layout = QGridLayout(b_users_container)
        
        # 为每个端口映射创建图表和信息显示
        self.plots = {}
        self.bandwidth_plots = {}
        self.stats_labels = {}
        
        # 计数器用于跟踪B类用户的位置
        b_user_count = 0
        
        for grpc_port, socket_port in self.port_mappings:
            # 创建每个用户的容器
            user_container = QWidget()
            user_layout = QVBoxLayout(user_container)
            
            # 创建用户标题标签
            if grpc_port == 50055:
                user_title = QLabel(f"全息教师：点云数据，访问端口号：{grpc_port}/{socket_port}")
            else:
                if self.point_cloud_grpc_port and grpc_port == self.point_cloud_grpc_port:
                    user_title = QLabel(f"点云用户：gRPC {grpc_port} / Socket {socket_port}")
                else:
                    b_user_count += 1
                    user_title = QLabel(f"全息学生{b_user_count}：gRPC {grpc_port} / Socket {socket_port}")
                
            user_title.setFont(QFont('Arial', 11, QFont.Bold))
            user_title.setStyleSheet("color: #003366; background-color: #e6f2ff; padding: 5px; border-radius: 4px;")
            user_title.setAlignment(Qt.AlignCenter)
            user_layout.addWidget(user_title)
            
            # 创建延迟图表
            latency_widget = pg.PlotWidget()
            latency_widget.setBackground('w')
            latency_widget.setTitle('延迟监测')
            latency_widget.setLabel('left', '延迟 (ms)')
            latency_widget.setLabel('bottom', '时间 (s)')
            latency_widget.showGrid(x=True, y=True)
            latency_widget.setFixedHeight(180)
            
            # 添加延迟曲线
            pen = pg.mkPen(color=(255, 0, 0), width=2)
            plot_item = latency_widget.plot(pen=pen)
            self.plots[grpc_port] = plot_item
            
            # 创建带宽图表
            bandwidth_widget = pg.PlotWidget()
            bandwidth_widget.setBackground('w')
            bandwidth_widget.setTitle('带宽使用')
            bandwidth_widget.setLabel('left', '带宽 (KB/s)')
            bandwidth_widget.setLabel('bottom', '时间 (s)')
            bandwidth_widget.showGrid(x=True, y=True)
            bandwidth_widget.setFixedHeight(150)
            
            # 添加带宽曲线
            bw_pen = pg.mkPen(color=(0, 128, 255), width=2)
            bw_plot_item = bandwidth_widget.plot(pen=bw_pen)
            self.bandwidth_plots[grpc_port] = bw_plot_item
            
            # 添加统计信息标签
            stats_label = QLabel()
            stats_label.setFont(QFont('Arial', 10))
            stats_label.setAlignment(Qt.AlignLeft)
            stats_label.setText("正在收集数据...")
            self.stats_labels[grpc_port] = stats_label
            
            # 将控件添加到用户容器
            user_layout.addWidget(latency_widget)
            user_layout.addWidget(bandwidth_widget)
            user_layout.addWidget(stats_label)
            
            # 根据用户类型放置在对应的网格位置
            if grpc_port == 50055:
                main_layout.addWidget(user_container)
            else:
                row = (b_user_count - 1) // 2
                col = (b_user_count - 1) % 2
                b_users_layout.addWidget(user_container, row, col)

        if b_user_count:
            main_layout.addWidget(b_users_container)
        main_layout.addStretch()

    # 添加状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('监控已启动')
    
    def update_plots(self):
        # 更新每个用户的图表和统计信息
        for grpc_port in self.plots.keys():
            # 更新延迟图表
            latency_data = self.latency_data.get(grpc_port, [])
            if latency_data:
                # 只显示最近30秒的数据
                if len(latency_data) > self.max_points:
                    latency_data = latency_data[-self.max_points:]
                    self.latency_data[grpc_port] = latency_data
                
                # 更新延迟图表
                self.plots[grpc_port].setData(range(len(latency_data)), latency_data)
            
            # 更新带宽图表
            bandwidth_data = self.bandwidth_data.get(grpc_port, [])
            if bandwidth_data:
                # 只显示最近30秒的数据
                if len(bandwidth_data) > self.max_points:
                    bandwidth_data = bandwidth_data[-self.max_points:]
                    self.bandwidth_data[grpc_port] = bandwidth_data
                
                # 更新带宽图表 (转换为KB/s)
                kb_data = [b/1024 for b in bandwidth_data]
                self.bandwidth_plots[grpc_port].setData(range(len(kb_data)), kb_data)
            
            # 更新统计信息
            stats = self.packet_stats.get(grpc_port, {})
            if stats:
                # 计算当前延迟和平均延迟
                current_latency = latency_data[-1] if latency_data else 0
                avg_latency = sum(latency_data) / len(latency_data) if latency_data else 0
                
                # 计算当前带宽和平均带宽
                current_bandwidth = bandwidth_data[-1] if bandwidth_data else 0
                avg_bandwidth = sum(bandwidth_data) / len(bandwidth_data) if bandwidth_data else 0
                
                # 更新统计信息标签
                stats_text = (
                    f"<b>延迟统计:</b> 当前: {current_latency:.1f} ms | 平均: {avg_latency:.1f} ms<br>"
                    f"<b>带宽统计:</b> 当前: {current_bandwidth/1024:.2f} KB/s | 平均: {avg_bandwidth/1024:.2f} KB/s<br>"
                    f"<b>数据统计:</b> 总包数: {stats['total_packets']} | 总数据量: {stats['total_bytes']/1024:.2f} KB"
                )
                self.stats_labels[grpc_port].setText(stats_text)
    
    def add_latency_data(self, grpc_port, latency):
        """添加新的延迟数据点"""
        if grpc_port in self.latency_data:
            self.latency_data[grpc_port].append(latency)
            # 保持数据点不超过最大值
            if len(self.latency_data[grpc_port]) > self.max_points:
                self.latency_data[grpc_port].pop(0)
    
    def add_packet_data(self, grpc_port, data_size):
        """添加新的数据包信息"""
        if grpc_port in self.packet_stats:
            stats = self.packet_stats[grpc_port]
            stats['total_packets'] += 1
            stats['total_bytes'] += data_size
            
            # 更新带宽计算
            current_time = time.time()
            stats['bytes_buffer'].append(data_size)
            
            # 每100ms更新一次带宽计算 (10次/秒)
            time_diff = current_time - stats['last_update']
            if time_diff >= 0.1:
                # 计算过去一秒的带宽 (bytes/sec)
                total_bytes = sum(stats['bytes_buffer'])
                # 由于我们保存了10个100ms的数据，所以直接使用总和作为每秒带宽
                current_bandwidth = total_bytes
                
                # 添加到带宽数据
                self.bandwidth_data[grpc_port].append(current_bandwidth)
                if len(self.bandwidth_data[grpc_port]) > self.max_points:
                    self.bandwidth_data[grpc_port].pop(0)
                    
                stats['last_second_bytes'] = current_bandwidth
                stats['last_update'] = current_time


def send_combined_data(face_data_str, limb_data_str, timestamp, socket_port):
    """将extDesc(时间戳)、facedata和limbdata拼接到一起，用逗号分隔"""
    # 直接拼接时间戳、face_data和limb_data，用逗号分隔
    data_str = timestamp + "," + face_data_str + "," + limb_data_str
    
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", socket_port))
    
    # 发送数据
    client.send(data_str.encode('utf-8'))
    
    # 关闭连接
    client.close()

def send_point_cloud_data(point_cloud_binary, socket_port):
    """解析PLY二进制数据并发送点云数据"""
    try:
        # 使用BytesIO创建内存文件对象来读取PLY数据
        from io import BytesIO
        ply_data_io = BytesIO(point_cloud_binary)
        
        # 解析PLY数据
        plydata = PlyData.read(ply_data_io)
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
        num_points = len(points)
        
        print(f"解析了 {num_points} 个点，{'包含' if has_rgb else '不包含'}RGB颜色")
        
        # 发送点云数据
        return send_formatted_point_cloud(points, socket_port)
    
    except Exception as e:
        print(f"[点云处理] 解析PLY数据错误: {e}")
        return False

def send_formatted_point_cloud(points, socket_port):
    """按照ply_socket.py的格式发送点云数据"""
    try:
        # 记录开始发送时间
        start_time = time.time()
        
        # 创建TCP连接
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", socket_port))
        
        # 首先发送点的数量
        num_points = len(points)
        
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
        
        # 发送数据
        client.sendall(point_buffer)
        
        # 记录完成发送时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"已发送 {num_points} 个点到Unity客户端，耗时: {elapsed_time:.4f} 秒")
        
        # 关闭连接
        client.close()
        return True
        
    except Exception as e:
        print(f"[点云处理] 发送点云数据错误: {e}")
        return False

def grpc_thread(grpc_port, socket_port, latency_monitor=None, point_cloud_grpc_port=None):
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port))
    server_thread.start()

    is_point_cloud_port = (point_cloud_grpc_port is not None and grpc_port == point_cloud_grpc_port)
    if is_point_cloud_port:
        print(f"[点云数据] 启动点云数据专用服务: gRPC端口 {grpc_port}, Socket端口 {socket_port}")

    while True:    
        # 缓冲区空了就等待
        buffer_size = servicer.receive_data_buffer.get_size()
        while buffer_size < 1:
            time.sleep(0.01)
            buffer_size = servicer.receive_data_buffer.get_size()
        # 从缓冲区获取数据
        payload_rec = servicer.receive_data_buffer.get_items()
        if payload_rec:
            try:
                # 获取时间戳
                timestamp = payload_rec.extDesc
                
                # 判断是否为点云数据
                if is_point_cloud_port:
                    # 处理点云数据
                    point_data_bytes = payload_rec.pointData
                    data_size = len(point_data_bytes)
                    
                    # 添加数据包信息到UI
                    if latency_monitor:
                        latency_monitor.add_packet_data(grpc_port, data_size)
                    
                    # 解析并发送点云数据
                    send_point_cloud_data(point_data_bytes, socket_port)
                else:
                    # 处理常规数据（面部和肢体数据）
                    face_data_bytes = payload_rec.faceData
                    limb_data_bytes = payload_rec.limbData
                    
                    # 直接解码字节串为字符串
                    face_data_str = face_data_bytes.decode('utf-8')
                    limb_data_str = limb_data_bytes.decode('utf-8')
                    
                    # 计算数据大小
                    data_size = len(face_data_bytes) + len(limb_data_bytes)
                    
                    # 添加数据包信息到UI
                    if latency_monitor:
                        latency_monitor.add_packet_data(grpc_port, data_size)
                    
                    # 发送合并后的数据
                    send_combined_data(face_data_str, limb_data_str, timestamp, socket_port)

            except AttributeError as e:
                print(f"[gRPC Port {grpc_port}] AttributeError: {e}")
            except Exception as e:
                print(f"[gRPC Port {grpc_port}] 处理数据错误: {e}")

def latency_feedback_server(feedback_port, port_mappings, latency_monitor=None,
                            latency_compensation=None, feedback_offset=1000):
    """接收Unity发送的延迟反馈数据的服务器"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", feedback_port))
    server.listen(5)
    print(f"[延迟反馈] 服务器已启动，监听端口: {feedback_port}")
    
    # 根据反馈端口找到对应的grpc端口
    target = next(((gp, sp) for gp, sp in port_mappings if feedback_port == sp + feedback_offset), None)
    grpc_port = target[0] if target else None
    
    # 获取该gRPC端口的延迟补偿值
    compensation = 0
    if latency_compensation and grpc_port in latency_compensation:
        compensation = latency_compensation[grpc_port]
        print(f"[延迟反馈] 端口 {grpc_port} 应用延迟补偿: {compensation}ms")
    
    while True:
        try:
            client, addr = server.accept()
            data = client.recv(1024)
            if data:
                feedback = data.decode('utf-8')
                if feedback.startswith("latency:"):
                    latency_value = float(feedback.split(':')[1])
                    latency_ms = int(latency_value * 1000)  # 转换为毫秒
                    
                    # 应用延迟补偿
                    compensated_latency = latency_ms - compensation
                    print(f"[延迟反馈] 从Unity接收到的实际延迟: {latency_value} 秒, 补偿后: {compensated_latency}ms")
                    
                    # 将补偿后的延迟数据添加到UI
                    if latency_monitor and grpc_port:
                        latency_monitor.add_latency_data(grpc_port, compensated_latency)
                    
            client.close()
        except Exception as e:
            print(f"[延迟反馈] 接收反馈数据错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="gRPC 转 Socket 转发与监控")
    parser.add_argument("--grpc_ports", nargs='+', type=int, help="gRPC端口列表")
    parser.add_argument("--socket_ports", nargs='+', type=int, help="Socket端口列表")
    parser.add_argument("--point_cloud_grpc_port", type=int, help="点云数据gRPC端口")
    parser.add_argument("--feedback_offset", type=int, default=1000, help="反馈端口与Socket端口的偏移量")
    args = parser.parse_args()

    default_mappings = [
        (50051, 8890),
        (50052, 8891),
        (50053, 8892),
        (50054, 8893),
        (50055, 8894)
    ]

    if args.grpc_ports and args.socket_ports:
        if len(args.grpc_ports) != len(args.socket_ports):
            parser.error("gRPC端口与Socket端口数量必须一致")
        port_mappings = list(zip(args.grpc_ports, args.socket_ports))
    else:
        port_mappings = default_mappings

    point_cloud_grpc_port = None
    if args.point_cloud_grpc_port:
        if args.point_cloud_grpc_port not in [gp for gp, _ in port_mappings]:
            parser.error("点云端口必须包含在gRPC端口列表中")
        point_cloud_grpc_port = args.point_cloud_grpc_port
    elif port_mappings:
        point_cloud_grpc_port = port_mappings[-1][0]

    latency_compensation = {grpc_port: 0 for grpc_port, _ in port_mappings}
    feedback_ports = [socket_port + args.feedback_offset for _, socket_port in port_mappings]

    print("[配置] 用户延迟补偿值:")
    for grpc_port in latency_compensation:
        label = "点云数据" if point_cloud_grpc_port and grpc_port == point_cloud_grpc_port else "通用用户"
        print(f"  - {label}（端口{grpc_port}）: 0ms")

    app = QApplication(sys.argv)
    latency_monitor = LatencyMonitor(port_mappings, point_cloud_grpc_port)
    latency_monitor.show()

    threads = []
    for grpc_port, socket_port in port_mappings:
        t = threading.Thread(target=grpc_thread,
                             args=(grpc_port, socket_port, latency_monitor, point_cloud_grpc_port))
        t.start()
        threads.append(t)

    for feedback_port in feedback_ports:
        t = threading.Thread(target=latency_feedback_server,
                             args=(feedback_port, port_mappings, latency_monitor, latency_compensation, args.feedback_offset))
        t.start()
        threads.append(t)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()