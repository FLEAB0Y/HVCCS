from server import THStreamServiceServicer, serve
import threading
import time
import json
import socket
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                           QPushButton, QWidget, QStatusBar, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
import numpy as np
from collections import deque


class LatencyMonitor(QMainWindow):
    def __init__(self, port_mappings):
        super().__init__()
        self.port_mappings = port_mappings
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
        
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建图表组
        charts_group = QGroupBox("用户监控 (30秒)")
        charts_layout = QGridLayout()
        charts_group.setLayout(charts_layout)
        
        # 为每个端口映射创建图表和信息显示
        self.plots = {}
        self.bandwidth_plots = {}
        self.stats_labels = {}
        
        row, col = 0, 0
        for i, (grpc_port, socket_port) in enumerate(self.port_mappings):
            # 创建每个用户的容器
            user_container = QWidget()
            user_layout = QVBoxLayout(user_container)
            
            # 创建延迟图表
            latency_widget = pg.PlotWidget()
            latency_widget.setBackground('w')
            latency_widget.setTitle(f'用户 {grpc_port}/{socket_port} - 延迟')
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
            
            # 添加到网格布局
            charts_layout.addWidget(user_container, row, col)
            
            # 更新行列位置
            col += 1
            if col > 1:  # 每行两个用户
                col = 0
                row += 1
        
        main_layout.addWidget(charts_group)
        
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

def grpc_thread(grpc_port, socket_port, latency_monitor=None):
    """gRPC线程处理函数"""
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port))
    server_thread.start()

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
                face_data_bytes = payload_rec.faceData
                limb_data_bytes = payload_rec.limbData
                timestamp = payload_rec.extDesc  # 获取时间戳
                
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

def latency_feedback_server(feedback_port, port_mappings, latency_monitor=None):
    """接收Unity发送的延迟反馈数据的服务器"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", feedback_port))
    server.listen(5)
    print(f"[延迟反馈] 服务器已启动，监听端口: {feedback_port}")
    
    # 根据反馈端口找到对应的grpc端口
    grpc_port = None
    for grpc_port, socket_port in port_mappings:
        if feedback_port == socket_port + 1000:  # 9890对应8890
            break
    
    while True:
        try:
            client, addr = server.accept()
            data = client.recv(1024)
            if data:
                feedback = data.decode('utf-8')
                if feedback.startswith("latency:"):
                    latency_value = float(feedback.split(':')[1])
                    latency_ms = int(latency_value * 1000)  # 转换为毫秒
                    print(f"[延迟反馈] 从Unity接收到的实际延迟: {latency_value} 秒")
                    
                    # 将Unity反馈的延迟数据添加到UI
                    if latency_monitor and grpc_port:
                        latency_monitor.add_latency_data(grpc_port, latency_ms)
                    
            client.close()
        except Exception as e:
            print(f"[延迟反馈] 接收反馈数据错误: {e}")

if __name__ == '__main__':
    # 定义 gRPC 和对应的 socket 端口
    port_mappings = [
        (50051, 8890),
        (50052, 8891),
        (50053, 8892),
        (50054, 8893)
    ]
    
    # 对应的反馈端口
    feedback_ports = [9890, 9891, 9892, 9893]

    # 创建 PyQt 应用和延迟监控窗口
    app = QApplication(sys.argv)
    latency_monitor = LatencyMonitor(port_mappings)
    latency_monitor.show()

    # 为每对端口启动一个线程
    threads = []
    for grpc_port, socket_port in port_mappings:
        t = threading.Thread(target=grpc_thread, args=(grpc_port, socket_port, latency_monitor))
        t.start()
        threads.append(t)
    
    # 为每个反馈端口启动一个监听线程
    for feedback_port in feedback_ports:
        t = threading.Thread(target=latency_feedback_server, args=(feedback_port, port_mappings, latency_monitor))
        t.start()
        threads.append(t)

    # 启动Qt事件循环
    sys.exit(app.exec_())