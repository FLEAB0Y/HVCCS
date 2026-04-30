import argparse
from server import THStreamServiceServicer, serve
import threading
import time
import socket
import json
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
from typing import Any, cast
from realtime_offline_splines_fit import cubic_hermite_coefficients


def qt_align(name):
    if hasattr(Qt, name):
        return getattr(Qt, name)
    if hasattr(Qt, 'AlignmentFlag') and hasattr(Qt.AlignmentFlag, name):
        return getattr(Qt.AlignmentFlag, name)
    return 0


class LatencyMonitor(QMainWindow):
    def __init__(self, port_mappings, point_cloud_grpc_port=None):
        super().__init__()
        self.port_mappings = port_mappings
        self.point_cloud_grpc_port = point_cloud_grpc_port
        self.latency_data = {}  # Stores total latency samples as (timestamp, value)
        self.network_latency_data = {}  # Network transit latency samples as (timestamp, value)
        self.decoding_latency_data = {}  # Decoding and fitting latency samples as (timestamp, value)
        self.rendering_latency_data = {}  # Rendering latency samples as (timestamp, value)
        self.sender_latency_data = {}  # Sender processing latency (t_encode - t_begin)
        self.bandwidth_data = {}  # Stores bandwidth samples as (timestamp, bytes/sec)
        self.packet_stats = {}  # 存储各用户的数据包统计信息
        self.max_points = 300  # 保留用于带宽统计，不用于延迟主图裁剪
        self.avg_window_seconds = 10
        self.plot_window_seconds = 3
        
        # 初始化每个用户的数据
        for grpc_port, socket_port in port_mappings:
            self.latency_data[grpc_port] = []
            self.network_latency_data[grpc_port] = []
            self.decoding_latency_data[grpc_port] = []
            self.rendering_latency_data[grpc_port] = []
            self.sender_latency_data[grpc_port] = []
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
        self.setWindowTitle('Latency and Bandwidth Monitor')
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
        self.sender_plots = {}
        self.network_plots = {}
        self.decoding_plots = {}
        self.rendering_plots = {}
        self.stats_labels = {}
        
        # 计数器用于跟踪B类用户的位置
        b_user_count = 0
        
        for grpc_port, socket_port in self.port_mappings:
            # 创建每个用户的容器
            user_container = QWidget()
            user_layout = QVBoxLayout(user_container)
            
            # 创建用户标题标签
            if grpc_port == 50055:
                user_title = QLabel(f"Teacher: Point Cloud Data, Ports {grpc_port}/{socket_port}")
            else:
                if self.point_cloud_grpc_port and grpc_port == self.point_cloud_grpc_port:
                    user_title = QLabel(f"Point Cloud User: gRPC {grpc_port} / Socket {socket_port}")
                else:
                    b_user_count += 1
                    user_title = QLabel(f"Avatar {b_user_count}: gRPC {grpc_port} / Socket {socket_port}")
                
            user_title.setFont(QFont('Arial', 11, QFont.Bold))
            user_title.setStyleSheet("color: #003366; background-color: #e6f2ff; padding: 5px; border-radius: 4px;")
            user_title.setAlignment(cast(Any, qt_align('AlignCenter')))
            user_layout.addWidget(user_title)
            
            # 创建单窗口多曲线延迟图表
            latency_widget = pg.PlotWidget()
            latency_widget.setBackground('w')
            latency_widget.setTitle('End-to-End Latency')
            latency_widget.setLabel('left', 'Latency (ms)')
            latency_widget.setLabel('bottom', 'Time (s)')
            latency_widget.showGrid(x=True, y=True)
            latency_widget.setFixedHeight(460)
            self._configure_time_axis(latency_widget)
            latency_widget.addLegend(offset=(10, -460))
            
            # 添加不同时延曲线（单窗口）
            total_pen = pg.mkPen(color=(255, 0, 0), width=2)
            self.plots[grpc_port] = latency_widget.plot(pen=total_pen, name='Total')

            sender_pen = pg.mkPen(color=(0, 180, 0), width=2)
            self.sender_plots[grpc_port] = latency_widget.plot(pen=sender_pen, name='Sender')

            network_pen = pg.mkPen(color=(255, 140, 0), width=2)
            self.network_plots[grpc_port] = latency_widget.plot(pen=network_pen, name='Network')

            decoding_pen = pg.mkPen(color=(0, 128, 255), width=2)
            self.decoding_plots[grpc_port] = latency_widget.plot(pen=decoding_pen, name='Decoding&Fitting')

            rendering_pen = pg.mkPen(color=(160, 32, 240), width=2)
            self.rendering_plots[grpc_port] = latency_widget.plot(pen=rendering_pen, name='Rendering')
            
            # 添加统计信息标签
            stats_label = QLabel()
            stats_label.setFont(QFont('Arial', 10))
            stats_label.setAlignment(cast(Any, qt_align('AlignLeft')))
            stats_label.setText("Collecting data...")
            self.stats_labels[grpc_port] = stats_label
            
            # 将控件添加到用户容器
            user_layout.addWidget(latency_widget)
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
        self.status_bar.showMessage('Monitoring started')

    def _trim_sample_window(self, samples):
        if len(samples) > self.max_points:
            del samples[:-self.max_points]

    def _recent_samples(self, samples):
        if not samples:
            return []

        cutoff = time.time() - self.avg_window_seconds
        recent_samples = [sample for sample in samples if sample[0] >= cutoff]
        if recent_samples:
            return recent_samples
        return [samples[-1]]

    def _extract_plot_series(self, samples, base_time=None):
        if not samples:
            return [], []
        # 只保留最近30秒的点绘图
        now = time.time()
        window_cutoff = now - 30.0
        visible = [(t, v) for t, v in samples if t >= window_cutoff]
        if not visible:
            visible = [samples[-1]]
        start_time = base_time if base_time is not None else visible[0][0]
        x_values = [t - start_time for t, _ in visible]
        y_values = [v for _, v in visible]
        return x_values, y_values

    def _average_recent_samples(self, samples):
        recent_values = [value for _, value in self._recent_samples(samples)]
        if not recent_values:
            return 0
        return sum(recent_values) / len(recent_values)

    def _configure_time_axis(self, plot_widget):
        plot_widget.enableAutoRange(y=True)
        plot_widget.setXRange(0, 30)
    
    def update_plots(self):
        # 更新每个用户的图表和统计信息
        for grpc_port in self.plots.keys():
            # 统一x轴起点：用当前时制30s前作为滚动窗口左端，保证各曲线严格对齐
            base_time = time.time() - 30.0

            # 更新延迟图表
            latency_data = self.latency_data.get(grpc_port, [])
            if latency_data:
                latency_x, latency_values = self._extract_plot_series(latency_data, base_time)
                self.plots[grpc_port].setData(latency_x, latency_values)
            else:
                latency_values = []

            # 更新发送端处理时延图表
            sender_data = self.sender_latency_data.get(grpc_port, [])
            if sender_data:
                sender_x, sender_values = self._extract_plot_series(sender_data, base_time)
                self.sender_plots[grpc_port].setData(sender_x, sender_values)
            else:
                sender_values = []

            # 更新网络时延图表
            network_data = self.network_latency_data.get(grpc_port, [])
            if network_data:
                network_x, network_values = self._extract_plot_series(network_data, base_time)
                self.network_plots[grpc_port].setData(network_x, network_values)
            else:
                network_values = []

            # 更新解码拟合时延图表
            decoding_data = self.decoding_latency_data.get(grpc_port, [])
            if decoding_data:
                decoding_x, decoding_values = self._extract_plot_series(decoding_data, base_time)
                self.decoding_plots[grpc_port].setData(decoding_x, decoding_values)
            else:
                decoding_values = []

            # 更新渲染时延图表
            rendering_data = self.rendering_latency_data.get(grpc_port, [])
            if rendering_data:
                rendering_x, rendering_values = self._extract_plot_series(rendering_data, base_time)
                self.rendering_plots[grpc_port].setData(rendering_x, rendering_values)
            else:
                rendering_values = []

            bandwidth_data = self.bandwidth_data.get(grpc_port, [])
            bandwidth_values = [value for _, value in bandwidth_data] if bandwidth_data else []
            
            # 更新统计信息
            stats = self.packet_stats.get(grpc_port, {})
            if stats:
                # 计算当前延迟和平均延迟
                current_latency = latency_values[-1] if latency_values else 0
                avg_latency = self._average_recent_samples(latency_data)
                
                # 计算当前带宽和平均带宽
                current_bandwidth = bandwidth_values[-1] if bandwidth_values else 0
                avg_bandwidth = self._average_recent_samples(bandwidth_data)

                # 计算分段时延统计
                current_sender = sender_values[-1] if sender_values else 0
                avg_sender = self._average_recent_samples(self.sender_latency_data.get(grpc_port, []))
                current_network = network_values[-1] if network_values else 0
                avg_network = self._average_recent_samples(network_data)
                current_decoding = decoding_values[-1] if decoding_values else 0
                avg_decoding = self._average_recent_samples(decoding_data)
                current_rendering = rendering_values[-1] if rendering_values else 0
                avg_rendering = self._average_recent_samples(rendering_data)
                current_bandwidth_kbps = current_bandwidth * 8.0 / 1000.0
                avg_bandwidth_kbps = avg_bandwidth * 8.0 / 1000.0
                
                # 更新统计信息标签
                stats_text = (
                    f"<b>Total Latency:</b> Current: {current_latency:.1f} ms | Avg (last {self.avg_window_seconds}s): {avg_latency:.1f} ms<br>"
                    f"<b>Sender Latency:</b> Current: {current_sender:.1f} ms | Avg: {avg_sender:.1f} ms<br>"
                    f"<b>Network Latency:</b> Current: {current_network:.1f} ms | Avg: {avg_network:.1f} ms<br>"
                    f"<b>Decoding&Fitting Latency:</b> Current: {current_decoding:.1f} ms | Avg: {avg_decoding:.1f} ms<br>"
                    f"<b>Rendering Latency:</b> Current: {current_rendering:.1f} ms | Avg: {avg_rendering:.1f} ms<br>"
                    f"<b>Bandwidth:</b> Current: {current_bandwidth_kbps:.2f} kbps | Average: {avg_bandwidth_kbps:.2f} kbps<br>"
                    f"<b>Data:</b> Total Packets: {stats['total_packets']} | Total Size: {stats['total_bytes']/1024:.2f} KB"
                )
                self.stats_labels[grpc_port].setText(stats_text)
    
    def add_latency_data(self, grpc_port, latency):
        """添加新的延迟数据点"""
        if grpc_port in self.latency_data:
            self.latency_data[grpc_port].append((time.time(), latency))

    def add_segment_data(self, grpc_port, sender_ms=None, network_ms=None, decoding_ms=None, rendering_ms=None):
        """添加分段时延数据点"""
        sample_time = time.time()
        if sender_ms is not None and grpc_port in self.sender_latency_data:
            self.sender_latency_data[grpc_port].append((sample_time, sender_ms))

        if network_ms is not None and grpc_port in self.network_latency_data:
            self.network_latency_data[grpc_port].append((sample_time, network_ms))

        if decoding_ms is not None and grpc_port in self.decoding_latency_data:
            self.decoding_latency_data[grpc_port].append((sample_time, decoding_ms))

        if rendering_ms is not None and grpc_port in self.rendering_latency_data:
            self.rendering_latency_data[grpc_port].append((sample_time, rendering_ms))
    
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
                self.bandwidth_data[grpc_port].append((current_time, current_bandwidth))
                if len(self.bandwidth_data[grpc_port]) > self.max_points:
                    self.bandwidth_data[grpc_port].pop(0)
                    
                stats['last_second_bytes'] = current_bandwidth
                stats['last_update'] = current_time


def parse_timing_meta(ext_data_bytes):
    """从payload.extData中解析t_begin/t_encode"""
    if not ext_data_bytes:
        return {}
    try:
        if isinstance(ext_data_bytes, bytes):
            text = ext_data_bytes.decode('utf-8', errors='ignore').strip()
        else:
            text = str(ext_data_bytes).strip()
        if not text or text == '\x00':
            return {}
        return json.loads(text)
    except Exception:
        return {}


class StreamingBaselineSplineUpsampler:
    """流式 baseline 样条：收到第 k 帧后拟合 [k-1,k]，并仅发送 [k-1,k) 段内上采样点。"""

    def __init__(self, upsample_factor=1):
        self.upsample_factor = max(1, int(upsample_factor))
        self.prev_vec = None
        self.prev_prev_vec = None
        self.prev_ts_ms = None
        self.prev_seg_right_v = None

    def _parse_limb(self, limb_data_str):
        vec = np.fromstring(limb_data_str, sep=',', dtype=np.float64)
        if vec.size == 0:
            raise ValueError("empty limb data")
        return vec

    def _vec_to_csv(self, vec):
        return ','.join(f"{float(v):.6f}" for v in vec)

    def generate_upsampled_frames(self, curr_ts_ms, limb_data_str):
        """返回[(ts_ms, limb_csv), ...]。首帧无可拟合区间时返回空列表。"""
        curr_vec = self._parse_limb(limb_data_str)

        # 首帧：仅缓存，不输出。
        if self.prev_vec is None or self.prev_ts_ms is None:
            self.prev_vec = curr_vec
            self.prev_ts_ms = int(curr_ts_ms)
            return []

        dt = max((int(curr_ts_ms) - int(self.prev_ts_ms)) / 1000.0, 1e-6)

        if self.prev_prev_vec is None:
            v_prev_seg = (curr_vec - self.prev_vec) / dt
        else:
            v_prev_seg = self.prev_seg_right_v if self.prev_seg_right_v is not None else (curr_vec - self.prev_vec) / dt

        if self.prev_prev_vec is None:
            v_curr_seg = v_prev_seg
        else:
            # baseline(history accel extrapolation): (3*x_k - 4*x_{k-1} + x_{k-2}) / (2*dt)
            v_curr_seg = (3.0 * curr_vec - 4.0 * self.prev_vec + self.prev_prev_vec) / (2.0 * dt)

        coeff = cubic_hermite_coefficients(
            self.prev_vec,
            v_prev_seg,
            curr_vec,
            v_curr_seg,
            dt,
        ).T  # (channels, 4)

        # 严格一帧延迟：只发 [k-1, k) 点，不发右端点 k。
        # 当 upsample_factor=1 时，仅发送 ratio=0 的左端点（即 k-1 帧）。
        out_frames = []
        for i in range(self.upsample_factor):
            ratio = i / float(self.upsample_factor)
            tau = dt * ratio
            # y(tau) = a*tau^3 + b*tau^2 + c*tau + d
            y = ((coeff[:, 0] * tau + coeff[:, 1]) * tau + coeff[:, 2]) * tau + coeff[:, 3]
            ts_i = int(self.prev_ts_ms + (int(curr_ts_ms) - int(self.prev_ts_ms)) * ratio)
            out_frames.append((ts_i, self._vec_to_csv(y)))

        self.prev_prev_vec = self.prev_vec
        self.prev_vec = curr_vec
        self.prev_ts_ms = int(curr_ts_ms)
        self.prev_seg_right_v = v_curr_seg.copy()

        return out_frames


def send_combined_data(face_data_str, limb_data_str, timestamp, socket_port):
    """将extDesc(时间戳)、facedata和limbdata拼接到一起，用逗号分隔"""
    # 直接拼接时间戳、face_data和limb_data，用逗号分隔
    data_str = str(timestamp) + "," + face_data_str + "," + limb_data_str
    # 建立TCP连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", socket_port))
    
    # 发送数据
    client.sendall(data_str.encode('utf-8'))
    
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

def grpc_thread(grpc_port, socket_port, latency_monitor=None, point_cloud_grpc_port=None,
                segment_cache=None, cache_lock=None, spline_upsample=1):
    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, grpc_port))
    server_thread.start()

    is_point_cloud_port = (point_cloud_grpc_port is not None and grpc_port == point_cloud_grpc_port)
    if is_point_cloud_port:
        print(f"[点云数据] 启动点云数据专用服务: gRPC端口 {grpc_port}, Socket端口 {socket_port}")

    spline_upsampler = None
    if (not is_point_cloud_port) and int(spline_upsample) >= 1:
        spline_upsampler = StreamingBaselineSplineUpsampler(upsample_factor=int(spline_upsample))
        print(f"[gRPC Port {grpc_port}] baseline 样条上采样已启用: x{int(spline_upsample)}")

    # 持久化 TCP 连接：避免每帧建立新连接带来的握手延迟
    persistent_sock = None

    def _send_persistent(data_str):
        nonlocal persistent_sock
        encoded = (data_str + "\n").encode('utf-8')
        for _ in range(2):  # 失败时重连一次
            if persistent_sock is None:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("127.0.0.1", socket_port))
                    persistent_sock = s
                    print(f"[Socket Port {socket_port}] 持久连接已建立")
                except Exception as e:
                    print(f"[Socket Port {socket_port}] 连接失败: {e}")
                    return
            try:
                persistent_sock.sendall(encoded)
                return
            except Exception:
                try:
                    persistent_sock.close()
                except Exception:
                    pass
                persistent_sock = None
                print(f"[Socket Port {socket_port}] 连接断开，尝试重连")

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
                timestamp = getattr(payload_rec, 'extDesc', '')
                
                # 判断是否为点云数据
                if is_point_cloud_port:
                    # 处理点云数据
                    point_data_bytes = getattr(payload_rec, 'pointData', b'')
                    data_size = len(point_data_bytes)
                    
                    # 添加数据包信息到UI
                    if latency_monitor:
                        latency_monitor.add_packet_data(grpc_port, data_size)
                    
                    # 解析并发送点云数据
                    send_point_cloud_data(point_data_bytes, socket_port)
                else:
                    # 处理常规数据（面部和肢体数据）
                    face_data_bytes = getattr(payload_rec, 'faceData', b'')
                    limb_data_bytes = getattr(payload_rec, 'limbData', b'')
                    timing_meta = parse_timing_meta(getattr(payload_rec, 'extData', b''))
                    
                    # 直接解码字节串为字符串
                    face_data_str = face_data_bytes.decode('utf-8')
                    limb_data_str = limb_data_bytes.decode('utf-8')
                    
                    # 计算数据大小
                    data_size = len(face_data_bytes) + len(limb_data_bytes)
                    
                    # 添加数据包信息到UI
                    if latency_monitor:
                        latency_monitor.add_packet_data(grpc_port, data_size)

                    # 计算分段时延
                    t_begin_ms = None
                    t_encode_ms = None
                    t_net_ms = None
                    if timing_meta:
                        t_begin_ms = timing_meta.get('t_begin')
                        t_encode_ms = timing_meta.get('t_encode')

                    # 统一当前帧时间戳，优先使用 t_begin（用于与反馈匹配）
                    frame_ts_ms = None
                    if isinstance(t_begin_ms, int):
                        frame_ts_ms = t_begin_ms
                    else:
                        ts_text = str(timestamp)
                        if ts_text.isdigit():
                            frame_ts_ms = int(ts_text)

                    # t_net：收到第 k 帧后，进入样条拟合/上采样之前。
                    if isinstance(t_encode_ms, int):
                        t_net_ms = int(time.time() * 1000)

                    send_items = []
                    if spline_upsampler is not None and frame_ts_ms is not None:
                        try:
                            send_items = spline_upsampler.generate_upsampled_frames(frame_ts_ms, limb_data_str)
                        except Exception as e:
                            print(f"[gRPC Port {grpc_port}] 样条拟合失败，回退原始数据: {e}")
                            send_items = [(frame_ts_ms, limb_data_str)]
                    else:
                        if frame_ts_ms is None:
                            frame_ts_ms = int(time.time() * 1000)
                        send_items = [(frame_ts_ms, limb_data_str)]

                    # 通过持久化连接发送合并后的数据（换行符分隔帧）
                    for send_ts_ms, send_limb_data_str in send_items:
                        _send_persistent(str(send_ts_ms) + "," + face_data_str + "," + send_limb_data_str)

                    if segment_cache is not None and t_begin_ms is not None:
                        snapshot = {
                            't_begin': t_begin_ms,
                            't_encode': t_encode_ms,
                            't_net': t_net_ms
                        }
                        if cache_lock:
                            with cache_lock:
                                if grpc_port not in segment_cache:
                                    segment_cache[grpc_port] = {}
                                segment_cache[grpc_port][t_begin_ms] = snapshot
                                # 流式传输下保留最近300帧，避免缓存无限增长
                                if len(segment_cache[grpc_port]) > 300:
                                    oldest_key = min(segment_cache[grpc_port].keys())
                                    del segment_cache[grpc_port][oldest_key]
                        else:
                            if grpc_port not in segment_cache:
                                segment_cache[grpc_port] = {}
                            segment_cache[grpc_port][t_begin_ms] = snapshot
                            if len(segment_cache[grpc_port]) > 300:
                                oldest_key = min(segment_cache[grpc_port].keys())
                                del segment_cache[grpc_port][oldest_key]

            except AttributeError as e:
                print(f"[gRPC Port {grpc_port}] AttributeError: {e}")
            except Exception as e:
                print(f"[gRPC Port {grpc_port}] 处理数据错误: {e}")

def latency_feedback_server(feedback_port, port_mappings, latency_monitor=None,
                            latency_compensation=None, feedback_offset=1000,
                            segment_cache=None, cache_lock=None):
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
                if feedback.startswith("timing:"):
                    payload = feedback[len("timing:"):]
                    parts = payload.split(',')
                    if len(parts) == 3:
                        t_begin_ms = int(parts[0])
                        t_transit_ms = int(parts[1])
                        t_final_ms = int(parts[2])

                        segment_snapshot = None
                        if segment_cache is not None and grpc_port:
                            if cache_lock:
                                with cache_lock:
                                    port_cache = segment_cache.get(grpc_port, {})
                                    segment_snapshot = port_cache.pop(t_begin_ms, None)
                            else:
                                port_cache = segment_cache.get(grpc_port, {})
                                segment_snapshot = port_cache.pop(t_begin_ms, None)

                        if segment_snapshot and latency_monitor and grpc_port:
                            t_encode_ms = segment_snapshot.get('t_encode')
                            t_net_ms = segment_snapshot.get('t_net')
                            if isinstance(t_encode_ms, int) and isinstance(t_net_ms, int):
                                sender_ms = max(0, t_encode_ms - t_begin_ms)
                                network_ms = max(0, t_net_ms - t_encode_ms)
                                decoding_ms = max(0, t_transit_ms - t_net_ms)
                                rendering_ms = max(0, t_final_ms - t_transit_ms)
                                total_ms = max(0, t_final_ms - t_begin_ms - compensation)

                                latency_monitor.add_latency_data(grpc_port, total_ms)
                                latency_monitor.add_segment_data(
                                    grpc_port,
                                    sender_ms=sender_ms,
                                    network_ms=network_ms,
                                    decoding_ms=decoding_ms,
                                    rendering_ms=rendering_ms
                                )
                    
            client.close()
        except Exception as e:
            print(f"[延迟反馈] 接收反馈数据错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="gRPC 转 Socket 转发与监控")
    parser.add_argument("--grpc_ports", nargs='+', type=int, help="gRPC端口列表")
    parser.add_argument("--socket_ports", nargs='+', type=int, help="Socket端口列表")
    parser.add_argument("--point_cloud_grpc_port", type=int, help="点云数据gRPC端口")
    parser.add_argument("--feedback_offset", type=int, default=1000, help="反馈端口与Socket端口的偏移量")
    parser.add_argument("--spline_upsample", type=int, default=1, help="baseline样条段内上采样倍数（>=1）")
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
    segment_cache = {}
    cache_lock = threading.Lock()

    threads = []
    for grpc_port, socket_port in port_mappings:
        t = threading.Thread(target=grpc_thread,
                             args=(grpc_port, socket_port, latency_monitor, point_cloud_grpc_port,
                                   segment_cache, cache_lock, args.spline_upsample))
        t.start()
        threads.append(t)

    for feedback_port in feedback_ports:
        t = threading.Thread(target=latency_feedback_server,
                             args=(feedback_port, port_mappings, latency_monitor, latency_compensation,
                                   args.feedback_offset, segment_cache, cache_lock))
        t.start()
        threads.append(t)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()