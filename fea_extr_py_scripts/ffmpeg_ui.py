import sys
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QStatusBar, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from ffmpeg_client import FFmpegClient
import datetime

# 调试开关
DEBUG = False

def log(message, level="INFO"):
    """
    日志输出函数，根据调试开关决定是否输出
    
    参数:
        message (str): 日志信息
        level (str): 日志级别 (DEBUG, INFO, ERROR 等)
    """
    if DEBUG or level == "ERROR":
        time_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[UI][{level}][{time_str}] {message}")


class VideoWindow(QMainWindow):
    """主窗口类，用于显示视频及性能指标"""
    
    def __init__(self, window_size, video_display_size):
        """
        初始化视频窗口
        
        参数:
            window_size (tuple): 窗口大小 (width, height)
            video_display_size (tuple): 视频显示区域大小 (width, height)
        """
        super().__init__()
        self.window_width, self.window_height = window_size
        self.video_width, self.video_height = video_display_size
        
        log(f"初始化窗口: {window_size}, 视频区域: {video_display_size}", "DEBUG")
        self.setup_ui()
        self.is_streaming = True
        self.client = None
        log("窗口初始化完成", "DEBUG")
    
    def setup_ui(self):
        """设置UI界面元素"""
        # 窗口基本设置
        self.setWindowTitle("720p视频流显示与分析")
        self.setGeometry(100, 100, self.window_width, self.window_height)
        log("开始设置UI界面", "DEBUG")
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建视频和数据区域
        self.setup_video_and_stats_area(main_layout)
        
        # 创建图表区域
        self.setup_charts_area(main_layout)
        
        # 创建状态栏
        self.setup_status_bar()
        log("UI界面设置完成", "DEBUG")
    
    # 其他方法保持不变，只在关键位置添加日志...
    def setup_video_and_stats_area(self, main_layout):
        """设置视频和统计数据区域"""
        # 上半部分：视频和基本信息
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout, 3)  # 占3份高度
        
        # 视频部分
        video_group = QGroupBox("视频流")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(self.video_width, self.video_height)
        video_layout.addWidget(self.video_label)
        
        # 控制按钮
        self.control_btn = QPushButton("停止")
        self.control_btn.clicked.connect(self.toggle_streaming)
        video_layout.addWidget(self.control_btn)
        
        top_layout.addWidget(video_group, 7)  # 占7份宽度
        
        # 实时数据显示部分
        stats_group = QGroupBox("实时数据")
        stats_layout = QGridLayout(stats_group)
        
        # 创建统计数据标签
        self.create_stat_labels(stats_layout)
        
        top_layout.addWidget(stats_group, 3)  # 占3份宽度
        log("视频和统计区域设置完成", "DEBUG")
    
    def create_stat_labels(self, layout):
        """创建统计数据标签"""
        # FPS显示
        layout.addWidget(QLabel("当前FPS:"), 0, 0)
        self.fps_label = QLabel("0.00")
        layout.addWidget(self.fps_label, 0, 1)
        
        # 码率显示
        layout.addWidget(QLabel("当前码率:"), 1, 0)
        self.bitrate_label = QLabel("0.00 Mbps")
        layout.addWidget(self.bitrate_label, 1, 1)
        
        # 时延显示
        layout.addWidget(QLabel("当前时延:"), 2, 0)
        self.latency_label = QLabel("0.00 ms")
        layout.addWidget(self.latency_label, 2, 1)
        
        # 平均值显示
        layout.addWidget(QLabel("平均FPS:"), 3, 0)
        self.avg_fps_label = QLabel("0.00")
        layout.addWidget(self.avg_fps_label, 3, 1)
        
        layout.addWidget(QLabel("平均码率:"), 4, 0)
        self.avg_bitrate_label = QLabel("0.00 Mbps")
        layout.addWidget(self.avg_bitrate_label, 4, 1)
        
        layout.addWidget(QLabel("平均时延:"), 5, 0)
        self.avg_latency_label = QLabel("0.00 ms")
        layout.addWidget(self.avg_latency_label, 5, 1)
        log("统计数据标签创建完成", "DEBUG")
    
    def setup_charts_area(self, main_layout):
        """设置图表显示区域"""
        # 下半部分：图表显示
        chart_group = QGroupBox("30秒历史数据")
        chart_layout = QVBoxLayout(chart_group)
        main_layout.addWidget(chart_group, 2)  # 占2份高度
        
        # 创建绘图部件
        self.graph_widget = pg.GraphicsLayoutWidget()
        chart_layout.addWidget(self.graph_widget)
        
        # 创建三个子图
        self.create_charts()
        log("图表区域设置完成", "DEBUG")
    
    def create_charts(self):
        """创建性能指标图表"""
        # FPS图表
        self.fps_plot = self.graph_widget.addPlot(row=0, col=0, title="FPS历史")
        self.fps_plot.setLabel('left', 'FPS')
        self.fps_plot.setLabel('bottom', '时间 (秒)')
        self.fps_curve = self.fps_plot.plot(pen='g')
        
        # 码率图表
        self.bitrate_plot = self.graph_widget.addPlot(row=0, col=1, title="码率历史 (Mbps)")
        self.bitrate_plot.setLabel('left', '码率 (Mbps)')
        self.bitrate_plot.setLabel('bottom', '时间 (秒)')
        self.bitrate_curve = self.bitrate_plot.plot(pen='r')
        
        # 时延图表
        self.latency_plot = self.graph_widget.addPlot(row=0, col=2, title="时延历史 (ms)")
        self.latency_plot.setLabel('left', '时延 (ms)')
        self.latency_plot.setLabel('bottom', '时间 (秒)')
        self.latency_curve = self.latency_plot.plot(pen='b')
        log("性能指标图表创建完成", "DEBUG")
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        log("状态栏设置完成", "DEBUG")
    
    def update_frame(self, frame, fps, bitrate, latency):
        """
        更新UI界面中的视频帧和当前性能指标
        
        参数:
            frame (numpy.ndarray): 视频帧
            fps (float): 帧率
            bitrate (float): 码率(Mbps)
            latency (float): 时延(ms)
        """
        if DEBUG:
            log(f"更新帧: FPS={fps:.2f}, 码率={bitrate:.2f}Mbps, 延迟={latency:.2f}ms", "DEBUG")
            
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)
        
        # 更新当前性能指标
        self.fps_label.setText(f"{fps:.2f}")
        self.bitrate_label.setText(f"{bitrate:.2f} Mbps")
        self.latency_label.setText(f"{latency:.2f} ms")
    
    def update_stats(self, avg_fps, avg_bitrate, avg_latency, fps_data, bitrate_data, latency_data, time_axis):
        """
        更新统计数据和图表
        
        参数:
            avg_fps (float): 平均帧率
            avg_bitrate (float): 平均码率(Mbps)
            avg_latency (float): 平均时延(ms)
            fps_data (list): 帧率历史数据
            bitrate_data (list): 码率历史数据
            latency_data (list): 时延历史数据
            time_axis (list): 时间轴数据
        """
        if DEBUG:
            log(f"更新统计: 平均FPS={avg_fps:.2f}, 平均码率={avg_bitrate:.2f}Mbps, 平均延迟={avg_latency:.2f}ms", "DEBUG")
            
        # 更新平均值
        self.avg_fps_label.setText(f"{avg_fps:.2f}")
        self.avg_bitrate_label.setText(f"{avg_bitrate:.2f} Mbps")
        self.avg_latency_label.setText(f"{avg_latency:.2f} ms")
        
        # 更新图表
        self.fps_curve.setData(time_axis, fps_data)
        self.bitrate_curve.setData(time_axis, bitrate_data)
        self.latency_curve.setData(time_axis, latency_data)
    
    def toggle_streaming(self):
        """切换流状态"""
        self.is_streaming = not self.is_streaming
        status = "启动" if self.is_streaming else "停止"
        log(f"切换流状态: {status}", "INFO")
        
        if self.is_streaming:
            self.control_btn.setText("停止")
            self.status_label.setText("正在接收视频流")
        else:
            self.control_btn.setText("启动")
            self.status_label.setText("已暂停视频流")
        
        # 通知客户端流状态更改
        if self.client:
            self.client.set_streaming_state(self.is_streaming)
    
    def set_client(self, client):
        """
        设置客户端引用
        
        参数:
            client (FFmpegClient): 解码客户端实例
        """
        log("设置客户端引用", "DEBUG")
        self.client = client
    
    def closeEvent(self, event):
        """
        窗口关闭事件，确保资源被清理
        
        参数:
            event (QCloseEvent): 关闭事件
        """
        log("窗口关闭，清理资源", "INFO")
        if self.client:
            self.client.stop()
        event.accept()


# 全局回调函数
def frame_callback(frame, fps, bitrate, latency):
    """
    转发帧到UI的回调函数
    
    参数:
        frame (numpy.ndarray): 视频帧
        fps (float): 帧率
        bitrate (float): 码率(Mbps)
        latency (float): 时延(ms)
    """
    window.update_frame(frame, fps, bitrate, latency)

def stats_callback(avg_fps, avg_bitrate, avg_latency, fps_data, bitrate_data, latency_data, time_axis):
    """
    转发统计数据到UI的回调函数
    
    参数:
        avg_fps (float): 平均帧率
        avg_bitrate (float): 平均码率(Mbps)
        avg_latency (float): 平均时延(ms)
        fps_data (list): 帧率历史数据
        bitrate_data (list): 码率历史数据
        latency_data (list): 时延历史数据
        time_axis (list): 时间轴数据
    """
    window.update_stats(avg_fps, avg_bitrate, avg_latency, fps_data, bitrate_data, latency_data, time_axis)


if __name__ == '__main__':
    # 配置参数
    CONFIG = {
        "window_size": (1280, 800),          # 窗口大小
        "video_display_size": (640, 360),    # 视频显示区域大小
        "grpc_server": "183.173.117.138",    # gRPC服务器地址
        "grpc_port": 50051,                  # gRPC端口
        "sdp_file": "stream.sdp",            # SDP文件名
        "video_resolution": (1280, 720),     # 视频分辨率
        "history_length": 30 * 10,           # 历史数据保存点数（30秒×10点/秒）
        "debug": DEBUG                        # 调试模式
    }
    
    log("程序启动", "INFO")
    log(f"配置参数: {CONFIG}", "DEBUG")
    
    # 启动应用
    app = QApplication(sys.argv)
    window = VideoWindow(CONFIG["window_size"], CONFIG["video_display_size"])
    window.show()
    log("UI窗口已显示", "INFO")
    
    # 创建客户端并设置回调
    log("创建FFmpeg客户端...", "DEBUG")
    client = FFmpegClient(
        frame_callback=frame_callback, 
        stats_callback=stats_callback,
        server_address=f"{CONFIG['grpc_server']}:{CONFIG['grpc_port']}",
        sdp_file=CONFIG["sdp_file"],
        video_resolution=CONFIG["video_resolution"],
        history_length=CONFIG["history_length"],
        debug=CONFIG["debug"]
    )
    window.set_client(client)
    
    # 在单独的线程中启动客户端
    log("在单独线程中启动客户端", "INFO")
    client_thread = threading.Thread(target=client.start)
    client_thread.daemon = True
    client_thread.start()
    
    # 运行PyQt应用
    log("进入PyQt主事件循环", "DEBUG")
    sys.exit(app.exec_())