import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys
import shlex

class HolographicClassroomGUI:
    def __init__(self, root):
        self.root = root
        self.base_title = "基于点云和全息数字人生成的全息课堂演示系统V1.0"
        self.root.title(self.base_title)
        self.root.geometry("600x600")

        # 虚拟环境路径（假设是conda环境，需根据实际情况调整）
        self.venv_name = "face_detec"  # 替换为实际环境名
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 创建输入字段
        self.create_widgets()

    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        self.content_container = ttk.Frame(self.root)
        self.content_container.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)
        self.content_container.columnconfigure(0, weight=1)

        self.main_menu_frame = self._create_main_menu()
        self.server_config_frame = self._create_server_config()
        self.teacher_config_frame = self._create_teacher_config()
        self.student_config_frame = self._create_student_config()

        self.config_frames = {
            "server": self.server_config_frame,
            "teacher": self.teacher_config_frame,
            "student": self.student_config_frame,
        }

        self.show_main_menu()

        ttk.Label(self.root, text="输出区域").grid(row=1, column=0, padx=10, sticky=tk.W)
        self.output_text = tk.Text(self.root, height=10, width=70)
        self.output_text.grid(row=2, column=0, padx=10, pady=(0, 10), sticky=tk.NSEW)

        ttk.Button(self.root, text="使用说明", command=self.show_help).grid(
            row=3, column=0, padx=10, pady=(0, 10), sticky=tk.W
        )
        ttk.Button(self.root, text="退出程序", command=self.root.quit).grid(
            row=3, column=0, padx=10, pady=(0, 10), sticky=tk.E
        )

    def _create_main_menu(self):
        frame = ttk.Frame(self.content_container)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="请选择配置模块").grid(row=0, column=0, pady=(0, 20))
        ttk.Button(frame, text="服务器配置", command=lambda: self.show_config("server")).grid(row=1, column=0, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="全息教师配置", command=lambda: self.show_config("teacher")).grid(row=2, column=0, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="全息学生配置", command=lambda: self.show_config("student")).grid(row=3, column=0, pady=5, sticky=tk.EW)

        return frame

    def _create_server_config(self):
        frame = ttk.Frame(self.content_container)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="返回上一级", command=self.show_main_menu).grid(row=0, column=0, pady=(0, 10), sticky=tk.W)

        server_frame = ttk.LabelFrame(frame, text="服务器转发配置")
        server_frame.grid(row=1, column=0, sticky=tk.NSEW)
        server_frame.columnconfigure(1, weight=1)

        ttk.Label(server_frame, text="端口映射 (gRPC → Socket)").grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        ports_frame = ttk.Frame(server_frame)
        ports_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        ports_frame.columnconfigure(2, weight=1)

        ttk.Label(ports_frame, text="角色").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(ports_frame, text="gRPC端口").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(ports_frame, text="Socket端口").grid(row=0, column=2, padx=5, pady=2)

        default_pairs = [
            (50051, 8890),
            (50052, 8891),
            (50053, 8892),
            (50054, 8893),
            (50055, 8894),
        ]
        self.port_entry_pairs = []
        total_pairs = len(default_pairs)
        for idx, (grpc_port, socket_port) in enumerate(default_pairs, start=1):
            role_label = "全息教师" if idx == total_pairs else f"全息学生{idx}"
            ttk.Label(ports_frame, text=role_label).grid(row=idx, column=0, padx=5, pady=2, sticky=tk.W)

            grpc_entry = ttk.Entry(ports_frame, width=10)
            grpc_entry.insert(0, str(grpc_port))
            grpc_entry.grid(row=idx, column=1, padx=5, pady=2, sticky=tk.EW)

            socket_entry = ttk.Entry(ports_frame, width=10)
            socket_entry.insert(0, str(socket_port))
            socket_entry.grid(row=idx, column=2, padx=5, pady=2, sticky=tk.EW)

            self.port_entry_pairs.append((grpc_entry, socket_entry))

        ttk.Button(server_frame, text="启动服务器转发", command=self.run_server).grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky=tk.EW)
        ttk.Button(server_frame, text="退出服务器程序", command=self.stop_server).grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky=tk.EW)

        return frame

    def _create_teacher_config(self):
        frame = ttk.Frame(self.content_container)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="返回上一级", command=self.show_main_menu).grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        teacher_frame = ttk.LabelFrame(frame, text="全息教师发送端配置")
        teacher_frame.grid(row=1, column=0, sticky=tk.NSEW)
        teacher_frame.columnconfigure(1, weight=1)

        ttk.Label(teacher_frame, text="服务器地址:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.teacher_server_addr_entry = ttk.Entry(teacher_frame)
        self.teacher_server_addr_entry.insert(0, "127.0.0.1")
        self.teacher_server_addr_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Label(teacher_frame, text="端口号:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.teacher_port_entry = ttk.Entry(teacher_frame)
        self.teacher_port_entry.insert(0, "50055")
        self.teacher_port_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Button(teacher_frame, text="运行全息教师发送端", command=self.run_teacher).grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky=tk.EW)
        ttk.Button(teacher_frame, text="退出全息教师发送端程序", command=self.stop_teacher).grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky=tk.EW)

        return frame

    def _create_student_config(self):
        frame = ttk.Frame(self.content_container)
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="返回上一级", command=self.show_main_menu).grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        sender_frame = ttk.LabelFrame(frame, text="全息学生发送端配置")
        sender_frame.grid(row=1, column=0, sticky=tk.NSEW)
        sender_frame.columnconfigure(1, weight=1)

        ttk.Label(sender_frame, text="服务器地址:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.server_addr_entry = ttk.Entry(sender_frame)
        self.server_addr_entry.insert(0, "127.0.0.1")
        self.server_addr_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Label(sender_frame, text="端口号:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.port_entry = ttk.Entry(sender_frame)
        self.port_entry.insert(0, "50051")
        self.port_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Label(sender_frame, text="姿势模型路径:").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        script_dir = self.script_dir
        default_pose_path = os.path.join(script_dir, "data", "pose_landmarker_full.task")
        self.pose_model_entry = ttk.Entry(sender_frame)
        self.pose_model_entry.insert(0, default_pose_path)
        self.pose_model_entry.grid(row=2, column=1, padx=10, pady=5, sticky=tk.EW)

        ttk.Label(sender_frame, text="面部模型路径:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        self.face_model_entry = ttk.Entry(sender_frame)
        default_face_path = os.path.join(script_dir, "data", "face_landmarker_v2_with_blendshapes.task")
        self.face_model_entry.insert(0, default_face_path)
        self.face_model_entry.grid(row=3, column=1, padx=10, pady=5, sticky=tk.EW)

        self.debug_var = tk.BooleanVar()
        ttk.Checkbutton(sender_frame, text="调试模式", variable=self.debug_var).grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        ttk.Button(sender_frame, text="运行全息学生发送端", command=self.run_sender).grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky=tk.EW)
        ttk.Button(sender_frame, text="退出全息学生发送端程序", command=self.stop_sender).grid(row=6, column=0, columnspan=2, padx=10, pady=(0, 10), sticky=tk.EW)

        return frame

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("420x360")
        help_text = (
            "根据运行演示系统的设备是服务器、全息教师还是全息学生设备来配置相应参数并启动程序。\n"
            "服务器转发配置：设置各角色的 gRPC → Socket 映射，启动/停止转发服务。\n"
            "全息教师发送端配置：指定服务器地址与端口，运行/退出教师端点云推送脚本。\n"
            "全息学生发送端配置：配置学生端服务器、端口及姿势/面部模型路径，可勾选调试模式，运行/退出脚本。\n"
            "输出区域：显示操作日志与状态信息。\n"
            "使用说明：打开此窗口。\n"
            "退出程序：关闭图形界面。"
        )
        ttk.Label(help_window, text=help_text, wraplength=380, justify=tk.LEFT).pack(
            padx=15, pady=15, anchor=tk.W
        )

    def run_script(self):
        server_addr = self.server_addr_entry.get()
        port_num = int(self.port_entry.get())
        pose_model_path = self.pose_model_entry.get()
        face_model_path = self.face_model_entry.get()
        debug = self.debug_var.get()

        script_path = os.path.join(os.path.dirname(__file__), "fea_extr_py_scripts", "grpc_avatar_fea_sender.py")

        # 使用conda run在虚拟环境中运行脚本
        cmd = [
            "conda", "run", "-n", self.venv_name, sys.executable, script_path,
            "--server_addr", server_addr,
            "--port_num", str(port_num),
            "--pose_model_path", pose_model_path,
            "--face_model_path", face_model_path
        ]
        if debug:
            cmd.append("--debug")

        try:
            quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
            escaped_cmd = quoted_cmd.replace('"', r'\"')
            apple_script = f'''tell application "Terminal" to do script "{escaped_cmd}"'''
            subprocess.run(["osascript", "-e", apple_script], check=True)
            self.output_text.insert(tk.END, "脚本已在新终端窗口中启动。\n")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("错误", f"运行脚本失败: {str(e)}")

    def run_sender(self):
        server_addr = self.server_addr_entry.get()
        try:
            port_num = int(self.port_entry.get())
        except ValueError:
            messagebox.showerror("错误", "端口号必须为数字")
            return

        pose_model_path = self.pose_model_entry.get()
        face_model_path = self.face_model_entry.get()
        debug = self.debug_var.get()
        script_path = os.path.join(self.script_dir, "fea_extr_py_scripts", "grpc_avatar_fea_sender.py")

        cmd = [
            "conda", "run", "-n", self.venv_name, sys.executable, script_path,
            "--server_addr", server_addr,
            "--port_num", str(port_num),
            "--pose_model_path", pose_model_path,
            "--face_model_path", face_model_path
        ]
        if debug:
            cmd.append("--debug")

        self._launch_in_terminal(cmd, "全息学生发送端脚本已在新终端窗口中启动。")

    def run_server(self):
        try:
            port_pairs = self._collect_port_pairs()
        except ValueError as exc:
            messagebox.showerror("错误", str(exc))
            return

        grpc_ports = [pair[0] for pair in port_pairs]
        socket_ports = [pair[1] for pair in port_pairs]

        server_script_path = os.path.join(self.script_dir, "fea_extr_py_scripts", "grpc2socket.py")
        cmd = [
            "conda", "run", "-n", self.venv_name, sys.executable, server_script_path,
            "--grpc_ports", *grpc_ports,
            "--socket_ports", *socket_ports,
            "--point_cloud_grpc_port", grpc_ports[-1]
        ]
        self._launch_in_terminal(cmd, "服务器转发程序已在新终端窗口中启动。")

    def run_teacher(self):
        server_addr = self.teacher_server_addr_entry.get()
        try:
            port_num = int(self.teacher_port_entry.get())
        except ValueError:
            messagebox.showerror("错误", "端口号必须为数字")
            return

        script_path = os.path.join(self.script_dir, "point_clouds_reconstruction", "point_clouds_reconstruction.py")
        if not os.path.exists(script_path):
            messagebox.showerror("错误", f"未找到全息教师发送端脚本: {script_path}")
            return

        cmd = [
            "conda", "run", "-n", self.venv_name, sys.executable, script_path,
            "--server_addr", server_addr,
            "--port_num", str(port_num),
        ]
        self._launch_in_terminal(cmd, "全息教师发送端脚本已在新终端窗口中启动。")

    def _collect_port_pairs(self):
        port_pairs = []
        for grpc_entry, socket_entry in self.port_entry_pairs:
            grpc_text = grpc_entry.get().strip()
            socket_text = socket_entry.get().strip()
            if not grpc_text and not socket_text:
                continue
            if not grpc_text or not socket_text:
                raise ValueError("每个端口映射需同时提供 gRPC 与 Socket 端口")
            port_pairs.append((str(int(grpc_text)), str(int(socket_text))))
        if not port_pairs:
            raise ValueError("至少配置一组端口映射")
        return port_pairs

    def _launch_in_terminal(self, cmd, success_message):
        try:
            quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
            escaped_cmd = quoted_cmd.replace('"', r'\"')
            apple_script = f'''tell application "Terminal" to do script "{escaped_cmd}"'''
            subprocess.run(["osascript", "-e", apple_script], check=True)
            self.output_text.insert(tk.END, success_message + "\n")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("错误", f"运行脚本失败: {e}")

    def stop_sender(self):
        script_path = os.path.join(self.script_dir, "fea_extr_py_scripts", "grpc_avatar_fea_sender.py")
        self._terminate_process(script_path, "全息学生发送端")

    def stop_teacher(self):
        script_path = os.path.join(self.script_dir, "point_clouds_reconstruction", "point_clouds_reconstruction.py")
        self._terminate_process(script_path, "全息教师发送端")

    def stop_server(self):
        script_path = os.path.join(self.script_dir, "fea_extr_py_scripts", "grpc2socket.py")
        self._terminate_process(script_path, "服务器转发")

    def _terminate_process(self, script_path, label):
        try:
            result = subprocess.run(["pkill", "-f", script_path], check=False)
            if result.returncode == 0:
                self.output_text.insert(tk.END, f"{label}程序已结束。\n")
            else:
                self.output_text.insert(tk.END, f"{label}程序未在运行。\n")
        except Exception as exc:
            messagebox.showerror("错误", f"结束{label}程序失败: {exc}")

    def _hide_all_config_frames(self):
        frames = [self.main_menu_frame, *self.config_frames.values()]
        for frame in frames:
            frame.grid_forget()

    def show_main_menu(self):
        self._hide_all_config_frames()
        self.main_menu_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.root.title(self.base_title)

    def show_config(self, frame_key):
        titles = {
            "server": "服务器配置",
            "teacher": "全息教师配置",
            "student": "全息学生配置",
        }
        frame = self.config_frames.get(frame_key)
        if not frame:
            return
        self._hide_all_config_frames()
        frame.grid(row=0, column=0, sticky=tk.NSEW)

        display = titles.get(frame_key, frame_key)
        self.root.title(f"{display} - {self.base_title}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HolographicClassroomGUI(root)
    root.mainloop()