import argparse
import json
import threading
import time
from collections import deque

from server import THStreamServiceServicer, serve


class ReceiveStats:
    def __init__(self, max_points: int = 300):
        self.max_points = max_points
        self.latency_data = []
        self.bandwidth_data = []
        self.total_packets = 0
        self.total_bytes = 0
        self.bytes_buffer = deque(maxlen=10)
        self.last_update = time.time()

    def add_latency(self, latency_ms: float):
        self.latency_data.append(latency_ms)
        if len(self.latency_data) > self.max_points:
            self.latency_data.pop(0)

    def add_packet(self, data_size: int):
        self.total_packets += 1
        self.total_bytes += data_size

        current_time = time.time()
        self.bytes_buffer.append(data_size)

        # 参考 grpc2socket.py：每100ms更新一次带宽，窗口为最近10个100ms数据
        if current_time - self.last_update >= 0.1:
            current_bandwidth = sum(self.bytes_buffer)
            self.bandwidth_data.append(current_bandwidth)
            if len(self.bandwidth_data) > self.max_points:
                self.bandwidth_data.pop(0)
            self.last_update = current_time

    def summary(self) -> str:
        current_latency = self.latency_data[-1] if self.latency_data else 0.0
        avg_latency = sum(self.latency_data) / len(self.latency_data) if self.latency_data else 0.0
        current_bw = self.bandwidth_data[-1] if self.bandwidth_data else 0.0
        avg_bw = sum(self.bandwidth_data) / len(self.bandwidth_data) if self.bandwidth_data else 0.0

        return (
            f"延迟: 当前 {current_latency:.2f} ms | 平均 {avg_latency:.2f} ms | "
            f"带宽: 当前 {current_bw / 1024:.2f} KB/s | 平均 {avg_bw / 1024:.2f} KB/s | "
            f"总包数: {self.total_packets} | 总流量: {self.total_bytes / 1024:.2f} KB"
        )


def extract_timestamp_ms(ext_desc: str):
    """优先解析纯时间戳；兼容 JSON 元数据中的 timestamp_ms/t1_ms。"""
    if not ext_desc:
        return None

    desc = ext_desc.strip()
    if not desc:
        return None

    if desc.isdigit():
        return int(desc)

    try:
        meta = json.loads(desc)
        if isinstance(meta, dict):
            if "timestamp_ms" in meta:
                return int(meta["timestamp_ms"])
            if "t1_ms" in meta:
                return int(meta["t1_ms"])
    except Exception:
        return None

    return None


def main():
    parser = argparse.ArgumentParser(description="在线接收 gRPC 姿态流并统计延迟")
    parser.add_argument("--grpc_port", type=int, default=50051, help="接收端 gRPC 端口")
    parser.add_argument("--report_interval", type=float, default=1.0, help="统计打印周期(秒)")
    parser.add_argument("--poll_interval", type=float, default=0.01, help="缓冲区轮询周期(秒)")
    parser.add_argument("--debug", action="store_true", help="打印每包信息")
    args = parser.parse_args()

    if args.report_interval <= 0:
        raise ValueError("report_interval must be > 0")
    if args.poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    servicer = THStreamServiceServicer()
    server_thread = threading.Thread(target=serve, args=(servicer, args.grpc_port), daemon=True)
    server_thread.start()

    stats = ReceiveStats()
    last_report_time = time.time()

    print(f"接收端已启动，监听 gRPC 端口: {args.grpc_port}")

    try:
        while True:
            if servicer.receive_data_buffer.get_size() < 1:
                time.sleep(args.poll_interval)
            else:
                payload = servicer.receive_data_buffer.get_items()

                data_size = (
                    len(payload.rgbData)
                    + len(payload.pointData)
                    + len(payload.faceData)
                    + len(payload.limbData)
                    + len(payload.extData)
                )
                stats.add_packet(data_size)

                sent_ts = extract_timestamp_ms(payload.extDesc)
                if sent_ts is not None:
                    now_ms = int(time.time() * 1000)
                    stats.add_latency(float(now_ms - sent_ts))

                if args.debug:
                    print(
                        f"recv seq={payload.seqNo}, bytes={data_size}, "
                        f"extDesc={payload.extDesc[:80]}"
                    )

            now = time.time()
            if now - last_report_time >= args.report_interval:
                print(stats.summary())
                last_report_time = now

    except KeyboardInterrupt:
        print("\n接收被用户中断")
        print("最终统计:")
        print(stats.summary())


if __name__ == "__main__":
    main()

