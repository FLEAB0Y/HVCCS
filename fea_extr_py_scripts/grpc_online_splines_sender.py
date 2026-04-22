import argparse
import os
import threading
import time

import numpy as np

from client import THStreamClient
from THStreamData import THStreamDataPayload


def to_tjc_from_npy(pose: np.ndarray, coord_dims: int = 3) -> np.ndarray:
    """Normalize common npy layouts into shape (T, J, C)."""
    arr = np.asarray(pose)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[-1] >= coord_dims:
            return arr[..., :coord_dims]
        if arr.shape[1] >= coord_dims:
            return np.transpose(arr, (0, 2, 1))[..., :coord_dims]
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 2:
        if arr.shape[1] % coord_dims == 0:
            joints = arr.shape[1] // coord_dims
            return arr.reshape(arr.shape[0], joints, coord_dims)
        if arr.shape[0] % coord_dims == 0:
            joints = arr.shape[0] // coord_dims
            return arr.T.reshape(arr.shape[1], joints, coord_dims)
        raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")

    if arr.ndim == 1:
        if arr.shape[0] % coord_dims != 0:
            raise ValueError(f"cannot infer npy pose layout from shape {arr.shape}")
        joints = arr.shape[0] // coord_dims
        return arr.reshape(1, joints, coord_dims)

    raise ValueError(f"unsupported npy pose shape: {arr.shape}")


def parse_pose_from_line(
    line: str,
    num_keypoints: int = 17,
    coord_dims: int = 3,
    start_col: int = 0,
):
    """Parse one text line into one pose frame with shape (17, 3)."""
    tokens = [token.strip() for token in line.replace(",", " ").split()]
    expected_values = num_keypoints * coord_dims
    end_col = start_col + expected_values
    if len(tokens) < end_col:
        return None

    try:
        flat_values = np.array(
            list(map(float, tokens[start_col:end_col])), dtype=np.float32
        )
    except ValueError:
        return None

    if flat_values.shape[0] != expected_values:
        return None

    return flat_values.reshape(num_keypoints, coord_dims)


def iter_h36m_frames(
    feature_file: str,
    num_keypoints: int = 17,
    coord_dims: int = 3,
    start_col: int = 0,
):
    """Iterate pose frames from .npy or text file."""
    suffix = os.path.splitext(feature_file)[1].lower()
    if suffix == ".npy":
        pose_array = np.load(feature_file)
        pose_tjc = to_tjc_from_npy(pose_array, coord_dims=coord_dims)
        if pose_tjc.shape[1] != num_keypoints:
            raise ValueError(
                f"npy joint count mismatch: file has {pose_tjc.shape[1]} joints, "
                f"expected {num_keypoints}"
            )
        for index in range(pose_tjc.shape[0]):
            yield pose_tjc[index].astype(np.float32, copy=False)
        return

    with open(feature_file, "r", encoding="utf-8") as feature_fp:
        for line in feature_fp:
            pose_frame = parse_pose_from_line(
                line,
                num_keypoints=num_keypoints,
                coord_dims=coord_dims,
                start_col=start_col,
            )
            if pose_frame is not None:
                yield pose_frame


class GRPCPoseSender:
    def __init__(self, host: str, port: int):
        self.client = THStreamClient(host=host, port=port)
        self._stop_event = threading.Event()
        self._thread = None

    def _send_loop(self):
        while not self._stop_event.is_set():
            if self.client.send_data_buffer.get_size() > 0:
                self.client.send_data()
            else:
                time.sleep(0.001)

    def start(self):
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def send_frame(self, pose_frame: np.ndarray, timestamp_ms: int):
        flat_pose = pose_frame.reshape(-1)
        limb_data_bytes = ",".join(map(str, flat_pose.tolist())).encode("utf-8")

        payload = THStreamDataPayload(
            rgb_data=b"\x00",
            point_data=b"\x00",
            face_data=b"\x00",
            limb_data=limb_data_bytes,
            ext_data=b"\x00",
            ext_desc=str(timestamp_ms),
        )

        while self.client.send_data_buffer.get_size() >= 80:
            time.sleep(0.002)
        self.client.send_data_buffer.add_item(payload)

    def shutdown(self, drain_timeout: float = 2.0):
        deadline = time.time() + max(drain_timeout, 0.0)
        while self.client.send_data_buffer.get_size() > 0 and time.time() < deadline:
            time.sleep(0.005)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        try:
            self.client.channel.close()
        except Exception:
            pass


def stream_pose_frames(
    feature_file: str,
    sender: GRPCPoseSender,
    fps: float,
    max_frames: int,
    start_col: int,
    debug: bool,
):
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    interval = 1.0 / fps
    sent_count = 0

    for frame_idx, pose_frame in enumerate(
        iter_h36m_frames(
            feature_file,
            num_keypoints=17,
            coord_dims=3,
            start_col=start_col,
        )
    ):
        tick = time.time()
        timestamp_ms = int(time.time() * 1000)

        sender.send_frame(pose_frame, timestamp_ms)
        sent_count += 1

        if debug:
            print(
                f"sent frame={frame_idx}, dims={pose_frame.shape[0]}x{pose_frame.shape[1]}, "
                f"timestamp_ms={timestamp_ms}"
            )

        if max_frames > 0 and sent_count >= max_frames:
            break

        elapsed = time.time() - tick
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    return sent_count


def main():
    parser = argparse.ArgumentParser(
        description="Send Human3.6M 17x3 pose frames via gRPC at target fps"
    )
    parser.add_argument("--feature_file", type=str, default="/Users/twz/demo_sys_user/h36m_pose_cam_1_downsample/test/S2_cam_1_30fps/Running_37_cam_1_h36m_30fps.npy", help="input .npy or text file")
    parser.add_argument("--server_addr", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--port_num", type=int, default=50051, help="server port")
    parser.add_argument("--fps", type=float, default=30.0, help="send fps")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="max frames to send, <=0 means all frames",
    )
    parser.add_argument(
        "--start_col",
        type=int,
        default=0,
        help="start column for text feature parsing",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logs")
    args = parser.parse_args()

    if not os.path.exists(args.feature_file):
        raise FileNotFoundError(f"feature_file not found: {args.feature_file}")

    sender = GRPCPoseSender(host=args.server_addr, port=args.port_num)
    sender.start()

    try:
        sent = stream_pose_frames(
            feature_file=args.feature_file,
            sender=sender,
            fps=args.fps,
            max_frames=args.max_frames,
            start_col=args.start_col,
            debug=args.debug,
        )
        print(f"sender_done: sent_frames={sent}, fps={args.fps}, file={args.feature_file}")
    finally:
        sender.shutdown(drain_timeout=2.0)


if __name__ == "__main__":
    main()
