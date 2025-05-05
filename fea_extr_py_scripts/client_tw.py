import subprocess
import cv2
import grpc
import data_stream_pb2
import data_stream_pb2_grpc
import time
import numpy as np
import threading
import queue
import torch
import sys
import traceback

class THStreamClient:
    def __init__(self, host='127.0.0.1', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = data_stream_pb2_grpc.THStreamServiceStub(self.channel)
        self.seq_no = 0
        self.running = True
        self.decoded_frame_queue = queue.Queue(maxsize=1)  # For storing decoded frames
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.start()

        print("2. Models initialized")
        self.ffmpeg_process = None
        self.stream_receiver_thread = None


    def next_seq_no(self):
        self.seq_no += 1
        return str(self.seq_no)

    def tensor2np(self, img_tensor):
        """Optimized tensor to numpy conversion"""
        with torch.no_grad():
            img_np = img_tensor[0].permute(1, 2, 0).mul_(255).clamp_(0, 255)
            return img_np.byte().cpu().numpy()[..., ::-1]  # BGR conversion


    @staticmethod

    def cleanup(self):
        """Clean up resources on exit"""
        self.running = False

        # Stop FFmpeg if running
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()

        # Clean up CUDA context
        if hasattr(self, 'cfx') and self.cfx:
            try:
                self.cfx.pop()
                self.cfx = None
            except:
                pass

        torch.cuda.empty_cache()
        cv2.destroyAllWindows()

    def start_ffmpeg_receiver(self):
        """Start FFmpeg to receive and decode the RTP stream with proper logging and format conversion"""
        ffmpeg_cmd = [
            'ffmpeg',
            #'-use_wallclock_as_timestamps', '1',  # 使用系统时钟作为时间戳
            #'-strict', 'experimental',
            #'-debug_ts',
            #'-loglevel', 'debug',  # 比debug更详细
            #'-report',
            '-fflags', 'nobuffer',  # 减少缓冲
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            #'-probesize', '1M',  # 减少探测数据量
            #'-analyzeduration', '0',  # 立即开始处理

            '-c:v', 'hevc_cuvid',

            '-protocol_whitelist', 'file,udp,rtp,sdp',
            '-i', 'stream.sdp',

            # 输出设置
            '-f', 'rawvideo',  # 输出原始帧数据
            '-pix_fmt', 'bgr24',
            '-flush_packets', '1',

            '-threads', '1',  # 单线程减少调度
            '-'
        ]
        try:
            # Create log directory if it doesn't exist
           # os.makedirs('logs', exist_ok=True)
            #log_file = open('logs/ffmpeg_receiver.log', 'w')

            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                #stderr=log_file,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
           # print("FFmpeg receiver started (logs in logs/ffmpeg_receiver.log)")

            # Frame parameters
            width, height = 512, 512
            frame_size = width * height * 3  # 3 channels for BGR

            while self.running:
                # Read raw frame data
                #time1 = time.time()

                raw_frame = self.ffmpeg_process.stdout.read(frame_size)

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

                if self.decoded_frame_queue.full():
                    self.decoded_frame_queue.get()
                self.decoded_frame_queue.put(frame )
                #print(f"numpy item: {time.time() - time1:.4f} seconds")

        except Exception as e:
            print(f"Error in FFmpeg receiver: {e}")
            traceback.print_exc()
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            #if 'log_file' in locals():
            #    log_file.close()
            print("FFmpeg receiver stopped")

    def display_frames(self):
        """从解码帧队列中取出帧并显示，并计算 FPS"""
        frame_count = 0
        start_time = time.time()
        fps = 0
        #cv2.namedWindow('Received from server', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('Received from server', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:
            if not self.decoded_frame_queue.empty():

                frame = self.decoded_frame_queue.get()

                #后处理....

                #time2 = time.time()
                frame_count += 1
                # 计算 FPS
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1.0:  # 每秒钟计算一次 FPS
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0  # 重置计数器
                    start_time = current_time  # 重置起始时间

                # 缩放帧为原来的一半
                resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                # 在左上角显示帧率
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(resized_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示帧
                cv2.imshow('Received from server', resized_frame)
                cv2.waitKey(1)  # 大约30帧每秒
                #print(f"9.diplay time: {time.time() - time2:.4f} seconds")
            else:
                time.sleep(0.01)  # 避免空转

    def maintain_connection(self):
        """Maintain gRPC connection without sending any data"""
        try:
            # Just keep the connection open without yielding anything
            while self.running:
                time.sleep(1)  # Small delay to prevent busy waiting
            # Yield an empty generator
            yield from ()

        except Exception as e:
            print(f"Connection error: {e}")

    def run(self):
        """Run the client"""
        try:
            # Start FFmpeg receiver thread first
            self.stream_receiver_thread = threading.Thread(target=self.start_ffmpeg_receiver)
            self.stream_receiver_thread.start()

            # Then establish gRPC connection (which tells server to start streaming)
            response_iterator = self.stub.BidirectionalStream(self.maintain_connection())

            # Just consume responses (though we're not really using them)
            for response in response_iterator:
                print(f"Server response: {response.retMsg}")
                if not self.running:
                    break

        except KeyboardInterrupt:
            print("Client stopping...")
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            self.cleanup()

if __name__ == '__main__':
    client = THStreamClient()
    client.run()