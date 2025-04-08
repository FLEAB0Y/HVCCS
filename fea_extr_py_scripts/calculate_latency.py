import ntplib
import time
import subprocess

# 初始化时间，获取本地时间戳与网络时间戳的差值 current_web_time = int(time.time() * 1000) + diff
def init_time():
    while True:
        try:
            # 测量网络延迟
            ping_result = subprocess.run(
                ["ping", "-c", "1", "pool.ntp.org"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if ping_result.returncode != 0:
                raise Exception("Ping 失败")
            
            # 提取 ping 时间 (以毫秒为单位)
            ping_output = ping_result.stdout
            latency_line = [line for line in ping_output.split("\n") if "time=" in line]
            if not latency_line:
                raise Exception("无法解析 ping 输出")
            latency = float(latency_line[0].split("time=")[1].split(" ")[0])
            
            # 获取网络时间
            ntp_client = ntplib.NTPClient()
            respinse = ntp_client.request('pool.ntp.org', version=3)
            web_timestamp = int(respinse.tx_time * 1000)
            local_timestamp = int(time.time() * 1000)
            break
        except Exception as e:
            print(f"从 pool.ntp.org 请求时间戳失败: {e}")
    
    # 将网络延迟的一半加入时间差
    diff = web_timestamp - local_timestamp + int(latency / 2)
    return diff
    
if __name__ == '__main__':
    local_time = int(time.time() * 1000)
    diff = init_time()
    # 计算本地时间与网络时间的差值
    print(f"本地时间戳: {local_time}")
    print(f"网络时间戳: {local_time + diff}")
    print(f"时间差: {diff}")