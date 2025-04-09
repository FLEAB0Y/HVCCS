import ntplib
import time
import subprocess

# 初始化时间，获取本地时间戳与网络时间戳的差值 current_web_time = int(time.time() * 1000) + diff
def init_time():
    while True:
        try:
            # 测量网络延迟
            ping_result = subprocess.run(
                ["ping", "-n", "1", "pool.ntp.org"],  # Windows 使用 -n 参数
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if ping_result.returncode != 0:
                raise Exception("Ping 失败")
            
            # 提取 ping 时间 (以毫秒为单位)
            ping_output = ping_result.stdout
            print(f"Ping 输出:\n{ping_output}")  # 调试用，打印 ping 输出
            latency_line = [line for line in ping_output.split("\n") if "时间=" in line or "time=" in line]
            if not latency_line:
                raise Exception("无法解析 ping 输出中的延迟信息")
            
            # 提取延迟值
            latency_str = latency_line[0].split("time=")[-1].split("ms")[0].strip()
            latency = float(latency_str)
            
            # 获取网络时间
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request('pool.ntp.org', version=3)
            web_timestamp = int(response.tx_time * 1000)
            local_timestamp = int(time.time() * 1000)
            break
        except Exception as e:
            print(f"从 pool.ntp.org 请求时间戳失败: {e}")
            time.sleep(1)  # 等待一秒后重试
    
    # 将网络延迟的一半加入时间差
    diff = web_timestamp - local_timestamp + int(latency / 2)
    return diff
    
if __name__ == '__main__':
    total_diff = 0
    for i in range(10):
        diff = init_time()
        print(f"时间差: {diff} ms")
        total_diff =+ diff
    # 计算平均时间差
    avg_diff = total_diff / 10
    
    local_time = int(time.time() * 1000)

    # 计算本地时间与网络时间的差值
    print(f"本地时间戳: {local_time}")
    print(f"网络时间戳: {local_time + avg_diff}")
    print(f"时间差: {avg_diff}")