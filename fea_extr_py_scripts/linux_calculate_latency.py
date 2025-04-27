import ntplib
import time
import subprocess

# 初始化时间，获取本地时间戳与网络时间戳的差值 current_web_time = int(time.time() * 1000) + diff
def init_time():
    while True:
        try:
            # 测量网络延迟
            ping_result = subprocess.run(
                ["ping", "-c", "1", "166.111.8.28"],
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
            print(f"网络延迟: {latency} ms")
            
            # 获取网络时间
            ntp_client = ntplib.NTPClient()
            respinse = ntp_client.request('166.111.8.28', version=3)
            web_timestamp = int(respinse.tx_time * 1000)
            local_timestamp = int(time.time() * 1000)
            break
        except Exception as e:
            print(f"从 166.111.8.28 请求时间戳失败: {e}")
    
    # 将网络延迟的一半加入时间差
    diff_consider_latency = web_timestamp - local_timestamp + int(latency / 2)
    diff_ignore_latency = web_timestamp - local_timestamp
    return diff_ignore_latency, diff_consider_latency
    
if __name__ == '__main__':
    total_diff = 0
    for i in range(10):
        diff_ignore_latency, diff_consider_latency = init_time()
        print(f"第 {i + 1} 次请求网络时间戳")
        print(f"网络时间戳: {diff_ignore_latency}")
        print(f"网络时间戳(考虑延迟): {diff_consider_latency}")
        # total_diff += diff  # 修正了这里的赋值运算符
    # 计算平均时间差
    avg_diff = total_diff / 10
    
    local_time = int(time.time() * 1000)

    # 计算本地时间与网络时间的差值
    print(f"本地时间戳: {local_time}")
    print(f"网络时间戳: {local_time + avg_diff}")
    print(f"平均时间差: {avg_diff}")