import math

def skelton_sim_data_gen(t, rv):
    """
    计算正弦函数值，值域限制在[-0.1,0.1]之间
    
    参数:
        t: 时间
        rv: 角速度
    
    返回:
        值域在[-0.1,0.1]之间的正弦值
    """
    amplitude = 0.1  # 振幅设为0.1，使值域在[-0.1,0.1]
    return amplitude * math.sin(rv * t)

# 示例用法
if __name__ == "__main__":
    time = 1.0
    angular_velocity = 0.2 * math.pi  # 例如2π弧度/秒
    result = sine_function(time, angular_velocity)
    print(f"t={time}, rv={angular_velocity}, sine value={result}")