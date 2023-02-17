import time
from matplotlib import pyplot as plt
from simple_pid import PID

# Python 实现 PID 控制基于 simple-pid 库
# https://blog.csdn.net/qq_51005828/article/details/109493386


class Heater:

    def __init__(self):
        self.temp = 25

    def update(self, power, dt):
        if power > 0:
            # 加热时房间温度随变量power和时间变量dt 的变化
            self.temp += 2 * power * dt
        # 表示房间的热量损失
        self.temp -= 0.5 * dt
        return self.temp


if __name__ == '__main__':

    # 将创建的模型写进主函数
    heater = Heater()
    temp = heater.temp
    # 设置PID的三个参数，以及限制输出
    pid = PID(2, 0.01, 0.1, setpoint=temp)  # 原版数据
    pid = PID(2, 0.01, 100, setpoint=temp)
    pid.output_limits = (0, None)
    # 用于设置时间参数
    start_time = time.time()
    last_time = start_time
    # 用于输出结果可视化
    setpoint, y, x = [], [], []
    # 设置系统运行时间
    while time.time() - start_time < 10:

        # 设置时间变量dt
        current_time = time.time()
        dt = (current_time - last_time)

        # 变量temp在整个系统中作为输出，变量temp与理想值之差作为反馈回路中的输入，通过反馈回路调节变量power的变化。
        power = pid(temp)
        temp = heater.update(power, dt)

        # 用于输出结果可视化
        x += [current_time - start_time]
        y += [temp]
        setpoint += [pid.setpoint]
        # 用于变量temp赋初值
        if current_time - start_time > 0:
            pid.setpoint = 100

        last_time = current_time

    # 输出结果可视化
    plt.plot(x, setpoint, label='target')
    plt.plot(x, y, label='PID')
    plt.xlabel('time')
    plt.ylabel('temperature')
    plt.legend()
    plt.show()




