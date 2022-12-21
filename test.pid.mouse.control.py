import ctypes
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from simple_pid import PID
from win32gui import GetCursorPos


def cost(interval):
    """
    转换耗时, 输入纳秒间距, 转换为合适的单位
    """
    if interval < 1000:
        return f'{interval}ns'
    elif interval < 1_000_000:
        return f'{round(interval / 1000, 3)}us'
    elif interval < 1_000_000_000:
        return f'{round(interval / 1_000_000, 3)}ms'
    else:
        return f'{round(interval / 1_000_000_000, 3)}s'


try:
    driver = ctypes.CDLL('logitech.driver.dll')
    ok = driver.device_open() == 1
    if not ok:
        print('初始化失败, 未安装罗技驱动')
except FileNotFoundError:
    print('初始化失败, 缺少文件')


def move(x, y):
    if (x == 0) & (y == 0):
        return
    driver.moveR(x, y, True)


# 用于输出结果可视化
setpoint, y, x = [], [], []

# 获取当前鼠标水平坐标
v = GetCursorPos()[0]
print(v)
# 水平方向 pid 移动, 从当前点v移动到 x=3000 的位置
pid = PID(0.25, 0, 0, setpoint=v)
begin = time.perf_counter_ns()

while time.perf_counter_ns() - begin < 1000 * 1_000_000:

    now = time.perf_counter_ns()

    control = int(pid(v))
    move(control, 0)
    # while time.perf_counter_ns() - now < 100_000:
    #     pass
    v = GetCursorPos()[0]

    # 用于输出结果可视化
    x += [now - begin]
    y += [v]
    setpoint += [pid.setpoint]

    pid.setpoint = 3000

    if abs(v - 3000) < 2:
        print(cost(now - begin))
        break

    time.sleep(0.001)


print(f'结束: {v}, {cost(time.perf_counter_ns() - begin)}')
# 输出结果可视化
plt.plot(x, setpoint, label='target')
plt.plot(x, y, label='PID')
plt.xlabel('time')
plt.ylabel('point.x')
plt.legend()
plt.show()
