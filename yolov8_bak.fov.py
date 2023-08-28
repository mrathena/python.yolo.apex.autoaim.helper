import ctypes
import multiprocessing
import time
from multiprocessing import Process
import cv2
import pynput
from pynput.mouse import Button
from pynput.keyboard import Key, Listener
from win32gui import FindWindow, SetWindowPos, GetWindowText, GetForegroundWindow
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
import winsound
from simple_pid import PID
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageGrab
import time
from ultralytics import YOLO
import pyautogui

import math
from tool import detect,convert
import pydirectinput as direct

from key_input.press_key import InputKey
from key_input import Mouse, Keyboard
input_key = InputKey(0)


ads = 'ads'
pidc = 'pidc'
size = 'size'
stop = 'stop'
lock = 'lock'
show = 'show'
head = 'head'
left = 'left'
title = 'title'
debug = 'debug'
region = 'region'
center = 'center'
radius = 'radius'
weights = 'weights'
classes = 'classes'
confidence = 'confidence'
fov = 'fov'
horizontal = 'horizontal'
sensitive = 'sensitive'
sensitivity = 'sensitivity'
vertical = 'vertical'
init = {
    fov: 110,  # 游戏内的 FOV
    horizontal: 16420,  # 游戏内以鼠标灵敏度为1测得的水平旋转360°对应的鼠标移动距离, 多次测量验证. 经过测试该值与FOV无关. 移动像素理论上等于该值除以鼠标灵敏度Horizontal Vertical
    vertical: 7710 * 2,  # 垂直, 注意垂直只能测一半, 即180°范围, 所以结果需要翻倍
    sensitivity: 2,  # 当前游戏鼠标灵敏度
    title: '',  # 可在后台运行 print(GetWindowText(GetForegroundWindow())) 来检测前台游戏窗体标题
    weights: 'yolov8n.pt',
    classes: 0.0,  # 要检测的标签的序号(标签序号从0开始), 多个时如右 [0, 1]
    confidence: 0.3,  # 置信度, 低于该值的认为是干扰
    size: 400,  # 截图的尺寸, 屏幕中心 size*size 大小
    radius: 400,  # 瞄准生效半径, 目标瞄点出现在以准星为圆心该值为半径的圆的范围内时才会锁定目标
    ads: 1.2,  # 移动倍数, 调整方式: 瞄准目标旁边并按住 Shift 键, 当准星移动到目标点的过程, 稳定精准快速不振荡时, 就找到了合适的 ADS 值
    center: None,  # 屏幕中心点
    region: None,  # 截图范围
    stop: False,  # 退出, End
    lock: False,  # 锁定, Shift, 按左键时不锁(否则扔雷时也会锁)
    show: True,  # 显示, Down
    head: False,  # 瞄头, Up
    pidc: True,  # 是否启用 PID Controller, 还未完善, Left
    left: True,  # 左键锁, Right, 按鼠标左键时锁
    debug: False,  # Debug 模式, 用来调试 PID 值
}






def oc():
    ac, _ = data[center]
    return ac / math.tan((data[fov] / 2 * math.pi / 180))


def rx(x):
    angle = math.atan(x / oc()) * 180 / math.pi
    return int(angle * data[horizontal] / data[sensitivity] / 360)


def ry(y):
    angle = math.atan(y / oc()) * 180 / math.pi
    return int(angle * data[vertical] / data[sensitivity] / 360)

def game():
    return init[title] == GetWindowText(GetForegroundWindow())


def mouse(data):

    def down(x, y, button, pressed):
        if not game():
            return
        if button == Button.left and data[left]:
            data[lock] = pressed

    with pynput.mouse.Listener(on_click=down) as m:
        m.join()


def keyboard(data):

    def press(key):
        if not game():
            return
        if key == Key.shift:
            data[lock] = True

    def release(key):
        if key == Key.end:
            # 结束程序
            data[stop] = True
            winsound.Beep(400, 200)
            return False
        if not game():
            return
        if key == Key.shift:
            data[lock] = False
        elif key == Key.up:
            data[head] = not data[head]
            winsound.Beep(800 if data[head] else 400, 200)
        elif key == Key.down:
            data[show] = not data[show]
            winsound.Beep(800 if data[show] else 400, 200)
        elif key == Key.left:
            data[pidc] = not data[pidc]
            winsound.Beep(800 if data[pidc] else 400, 200)
        elif key == Key.right:
            data[left] = not data[left]
            winsound.Beep(800 if data[left] else 400, 200)
        elif key == Key.page_down:
            data[debug] = not data[debug]
            winsound.Beep(800 if data[debug] else 400, 200)

    with Listener(on_release=release, on_press=press) as k:
        k.join()


def loop(data):
    from toolkit import Capturer, Timer
    from tool import detect
    capturer = Capturer(data[title], data[region])
    winsound.Beep(800, 200)

    def move(x: int, y: int):  # 开镜模式下鼠标移动
        if (x == 0) & (y == 0):
            return
        center_x = 960
        center_y = 540  # 可能要-5
        # 计算屏幕中心到目标点的相对位移
        x_offset = x - center_x
        y_offset = y - center_y
        print(x_offset, y_offset)
        input_key.mouse_move(x_offset, y_offset)

    def inner(point):
        """
        判断该点是否在准星的瞄准范围内
        """
        a, b = data[center]
        x, y = point
        return (x - a) ** 2 + (y - b) ** 2 < data[radius] ** 2

    def follow(aims):
        """
        从 targets 里选目标瞄点距离准星最近的
        """
        if len(aims) == 0:
            return None

        # 瞄点调整
        targets = []
        for index, clazz, conf, sc, gc, sr, gr in aims:
            if conf < data[confidence]:  # 特意把置信度过滤放到这里(便于从图片中查看所有识别到的目标的置信度)
                continue
            _, _, _, height = sr
            sx, sy = sc
            gx, gy = gc
            differ = (height // 7) if data[head] else (height // 3)
            newSc = sx, sy - height // 2 + differ  # 屏幕坐标系下各目标的瞄点坐标, 计算身体和头在方框中的大概位置来获得瞄点, 没有采用头标签的方式(感觉效果特别差)
            newGc = gx, gy - height // 2 + differ
            targets.append((index, clazz, conf, newSc, newGc, sr, gr))
        if len(targets) == 0:
            return None

        # 找到目标
        cx, cy = data[center]
        index = 0
        minimum = 0
        for i, item in enumerate(targets):
            index, clazz, conf, sc, gc, sr, gr = item
            sx, sy = sc
            distance = (sx - cx) ** 2 + (sy - cy) ** 2
            if minimum == 0:
                index = i
                minimum = distance
            else:
                if distance < minimum:
                    index = i
                    minimum = distance
        return targets[index]

    text = 'Realtime Screen Capture Detect'
    pidx = PID(2, 0, 0.02, setpoint=0)
    times, targets, distances = [], [], []  # 用于绘图

    # 主循环
    while True:
        try:
            if data[stop]:
                break

            # 生产数据
            t1 = time.perf_counter_ns()
            image = Capturer.backup(data[region])
            t2 = time.perf_counter_ns()
            final_aims = detect(image)
            t3 = time.perf_counter_ns()
            aims = convert(final_aims=final_aims, region=data[region])

            # 找到目标
            # 找到目标
            target = follow(aims)
            screen_width, screen_height = pyautogui.size()

            # 计算屏幕中心坐标
            center_x = screen_width // 2
            center_y = screen_height // 2

            print(f"屏幕中心坐标：({center_x}, {center_y})")
            # 移动准星
            if data[lock] and target:
                index, clazz, conf, sc, gc, sr, gr = target
                if inner(sc):
                    screen_width, screen_height = pyautogui.size()
                    cx = screen_width // 2
                    cy = screen_height // 2
                    sx, sy = sc
                    x = sx - cx
                    y = sy - cy

                    if data[pidc]:
                        if data[debug]:  # 用于绘图
                            times.append(time.time())
                            targets.append(0)
                            distances.append(x)
                        px = -int(pidx(x))
                        print(px, y)
                        move(px, y)

                    else:
                        ax = int(x * data[ads])
                        ay = int(y * data[ads])
                        pyautogui.moveTo(ax, ay)
                        move(ax, ay)
            else:  # 用于绘图
                if data[debug] and len(times) != 0:
                    try:
                        plt.plot(times, targets, label='target')
                        plt.plot(times, distances, label='distance')
                        plt.legend()  # 图例
                        plt.xlabel('time')
                        plt.ylabel('distance')
                        times.clear()
                        targets.clear()
                        distances.clear()
                        matplotlib.use('TkAgg')  # TkAgg, module://backend_interagg
                        winsound.Beep(600, 200)
                        plt.show()
                    except:
                        pass

            # 显示检测
            if data[show] and image is not None:
                # 记录耗时
                cv2.putText(image, f'{Timer.cost(t3 - t1)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(image, f'{Timer.cost(t2 - t1)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cv2.putText(image, f'{Timer.cost(t3 - t2)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                # 瞄点划线
                if target:
                    index, clazz, conf, sc, gc, sr, gr = target
                    cv2.circle(image, gc, 2, (0, 0, 0), 2)
                    r = data[size] // 2
                    cv2.line(image, gc, (r, r), (255, 255, 0), 2)
                # 展示图片
                cv2.namedWindow(text, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(text, image)
                SetWindowPos(FindWindow(None, text), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                cv2.waitKey(1)
            if not data[show]:
                cv2.destroyAllWindows()

        except:
            pass
            # 如果没有找到目标，仍然显示检测窗口
        if data[show]:
            cv2.namedWindow(text, cv2.WINDOW_AUTOSIZE)
            blank_image = Capturer.backup(data[region])  # 创建一个空白图像
            cv2.imshow(text, blank_image)
            SetWindowPos(FindWindow(None, text), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            cv2.waitKey(1)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()
    data = manager.dict()
    data.update(init)
    # 初始化数据
    from toolkit import Monitor
    data[center] = Monitor.resolution.center()
    c1, c2 = data[center]
    data[region] = c1 - data[size] // 2, c2 - data[size] // 2, data[size], data[size]
    # 创建进程
    pm = Process(target=mouse, args=(data,), name='Mouse')
    pk = Process(target=keyboard, args=(data,), name='Keyboard')
    pl = Process(target=loop, args=(data,), name='Loop')
    # 启动进程
    pm.start()
    pk.start()
    pl.start()
    pk.join()  # 不写 join 的话, 使用 dict 的地方就会报错 conn = self._tls.connection, AttributeError: 'ForkAwareLocal' object has no attribute 'connection'
    pm.terminate()  # 鼠标进程无法主动监听到终止信号, 所以需强制结束
