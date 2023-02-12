import ctypes
import multiprocessing
import time
from multiprocessing import Process
import cv2
import pynput
from pynput.mouse import Button
from pynput.keyboard import Key, KeyCode, Listener
from win32gui import FindWindow, SetWindowPos, GetWindowText, GetForegroundWindow
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
import winsound

a = 'a'
d = 'd'
ad = 'ad'
ads = 'ads'
size = 'size'
stop = 'stop'
lock = 'lock'
show = 'show'
head = 'head'
left = 'left'
title = 'title'
region = 'region'
center = 'center'
radius = 'radius'
weights = 'weights'
classes = 'classes'
predict = 'predict'
confidence = 'confidence'

init = {
    title: 'Apex Legends',  # 可在后台运行 print(GetWindowText(GetForegroundWindow())) 来检测前台游戏窗体标题
    weights: 'weights.apex.private.crony.1435244588.1127E7B7107206013DE38A10EDDEEEB3-v5-n-416-50000-3-0.1.2.engine',
    classes: 0,  # 要检测的标签的序号(标签序号从0开始), 多个时如右 [0, 1]
    confidence: 0.5,  # 置信度, 低于该值的认为是干扰
    size: 400,  # 截图的尺寸, 屏幕中心 size*size 大小
    radius: 100,  # 瞄准生效半径, 目标瞄点出现在以准星为圆心该值为半径的圆的范围内时才会锁定目标
    ads: 1.2,  # 移动倍数, 调整方式: 瞄准目标旁边并按住 Shift 键, 当准星移动到目标点的过程, 稳定精准快速不振荡时, 就找到了合适的 ADS 值
    center: None,  # 屏幕中心点
    region: None,  # 截图范围
    stop: False,  # 退出, End
    lock: False,  # 锁定, Shift, 按左键时不锁(否则扔雷时也会锁)
    show: False,  # 显示, Down
    head: False,  # 瞄头, Up
    left: False,  # 左键锁, PgDn, 按左键时锁
    predict: False,  # 预瞄, Left
    ad: True,  # AD, Right
    a: False,  # A, 是否被按下
    d: False,  # D, 是否被按下
}


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
        elif key in (KeyCode.from_char('a'), KeyCode.from_char('A')):
            data[a] = True
        elif key in (KeyCode.from_char('d'), KeyCode.from_char('D')):
            data[d] = True

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
        elif key in (KeyCode.from_char('a'), KeyCode.from_char('A')):
            data[a] = False
        elif key in (KeyCode.from_char('d'), KeyCode.from_char('D')):
            data[d] = False
        elif key == Key.up:
            data[head] = not data[head]
            winsound.Beep(800 if data[head] else 400, 200)
        elif key == Key.down:
            data[show] = not data[show]
            winsound.Beep(800 if data[show] else 400, 200)
        elif key == Key.left:
            data[predict] = not data[predict]
            winsound.Beep(800 if data[predict] else 400, 200)
        elif key == Key.right:
            data[ad] = not data[ad]
            winsound.Beep(800 if data[ad] else 400, 200)
        elif key == Key.page_down:
            data[left] = not data[left]
            winsound.Beep(800 if data[left] else 400, 200)

    with Listener(on_release=release, on_press=press) as k:
        k.join()


def loop(data):

    from toolkit import Capturer, Detector, Predictor, Timer
    capturer = Capturer(data[title], data[region])
    detector = Detector(data[weights], data[classes])
    predictor = Predictor()
    winsound.Beep(800, 200)

    try:
        import os
        root = os.path.abspath(os.path.dirname(__file__))
        driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
        ok = driver.device_open() == 1
        if not ok:
            print('初始化失败, 未安装罗技驱动')
    except FileNotFoundError:
        print('初始化失败, 缺少文件')

    def move(x: int, y: int):
        if (x == 0) & (y == 0):
            return
        driver.moveR(x, y, True)

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
            scx, scy = sc
            point = scx, scy - (height // 2 - height // (8 if data[head] else 3))  # 屏幕坐标系下各目标的瞄点坐标, 计算身体和头在方框中的大概位置来获得瞄点, 没有采用头标签的方式(感觉效果特别差)
            targets.append((point, gr))
        if len(targets) == 0:
            return None

        # 找到目标
        cx, cy = data[center]
        index = 0
        minimum = 0
        for i, item in enumerate(targets):
            point, gr = item
            scx, scy = point
            distance = (scx - cx) ** 2 + (scy - cy) ** 2
            if minimum == 0:
                index = i
                minimum = distance
            else:
                if distance < minimum:
                    index = i
                    minimum = distance
        return targets[index]

    text = 'Realtime Screen Capture Detect'

    # 主循环
    while True:

        if data[stop]:
            break

        # 生产数据
        t1 = time.perf_counter_ns()
        img = capturer.grab()
        t2 = time.perf_counter_ns()
        aims, img = detector.detect(image=img, show=data[show])  # 目标检测, 得到截图坐标系内识别到的目标和标注好的图片(无需展示图片时img为none)
        t3 = time.perf_counter_ns()
        aims = detector.convert(aims=aims, region=data[region])  # 将截图坐标系转换为屏幕坐标系
        # print(f'{Timer.cost(t3 - t1)}, {Timer.cost(t2 - t1)}, {Timer.cost(t3 - t2)}')
        if data[show] and img is not None:
            cv2.putText(img, f'{Timer.cost(t3 - t1)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            cv2.putText(img, f'{Timer.cost(t2 - t1)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            cv2.putText(img, f'{Timer.cost(t3 - t2)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        # 数据处理
        target = follow(aims)

        # 预测目标
        predicted = None
        if target:
            sc, gr = target
            predicted = predictor.predict(sc)

        # 检测锁定开关
        if data[lock] and target:
            sc, gr = target
            if inner(sc):
                # 计算要移动的像素
                cx, cy = data[center]  # 准星所在点(屏幕中心)
                scx, scy = sc  # 目标所在点
                # 考虑目标预测
                px, py = predicted  # 目标将在点
                if data[predict]:
                    if abs(px - scx) > 100:
                        x = scx - cx
                        y = scy - cy
                    else:
                        x = px - cx
                        y = py - cy
                else:
                    # 考虑AD偏移
                    if data[ad]:
                        shift = gr[2] // 3
                        if data[a] and data[d]:
                            scx = scx
                        elif data[a] and not data[d]:
                            scx = scx + shift
                        elif not data[a] and data[d]:
                            scx = scx - shift
                    x = scx - cx
                    y = scy - cy
                # 考虑倍数
                ax = int(x * data[ads])
                ay = int(y * data[ads])
                # print(f'目标位置:{sx},{sy}, 移动像素:{x},{y}, ADS:{ax},{ay}')
                # 移动
                move(ax, ay)

        # 检测显示开关
        if data[show] and img is not None:
            cv2.namedWindow(text, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(text, img)
            SetWindowPos(FindWindow(None, text), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            cv2.waitKey(1)
        if not data[show]:
            cv2.destroyAllWindows()


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
