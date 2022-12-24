import ctypes
import multiprocessing
import random
import time
from multiprocessing import Process
from queue import Full, Empty

import cv2
import pynput
from win32gui import GetCursorPos, FindWindow, SetWindowPos, GetWindowText, GetForegroundWindow
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
import winsound

a = 'a'
d = 'd'
ad = 'ad'
ads = 'ads'
end = 'end'
box = 'box'
aim = 'aim'
show = 'show'
head = 'head'
lock = 'lock'
size = 'size'
region = 'region'
center = 'center'
radius = 'radius'
weights = 'weights'
classes = 'classes'
predict = 'predict'
vertical = 'vertical'
timestamp = 'timestamp'
emulation = 'emulation'
horizontal = 'horizontal'
confidence = 'confidence'
randomness = 'randomness'

init = {
    weights: 'weights.apex.private.crony.1435244588.1127E7B7107206013DE38A10EDDEEEB3-v5-n-416-50000-3-0.1.2.engine',  # 权重文件, weights.apex.public.dummy.engine, weights.apex.public.engine, weights.apex.private.crony.1435244588.1127E7B7107206013DE38A10EDDEEEB3-v5-n-416-50000-3-0.1.2.engine
    classes: 0,  # 要检测的标签的序号(标签序号从0开始), 多个时如右 [0, 1]
    confidence: 0.5,  # 置信度, 低于该值的认为是干扰
    size: 400,  # 截图的尺寸, 屏幕中心 size*size 大小
    radius: 50,  # 瞄准生效半径, 目标瞄点出现在以准星为圆心该值为半径的圆的范围内时才会自动瞄准
    ads: 1,  # 移动倍数, 调整方式: 关闭仿真并开启自瞄后, 不断瞄准目标旁边并按住 F 键, 当准星移动稳定且精准快速不振荡时, 就找到了合适的 ADS 值
    horizontal: 0.5,  # 水平方向的额外瞄准力度倍数, 该类值小一点有利于防止被别人识破 AI
    vertical: 0.5,  # 垂直方向的额外瞄准力度倍数, 该类值小一点有利于防止被别人识破 AI
    center: None,  # 屏幕中心点
    region: None,  # 截图范围
    end: False,  # 退出标记, End
    box: False,  # 显示开关, Up
    show: False,  # 显示状态
    aim: True,  # 瞄准开关, Down, X2(侧上键)
    lock: False,  # 锁定状态(开火/预瞄)
    timestamp: None,  # 开火时间
    head: False,  # 是否瞄头, Right
    predict: False,  # 是否预瞄, Left
    emulation: False,  # 是否仿真(减小力度加随机值), PageDown
    randomness: False,  # 仿真时是否随机左右偏移, PageUp
    ad: True,  # AD 模式开关, F11
    a: False,  # A 键状态, 是否被按下
    d: False,  # D 键状态, 是否被按下
}


def game():
    return 'Apex Legends' in GetWindowText(GetForegroundWindow())
    # return True


def mouse(data):

    def down(x, y, button, pressed):
        if not game():
            return
        if button == pynput.mouse.Button.left:
            data[lock] = pressed
            if pressed:
                data[timestamp] = time.time_ns()
        elif button == pynput.mouse.Button.x2:
            if pressed:
                data[aim] = not data[aim]
                winsound.Beep(800 if data[aim] else 400, 200)

    with pynput.mouse.Listener(on_click=down) as m:
        m.join()


def keyboard(data):

    def press(key):
        if not game():
            return
        if key == pynput.keyboard.KeyCode.from_char('f'):
            data[lock] = True
        elif key == pynput.keyboard.KeyCode.from_char('a'):
            data[a] = True
        elif key == pynput.keyboard.KeyCode.from_char('d'):
            data[d] = True

    def release(key):
        if key == pynput.keyboard.Key.end:
            # 结束程序
            data[end] = True
            winsound.Beep(400, 200)
            return False
        if not game():
            return
        if key == pynput.keyboard.KeyCode.from_char('f'):
            data[lock] = False
        elif key == pynput.keyboard.KeyCode.from_char('a'):
            data[a] = False
        elif key == pynput.keyboard.KeyCode.from_char('d'):
            data[d] = False
        elif key == pynput.keyboard.Key.f11:
            data[ad] = not data[ad]
            winsound.Beep(800 if data[ad] else 400, 200)
        elif key == pynput.keyboard.Key.up:
            data[box] = not data[box]
            winsound.Beep(800 if data[box] else 400, 200)
        elif key == pynput.keyboard.Key.down:
            data[aim] = not data[aim]
            winsound.Beep(800 if data[aim] else 400, 200)
        elif key == pynput.keyboard.Key.left:
            data[predict] = not data[predict]
            winsound.Beep(800 if data[predict] else 400, 200)
        elif key == pynput.keyboard.Key.right:
            data[head] = not data[head]
            winsound.Beep(800 if data[head] else 400, 200)
        elif key == pynput.keyboard.Key.page_down:
            data[emulation] = not data[emulation]
            winsound.Beep(800 if data[emulation] else 400, 200)
        elif key == pynput.keyboard.Key.page_up:
            data[randomness] = not data[randomness]
            winsound.Beep(800 if data[randomness] else 400, 200)

    with pynput.keyboard.Listener(on_release=release, on_press=press) as k:
        k.join()


def producer(data, queue):

    from toolkit import Capturer, Detector, Timer
    detector = Detector(data[weights], data[classes], data[confidence])
    winsound.Beep(800, 200)

    while True:

        if data[end]:
            break
        if data[box] or data[aim]:
            begin = time.perf_counter_ns()
            img = Capturer.grab(win=True, region=data[region], convert=True)
            aims, img = detector.detect(image=img, show=data[box])
            aims = detector.convert(aims=aims, region=data[region])
            if data[box]:
                cv2.putText(img, f'{Timer.cost(time.perf_counter_ns() - begin)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            try:
                queue.put((aims, img), block=True, timeout=1)
            except Full:
                print(f'Producer: Queue Full')
            except:
                print('Producer Error')


def consumer(data, queue):

    from toolkit import Monitor, Predictor
    data[center] = Monitor.resolution.center()
    c1, c2 = data[center]
    data[region] = c1 - data[size] // 2, c2 - data[size] // 2, data[size], data[size]
    predictor = Predictor()

    try:
        import os
        root = os.path.abspath(os.path.dirname(__file__))
        driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
        ok = driver.device_open() == 1
        if not ok:
            print('初始化失败, 未安装罗技驱动')
    except FileNotFoundError:
        print('初始化失败, 缺少文件')

    def move(x, y, absolute=False):
        if (x == 0) & (y == 0):
            return
        mx, my = x, y
        if absolute:
            ox, oy = GetCursorPos()
            mx = x - ox
            my = y - oy
        driver.moveR(mx, my, True)

    def lowestCommonMultiple(x: int, y: int):  # 最小公倍数
        m, n = x, y
        k = m * n  # k存储两数的乘积
        if m < n:  # 比较两个数的大小，使得m中存储大数，n中存储小数
            temp = m
            m = n
            n = temp
        b = m % n  # b存储m除以n的余数
        while b != 0:
            m = n  # 原来的小数作为下次运算时的大数
            n = b  # 将上一次的余数作为下次相除时的小数
            b = m % n
        result = k // n  # 两数乘积除以最大公约数即为它们的最小公倍数
        return result

    def m(millis: int, horizontal: bool, pixel: int):
        if pixel == 0:
            return
        begin = time.perf_counter_ns()
        times = 0  # 移动次数, 循环判断条件
        absx = abs(pixel)
        direction = pixel // absx  # 方向(值只能是正负1)
        nanos = millis * 1_000_000  # 毫秒转纳秒
        while True:  # do-while
            # setup code
            times += 1  # 循环次数的取值范围是[1,100], 101时会break
            # break condition
            if times > absx:
                break
            # loop body
            # 移动
            if horizontal:
                move(direction, 0)
            else:
                move(0, direction)  # 移动方式为: 移动-间隔-移动-间隔-移动-间隔-...-间隔-移动-间隔-移动-间隔-移动, 所以间隔比移动次数少一次
            # 间隔
            if times < absx:
                flag = time.perf_counter_ns()
                cost = (nanos - (flag - begin)) // (absx - times)  # 每移动一个像素的耗时
                while time.perf_counter_ns() - flag < cost:
                    pass

    def mxy(millis: int, x: int, y: int):  # 在指定毫秒内在水平和垂直方向上移动指定像素
        if x == 0 and y == 0:
            return
        elif x == 0 and y != 0:
            m(millis=millis, horizontal=False, pixel=y)
        elif x != 0 and y == 0:
            m(millis=millis, horizontal=True, pixel=x)
        else:
            absx = abs(x)  # 距离的绝对值
            absy = abs(y)
            nanos = millis * 1_000_000  # 毫秒转纳秒
            dirx = x // absx  # direction 方向 (值只能是正负1)
            diry = y // absy
            multiple = lowestCommonMultiple(absx, absy)  # 最小公倍数, 需要将时间分割的段数
            divisorx = multiple // absx
            divisory = multiple // absy
            step = nanos // multiple  # 时间跨度步长, 时间每过一个步长, 都需要判断下该时间点是否需要对两个方向做移动
            for i in range(multiple):
                if i % divisorx == 0:
                    move(dirx, 0)
                if i % divisory == 0:
                    move(0, diry)
                flag = time.perf_counter_ns()
                while time.perf_counter_ns() - flag < step:
                    pass
                # 获取坐标
                # print(GetCursorPos())

    def inner(point):
        """
        判断该点是否在准星的瞄准范围内
        """
        a, b = data[center]
        x, y = point
        return (x - a) ** 2 + (y - b) ** 2 < data[radius] ** 2

    def follow(targets, last):
        """
        从 targets 里选距离 last 最近的
        """
        if len(targets) == 0:
            return None
        if last is None:
            lx, ly = data[center]
        else:
            lsc, _ = last
            lx, ly = lsc
        index = 0
        minimum = 0
        for i, item in enumerate(targets):
            sc, _ = item
            sx, sy = sc
            distance = (sx - lx) ** 2 + (sy - ly) ** 2
            if minimum == 0:
                index = i
                minimum = distance
            else:
                if distance < minimum:
                    index = i
                    minimum = distance
        return targets[index]

    title = 'Realtime ScreenGrab Detect'

    last = None
    while True:

        if data[end]:
            cv2.destroyAllWindows()
            break
        if not (data[box] or data[aim]):
            continue
        product = None
        try:
            product = queue.get(block=True, timeout=1)
        except Empty:
            print(f'Consumer: Queue Empty')
        except:
            print('Consumer Error')
        if not product:
            continue
        aims, img = product
        targets = []
        for index, clazz, conf, sc, gc, sr, gr in aims:
            _, _, _, height = sr
            cx, cy = sc
            targets.append(((cx, cy - (height // 2 - height // (8 if data[head] else 3))), gr))  # 计算身体和头在方框中的大概位置来获得瞄点, 没有采用头标签的方式(感觉效果特别差)
        target = None  # 格式: (sc, gr), sc:屏幕坐标系下的目标所在点(瞄点坐标), gr:截图坐标系下的边框ltwh
        predicted = None
        if len(targets) != 0:
            # 拿到瞄准目标
            # 尽量跟一个目标, 不要来回跳, 目标消失后在原地停顿几个循环, 如目标仍未再次出现, 才认为目标消失, 开始找下一个目标
            target = follow(targets, last)
            # 重置上次瞄准的目标
            last = target
            # 解析目标里的信息
            if target:
                sc, gr = target
                predicted = predictor.predict(sc)  # 目标预测点
                # 计算移动距离, 展示预瞄位置
                if data[box]:
                    sx, sy = sc  # 目标所在点
                    px, py = predicted  # 目标将在点
                    dx = (px - sx) * 2
                    dy = (py - sy) * 2
                    gl, gt, gw, gh = gr
                    px1 = gl + dx
                    py1 = gt + dy
                    px2 = px1 + gw
                    py2 = py1 + gh
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 256, 0), 2)
        # 检测瞄准开关
        if data[aim] and data[lock]:
            if target:
                sc, gr = target
                if inner(sc):
                    # 计算要移动的像素
                    cx, cy = data[center]  # 准星所在点(屏幕中心)
                    sx, sy = sc  # 目标所在点
                    # 考虑目标预测
                    px, py = predicted  # 目标将在点
                    if data[predict] and abs(px - sx) < 50:
                        x = int(px - cx)
                        y = int(py - cy)
                    else:
                        # 考虑AD偏移
                        if data[ad]:
                            shift = gr[2] // 3
                            if data[a] and data[d]:
                                sx = sx
                            elif data[a] and not data[d]:
                                sx = sx + shift
                            elif not data[a] and data[d]:
                                sx = sx - shift
                        x = sx - cx
                        y = sy - cy
                    # 考虑倍数和仿真
                    ax = int(x * data[ads] * (data[horizontal] if data[emulation] else 1))
                    if data[emulation] and data[randomness]:
                        temp = -1 if x >= 0 else 1
                        ax = (random.randint(0, 8) * temp) if random.random() <= 0.2 else ax
                    ay = int(y * data[ads] * (data[vertical] if data[emulation] else 1))
                    # pid
                    # px = int(pidx(ax))
                    # py = int(pidy(ay))
                    px = int(ax)
                    py = int(ay)
                    # print(f'目标位置:{sx},{sy}, 移动像素:{x},{y}, ADS:{ax},{ay}')
                    # 移动
                    if data[emulation]:
                        mxy(10, px, py)
                    else:
                        move(px, py)
                        # mxy(10, px, py)
        # 检测显示开关
        if data[box]:
            if img is None:
                continue
            data[show] = True
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(title, img)
            SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            cv2.waitKey(1)
        if not data[box] and data[show]:
            data[show] = False
            cv2.destroyAllWindows()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # windows 平台使用 multiprocessing 必须在 main 中第一行写这个
    manager = multiprocessing.Manager()
    queue = manager.Queue(maxsize=1)
    data = manager.dict()  # 创建进程安全的共享变量
    data.update(init)  # 将初始数据导入到共享变量
    # 将键鼠监听和压枪放到单独进程中跑
    pm = Process(target=mouse, args=(data,), name='Mouse')
    pk = Process(target=keyboard, args=(data,), name='Keyboard')
    pp = Process(target=producer, args=(data, queue,), name='Producer')
    pc = Process(target=consumer, args=(data, queue,), name='Consumer')
    pm.start()
    pk.start()
    pp.start()
    pc.start()
    pk.join()  # 不写 join 的话, 使用 dict 的地方就会报错 conn = self._tls.connection, AttributeError: 'ForkAwareLocal' object has no attribute 'connection'
    pm.terminate()  # 鼠标进程无法主动监听到终止信号, 所以需强制结束
