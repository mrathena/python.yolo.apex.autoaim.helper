import ctypes
import multiprocessing
import time
from multiprocessing import Process
from queue import Full, Empty

import cv2
import pynput
from win32gui import FindWindow, SetWindowPos, GetWindowText, GetForegroundWindow
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
import winsound

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
horizontal = 'horizontal'
confidence = 'confidence'

init = {
    weights: 'weights.apex.private.crony.1435244588.1127E7B7107206013DE38A10EDDEEEB3-v5-n-416-50000-3-0.1.2.engine',  # 权重文件, weights.apex.public.dummy.engine, weights.apex.public.engine, weights.apex.private.crony.1435244588.1127E7B7107206013DE38A10EDDEEEB3-v5-n-416-50000-3-0.1.2.engine
    classes: 0,  # 要检测的标签的序号(标签序号从0开始), 多个时如右 [0, 1]
    confidence: 0.5,  # 置信度, 低于该值的认为是干扰
    size: 320,  # 截图的尺寸, 屏幕中心 size*size 大小
    radius: 50,  # 瞄准生效半径, 目标瞄点出现在以准星为圆心该值为半径的圆的范围内时才会自动瞄准
    ads: 1,  # 移动倍数, 调整方式: 关闭仿真并开启自瞄后, 不断瞄准目标旁边并按住 F 键, 当准星移动稳定且精准快速不振荡时, 就找到了合适的 ADS 值
    center: None,  # 屏幕中心点
    region: None,  # 截图范围
    end: False,  # 退出标记, End
    box: False,  # 显示开关, Up
    show: False,  # 显示状态
    aim: True,  # 瞄准开关, Down, X2(侧上键)
    lock: False,  # 锁定状态(开火/预瞄)
    timestamp: None,  # 开火时间
    head: False,  # 是否瞄头, Right
    predict: True,  # 是否预瞄, Left
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
            aims, img = detector.detect(image=img, show=data[box])  # 目标检测, 得到截图坐标系内识别到的目标和标注好的图片(无需展示图片时img为none)
            aims = detector.convert(aims=aims, region=data[region])   # 将截图坐标系转换为屏幕坐标系
            if data[box]:
                cv2.putText(img, f'{Timer.cost(time.perf_counter_ns() - begin)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            try:
                product = (aims, img)
                queue.put(product, block=True, timeout=1)
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

    def move(x, y):
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
        # 瞄点调整
        targets = []
        for index, clazz, conf, sc, gc, sr, gr in aims:
            _, _, _, height = sr
            scx, scy = sc
            point = scx, scy - (height // 2 - height // (8 if data[head] else 3))  # 屏幕坐标系下各目标的瞄点坐标, 计算身体和头在方框中的大概位置来获得瞄点, 没有采用头标签的方式(感觉效果特别差)
            targets.append((point, gr))

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


    title = 'Realtime ScreenGrab Detect'

    # 主循环
    while True:

        if data[end]:
            cv2.destroyAllWindows()
            break
        if not (data[box] or data[aim]):
            continue

        # 数据获取
        product = None
        try:
            product = queue.get(block=True, timeout=1)
        except Empty:
            print(f'Consumer: Queue Empty')
        except:
            print('Consumer Error')

        # 数据处理, 得到 target 和 img
        target = None  # 目标, (sc, gr), sc:屏幕坐标系下目标的中心点, gr:截图坐标系下目标的矩形ltwh
        img = None  # 展示的截图
        if product:
            aims, img = product
            # 找到目标
            if len(aims) > 0:
                target = follow(aims)  # todo 尽量跟一个目标, 不要来回跳, 保证目标未检测到时能在原地停顿一会儿, 不直接跳到其他目标身上, 如目标长时间未被检测到, 才认为目标消失, 开始找下一个目标

        # 预测目标
        predicted = None
        if target:
            sc, gr = target
            predicted = predictor.predict(sc)
            if data[box] and img is not None:
                cx, cy = data[center]
                scx, scy = sc  # 目标所在点
                px, py = predicted  # 目标将在点
                dx = px - scx
                dy = py - scy
                if abs(dx) > 0 or abs(dy) > 0:
                    gl, gt, gw, gh = gr
                    px1 = gl + dx * 3
                    py1 = gt + dy * 3
                    px2 = px1 + gw
                    py2 = py1 + gh
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 256, 0), 2)
                    top = 60
                    cv2.putText(img, f'{scx - cx}, {scy - cy}', (10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                    cv2.putText(img, f'{px - cx}, {py - cy}', (10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                    cv2.putText(img, f'{scx - cx + px - cx}, {scy - cy + py - cy}', (10, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

        # 检测显示开关
        if data[box] and img is not None:
            data[show] = True
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(title, img)
            SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            cv2.waitKey(1)
        if not data[box] and data[show]:
            data[show] = False
            cv2.destroyAllWindows()

        # 检测瞄准开关
        if data[aim] and data[lock] and target:  # 瞄准开关是打开的, 且处于目标锁定状态
            sc, gr = target
            if inner(sc):
                # 计算要移动的像素
                cx, cy = data[center]  # 准星所在点(屏幕中心)
                scx, scy = sc  # 目标所在点
                # 考虑目标预测
                px, py = predicted  # 目标将在点
                if data[predict]:
                    x = scx - cx + px - cx
                    y = scy - cy + py - cy
                else:
                    x = scx - cx
                    y = scy - cy
                # 考虑倍数
                ax = int(x * data[ads])
                ay = int(y * data[ads])
                # print(f'目标位置:{sx},{sy}, 移动像素:{x},{y}, ADS:{ax},{ay}')
                # 移动
                move(ax, ay)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # windows 平台使用 multiprocessing 必须在 main 中第一行写这个
    manager = multiprocessing.Manager()
    queue = manager.Queue(maxsize=1)
    data = manager.dict()
    data.update(init)
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
