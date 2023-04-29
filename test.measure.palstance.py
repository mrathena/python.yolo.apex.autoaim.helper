import ctypes
import multiprocessing
from multiprocessing import Process
import pynput
from win32api import GetCursorPos

shift = 'shift'
point = 'point'
total = 'total'
vertical = 'vertical'
horizontal = 'horizontal'


def keyboard(data):

    try:
        driver = ctypes.CDLL(r'mouse.device.lgs.dll')
        ok = driver.device_open() == 1
        if not ok:
            print('初始化失败, 未安装lgs/ghub驱动')
    except FileNotFoundError:
        print('初始化失败, 缺少文件')

    def move(x, y, absolute=False):
        if ok:
            if (x == 0) & (y == 0):
                return
            mx, my = x, y
            if absolute:
                ox, oy = GetCursorPos()
                mx = x - ox
                my = y - oy
            driver.moveR(mx, my, True)

    def press(key):
        if key == pynput.keyboard.Key.shift:
            data[shift] = True

    def release(key):
        if key == pynput.keyboard.Key.shift:
            data[shift] = False
        else:
            x, y = 0, 0
            if key == pynput.keyboard.Key.end:
                return False
            elif key == pynput.keyboard.Key.up:
                y = -100
            elif key == pynput.keyboard.Key.down:
                y = 100
            elif key == pynput.keyboard.Key.left:
                x = -100
            elif key == pynput.keyboard.Key.right:
                x = 100
            elif key == pynput.keyboard.Key.enter:
                data[vertical] = 0
                data[horizontal] = 0
            if x != 0:
                if data[shift]:
                    x = x // 10
                data[horizontal] += x
            if y != 0:
                if data[shift]:
                    y = y // 10
                data[vertical] += y
            move(x, y)
            print(f'水平:{abs(data[horizontal])}, 垂直:{abs(data[vertical])}')

    with pynput.keyboard.Listener(on_press=press, on_release=release) as k:
        k.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()
    data = manager.dict()
    data[shift] = False
    data[horizontal] = 0
    data[vertical] = 0
    pk = Process(target=keyboard, args=(data,))
    pk.start()
    pk.join()
