import sys
import time

import cv2
import numpy as np
from win32con import SRCCOPY, HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
from win32gui import GetDesktopWindow, FindWindow, EnumWindows, GetWindowText, GetWindowDC, DeleteObject, ReleaseDC, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap

from toolkit import Monitor, Timer


class Win:

    @staticmethod  # 不加静态声明会打印对象地址
    def __warning(*args):
        sys.stdout.write('\033[0;33m')
        size = len(args)
        for i, item in enumerate(args):
            sys.stdout.write(str(item))
            if i < size - 1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[0m')
        print()

    def __init__(self, title: str, region: tuple, interval=60):
        """
        title: window title, part of title is ok
        region: tuple, (left, top, width, height)
        """
        self.title = title
        self.region = region
        # 设置窗体句柄属性
        self.hwnd = None  # 截图的窗体句柄
        self.timestamp = None  # 上次成功设置句柄的时间戳
        self.interval = interval  # 秒, 更新间隔

    def __findHwnd(self):
        """
        找出符合标题条件的窗体(唯一), 返回该窗体的句柄
        None: 没有找到符合条件的窗体 / 符合条件的窗体不止一个
        """
        # 提示信息
        windowNotFoundMessage = f'Window [{self.title}] not found, capturer instance initialize failure'
        windowNotExplicitMessage = f'Window [{self.title}] not explicit, more than one window found, capturer instance initialize failure'
        # 枚举窗体
        windowHandleList = []
        EnumWindows(lambda hwnd, param: param.append(hwnd), windowHandleList)
        filteredWindowHandleList = []
        for hwnd in windowHandleList:
            if self.title in GetWindowText(hwnd):
                filteredWindowHandleList.append(hwnd)
        size = len(filteredWindowHandleList)
        if size == 0:
            self.__warning(windowNotFoundMessage)
            return None
        elif size > 1:
            message = windowNotExplicitMessage
            for i, hwnd in enumerate(filteredWindowHandleList):
                message += f'\r\n\t{i + 1}: {hwnd}, ' + GetWindowText(hwnd)
            self.__warning(message)
            return None
        # 符合条件的句柄
        return filteredWindowHandleList[0]

    def __updateHwnd(self):
        """
        在以下时机更新句柄
        1. 句柄属性为空时
        2. 时间戳超过指定更新间隔时
        """
        if (self.hwnd is None) or (self.timestamp is not None and time.perf_counter_ns() - self.timestamp > 1_000_000_000 * self.interval):
            hwnd = self.__findHwnd()
            if hwnd is not None:
                self.hwnd = hwnd
                self.timestamp = time.perf_counter_ns()
            else:
                self.hwnd = None
                self.timestamp = None

    def grab(self):
        """
        还有优化空间, 比如把各个HDC缓存起来, 在截图方法中每次执行BitBlt, 但是考虑到比较麻烦, 而且提升的效果也有限, 就先这样了
        """
        self.__updateHwnd()
        left, top, width, height = self.region
        hWinDC = GetWindowDC(self.hwnd)  # 具有要检索的设备上下文的窗口的句柄。 如果此值为 NULL， GetWindowDC 将检索整个屏幕的设备上下文。Null 时等同于调用 GetDesktopWindow 获得到句柄
        srcDC = CreateDCFromHandle(hWinDC)
        memDC = srcDC.CreateCompatibleDC()
        bmp = CreateBitmap()
        bmp.CreateCompatibleBitmap(srcDC, width, height)
        memDC.SelectObject(bmp)
        memDC.BitBlt((0, 0), (width, height), srcDC, (left, top), SRCCOPY)
        array = bmp.GetBitmapBits(True)
        DeleteObject(bmp.GetHandle())
        memDC.DeleteDC()
        srcDC.DeleteDC()
        ReleaseDC(self.hwnd, hWinDC)
        img = np.frombuffer(array, dtype='uint8')
        img.shape = (height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def grabWithWin(region):
    left, top, width, height = region
    hWin = GetDesktopWindow()
    hWinDC = GetWindowDC(hWin)
    srcDC = CreateDCFromHandle(hWinDC)
    memDC = srcDC.CreateCompatibleDC()
    bmp = CreateBitmap()
    bmp.CreateCompatibleBitmap(srcDC, width, height)
    memDC.SelectObject(bmp)
    memDC.BitBlt((0, 0), (width, height), srcDC, (left, top), SRCCOPY)
    array = bmp.GetBitmapBits(True)
    DeleteObject(bmp.GetHandle())
    memDC.DeleteDC()
    srcDC.DeleteDC()
    ReleaseDC(hWin, hWinDC)
    img = np.frombuffer(array, dtype='uint8')
    img.shape = (height, width, 4)
    return img


def grab(hWin: int, region):
    left, top, width, height = region
    hWinDC = GetWindowDC(hWin)
    srcDC = CreateDCFromHandle(hWinDC)
    memDC = srcDC.CreateCompatibleDC()
    bmp = CreateBitmap()
    bmp.CreateCompatibleBitmap(srcDC, width, height)
    memDC.SelectObject(bmp)
    memDC.BitBlt((0, 0), (width, height), srcDC, (left, top), SRCCOPY)
    array = bmp.GetBitmapBits(True)
    DeleteObject(bmp.GetHandle())
    memDC.DeleteDC()
    srcDC.DeleteDC()
    ReleaseDC(hWin, hWinDC)
    img = np.frombuffer(array, dtype='uint8')
    img.shape = (height, width, 4)
    return img


if __name__ == '__main__':

    w, h = Monitor.resolution.real()
    region = w // 7 * 3, h // 3, w // 7, h // 3
    region = 1000, 100, 320, 320
    print(region)

    win = Win('百度一下', region)

    print(FindWindow(None, '百度一下，你就知道 - 360极速浏览器X 21.0'))

    # title = 'Realtime Screen Capture'
    # while True:
    #
    #
    #     t1 = time.perf_counter_ns()
    #     img = win.grab()
    #     t2 = time.perf_counter_ns()
    #     cv2.putText(img, f'{Timer.cost(t2 - t1)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    #     # print(f'{Timer.cost(t2 - t1)}')
    #     cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow(title, img)
    #     SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    #     t3 = time.time()
    #     k = cv2.waitKey(1)  # 0:不自动销毁也不会更新, 1:1ms延迟销毁
    #     if k % 256 == 27:
    #         cv2.destroyAllWindows()
    #         exit('ESC ...')

    t1 = time.perf_counter_ns()
    for i in range(100):
        grabWithWin(region)
    t2 = time.perf_counter_ns()
    print(f'{Timer.cost(t2 - t1)}')

    hwnd = FindWindow('SunAwtFrame', 'python.yolo.apex.autoaim.helper – test.py Administrator')
    t1 = time.perf_counter_ns()
    for i in range(100):
        grab(hwnd, region)
    t2 = time.perf_counter_ns()
    print(f'{Timer.cost(t2 - t1)}')

    t1 = time.perf_counter_ns()
    for i in range(100):
        win.grab()
    t2 = time.perf_counter_ns()
    print(f'{Timer.cost(t2 - t1)}')

