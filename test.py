import time

import cv2
import numpy as np
from win32con import SRCCOPY, HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
from win32gui import GetDesktopWindow, FindWindow, EnumWindows, GetWindowText, GetWindowDC, DeleteObject, ReleaseDC, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap

from toolkit import Monitor, Timer


class Win:

    def __init__(self, title: str, region: tuple):
        """
        region: tuple, (left, top, width, height)
        """
        # 找到窗体
        windowNotFoundMessage = f'Window [{title}] not found, capturer instance initialize failure'
        windowNotExplicitMessage = f'Window [{title}] not explicit, more than one window found, capturer instance initialize failure'
        """
        # 方式一, 需要完整窗体标题
        windowHandle = FindWindow(None, title)
        if 0 == windowHandle:
            raise Exception(windowNotFoundMessage)
        self.windowHandle = windowHandle
        """
        # 方式二, 需要部分窗体标题
        windowHandleList = []
        EnumWindows(lambda hwnd, param: param.append(hwnd), windowHandleList)
        resultHandleList = []
        for hwnd in windowHandleList:
            if title in GetWindowText(hwnd):
                resultHandleList.append(hwnd)
        size = len(resultHandleList)
        if size == 0:
            raise Exception(windowNotFoundMessage)
        elif size > 1:
            message = windowNotExplicitMessage
            for i, hwnd in enumerate(resultHandleList):
                message += f'\r\n\t{i + 1}: {hwnd}, ' + GetWindowText(hwnd)
            raise Exception(message)
        self.windowHandle = resultHandleList[0]
        self.region = region
        # 初始化
        left, top, width, height = region
        self.windowDCHandle = GetWindowDC(self.windowHandle)
        self.sourceDCHandle = CreateDCFromHandle(self.windowDCHandle)
        self.memoryDCHandle = self.sourceDCHandle.CreateCompatibleDC()
        self.bmp = CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.sourceDCHandle, width, height)
        self.memoryDCHandle.SelectObject(self.bmp)

    def __del__(self):
        DeleteObject(self.bmp.GetHandle())
        self.memoryDCHandle.DeleteDC()
        self.sourceDCHandle.DeleteDC()
        ReleaseDC(self.windowHandle, self.windowDCHandle)

    def grab(self):
        left, top, width, height = self.region
        self.memoryDCHandle.BitBlt((0, 0), (width, height), self.sourceDCHandle, (left, top), SRCCOPY)
        array = self.bmp.GetBitmapBits(True)
        image = np.frombuffer(array, dtype='uint8')
        image.shape = (height, width, 4)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image


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
    region = w // 5 * 2, h // 3, w // 5, h // 3
    print(region)

    win = Win('yolo', region)


    title = 'Realtime Screen Capture'
    while True:

        t1 = time.perf_counter_ns()
        img = win.grab()
        t2 = time.perf_counter_ns()
        cv2.putText(img, f'Capture: {Timer.cost(t2 - t1)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, img)
        SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        t3 = time.time()
        k = cv2.waitKey(1)  # 0:不自动销毁也不会更新, 1:1ms延迟销毁
        if k % 256 == 27:
            cv2.destroyAllWindows()
            exit('ESC ...')




    """t1 = time.perf_counter_ns()
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
    print(f'{Timer.cost(t2 - t1)}')"""

