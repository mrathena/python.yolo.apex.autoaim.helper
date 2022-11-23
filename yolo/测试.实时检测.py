import time

import cv2
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
from win32gui import FindWindow, SetWindowPos

from toolkit import Detector, Timer

region = (3440 // 7 * 3, 1440 // 3, 3440 // 7, 1440 // 3)
weight = 'yolov7-tiny.pt'
detector = Detector(weight)

title = 'Realtime ScreenGrab Detect'
while True:

    t = time.perf_counter_ns()
    _, img = detector.detect(region=region, image=True)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.putText(img, f'{Timer.cost(time.perf_counter_ns() - t)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    cv2.imshow(title, img)
    # 寻找窗口, 设置置顶
    SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    k = cv2.waitKey(1)  # 0:不自动销毁也不会更新, 1:1ms延迟销毁
    if k % 256 == 27:
        # ESC 关闭窗口
        cv2.destroyAllWindows()
        exit('ESC ...')
