import time

import cv2
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
from win32gui import FindWindow, SetWindowPos

from toolkit import Detector

region = (3440 // 5 * 2, 1440 // 3, 3440 // 5, 1440 // 3)
weight = 'weights.apex.public.dummy.pt'
detector = Detector(weight)

title = 'Realtime ScreenGrab Detect'
while True:

    _, img = detector.detect(region=region, image=True, label=True, confidence=True)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, img)
    SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    t3 = time.time()
    k = cv2.waitKey(1)  # 0:不自动销毁也不会更新, 1:1ms延迟销毁
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit('ESC ...')
