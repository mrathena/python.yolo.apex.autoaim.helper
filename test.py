import multiprocessing
import time
from multiprocessing import Process

import cv2
from pynput.mouse import Button, Listener
from win32gui import FindWindow, SetWindowPos, GetWindowText, GetForegroundWindow
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
import winsound
from toolkit import Capturer

title = 'Realtime Screen Capture Detect'
region = 3440 // 5 * 2, 1440 // 3, 3440 // 5, 1440 // 3



# while True:

for i in range(5):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, Capturer.backup(region))
    SetWindowPos(FindWindow(None, title), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    cv2.waitKey(1)
    time.sleep(1)
winsound.Beep(400, 200)

time.sleep(1)
winsound.Beep(400, 200)
