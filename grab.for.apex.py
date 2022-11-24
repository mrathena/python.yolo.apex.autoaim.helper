import time

import cv2
import pynput
import winsound

from toolkit import Monitor


def keyboard():

    # region = (3440 // 7 * 3, 1440 // 3, 3440 // 7, 1440 // 3)
    region = (1450, 140, 1280, 1280)

    def release(key):
        if key == pynput.keyboard.Key.end:
            winsound.Beep(400, 200)
            return False
        elif key == pynput.keyboard.KeyCode.from_char('f'):
            name = f'D:\\resource\\develop\\python\\dataset.yolo.v5\\apex\\action\\data\\{int(time.time())}.png'
            print(name)
            # img = Monitor.grab(region)
            # mss.tools.to_png(img.rgb, img.size, output=name)
            img = Monitor.grab(region, convert=True)
            cv2.imwrite(name, img)
            winsound.Beep(800, 200)

    with pynput.keyboard.Listener(on_release=release) as k:
        k.join()


keyboard()
