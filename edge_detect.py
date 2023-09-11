from detect import *
import time
import cv2
from win32gui import FindWindow, SetWindowPos
from win32con import HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE
from PIL import Image
import pytesseract
from find import find

from key_input.press_key import InputKey
from key_input import Mouse, Keyboard
input_key = InputKey(0)

center_point = 'center_point'
show = 'show'
size = 'size'


init = {
    center_point: (960, 540),  # 中心点
    show: True,  # 显示
    size: 250,  # 截图的尺寸, 屏幕中心截图周围大小      检测紫色点


}


def ocr_digit(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 使用Tesseract进行OCR识别
    text = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c itemised_char_whitelist=0123456789')

    # 返回识别的数字文本
    return text.strip()


def move(x: int, y: int):  # 不开镜模式下鼠标移动
    if (x == 0) & (y == 0):
        return

    screen_width, screen_height = pyautogui.size()

    # 计算屏幕中心点的坐标
    center_x = screen_width // 2
    center_y = screen_height // 2

    print(center_x, center_y)

    # 计算屏幕中心到目标点的相对位移
    x_offset = x - center_x
    y_offset = y - center_y
    print(int(x_offset), int(y_offset))
    input_key.mouse_move(x_offset, y_offset)




import keyboard
def loop():  # 主函数
    while True:

        # 检测是否按下 "h" 键
        if keyboard.is_pressed('h'):
            print("退出")
            break
        capture_screen_around_center(init[size])  # 截图
        b = cv2.imread('detect.png')
        target, image1 = find_specific_purple_edges('detect.png', show=init[show])  # 调用边缘检测求中心点
        if target is not None:
            x1, y1 = target
            aim = get_coordinate(init[size], x1, y1)  # 转为屏幕坐标
            px2 = aim[0]
            py2 = aim[1]
            time.sleep(0.5)
            print(px2, py2)
            move(px2, py2)  # 移动
            input_key.mouse_key_click(Mouse.MOUSE_LEFT)  # 开火
            print("开火")

        # 显示

        if init[show]:
            cv2.namedWindow('detect', cv2.WINDOW_AUTOSIZE)
            im = cv2.resize(image1, (400, 400))
            cv2.imshow('detect', im)
            SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
            cv2.waitKey(1)

        else:
            # 显示
            if init[show]:
                resized_img = cv2.resize(b, (400, 400))
                cv2.imshow('detect', resized_img)
                SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                cv2.waitKey(1)
            else:
                # 显示
                print("未能找到目标")
                if init[show]:
                    img = cv2.resize(b, (400, 400))
                    cv2.imshow('detect', img)
                    SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                    cv2.waitKey(1)


if __name__ == '__main__':
    loop()
