from detect_rang import *
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
ranging_speed = 'ranging_speed'
show = 'show'
size = 'size'
size_point = 'size_point'
speed_w = 'speed_w'

init = {
    center_point: (644, 377),  # 中心点
    ranging_speed: 0.025,  # 装表速度
    show: True,  # 显示
    size: 250 // 2,  # 截图的尺寸, 屏幕中心截图周围大小      检测紫色点
    size_point: 450 // 2,  # 截图的尺寸, 屏幕中心截图周围大小      检测紫色边缘
    speed_w: (31, 721, 61, 735)  # 速度的屏幕坐标(左上角x,左上角y，右下角x,右下角y)
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

    center_x = 643  # 要自行调整，数越大越靠右，越小越靠左    游戏特性无法更改  因为战雷鼠标不在屏幕中心点
    center_y = 230  # 要自行调整，数越大越靠上，越小越靠下

    print(center_x, center_y)

    # 计算屏幕中心到目标点的相对位移
    x_offset = x - center_x
    y_offset = y - center_y
    print(int(x_offset), int(y_offset))
    input_key.mouse_move(x_offset, y_offset)


def move1(x: int, y: int):  # 开镜模式下鼠标移动
    if (x == 0) & (y == 0):
        return
    center_x = 643
    center_y = 380  # 可能要-5
    # 计算屏幕中心到目标点的相对位移
    x_offset = x - center_x
    y_offset = y - center_y
    print(x_offset, y_offset)
    input_key.mouse_move(x_offset, y_offset)


def loop():  # 主函数
    while True:
        box1 = find('pic/0.png', 0.6)
        if box1 is None:
            print("瞄准控制停止")
            break
        capture_screen_around_centers(init[size_point])  # 截图
        # 替换为您的图像文件路径
        image_path = 'detect_full.png'  # 原理是截图然后处理最后显示
        image = cv2.imread(image_path)
        result_image, x, y = find_purple_points(image_path,
                                                target_point=(init[size_point], init[size_point]))  # 找离屏幕中心最近的紫色点
        if result_image is not None:
            px, py = get_coordinate(init[size_point], x, y)  # 转为屏幕坐标
            move(px, py)  # 移动
            input_key.click_key(Keyboard.X, 0.1)  # 按X
            time.sleep(0.5)
            input_key.click_key(Keyboard.LSHIFT, 0.1)  # 开镜
            # 显示标记了最近紫色点的中心坐标的图像
            time.sleep(0.5)
            # 开镜后处理
            result_image, x, y = find_purple_points(image_path,
                                                    target_point=(init[size_point], init[size_point]))  # 找离屏幕中心最近的紫色点
            px1, py1 = get_coordinate(init[size_point], x, y)
            move1(px1, py1)
            while True:
                capture_screen_around_center(init[size])  # 截图
                target, image1 = find_specific_purple_edges('detect.png', show=init[show])  # 调用边缘检测求中心点
                if target is not None:
                    x1, y1 = target
                    aim = get_coordinate(init[size], x1, y1)  # 转为屏幕坐标
                    px2 = aim[0]
                    py2 = aim[1]
                    time.sleep(0.5)
                    print(px2, py2)
                    move1(px2, py2)  # 移动
                    range_detect()
                    # input_key.click_key(Keyboard.U, 5)归0
                    input_key.mouse_key_click(Mouse.MOUSE_LEFT)  # 开火
                    print("开火")
                    time.sleep(1)
                    # 显示
                if target is None:
                    input_key.click_key(Keyboard.LSHIFT, 0.1)
                    print("完成循环")
                    break
                if init[show]:
                    cv2.namedWindow('detect', cv2.WINDOW_AUTOSIZE)
                    im = cv2.resize(image1, (400, 400))
                    cv2.imshow('detect', im)
                    SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                    cv2.waitKey(1)
                else:
                    # 显示
                    if init[show]:
                        resized_img = cv2.resize(result_image, (400, 400))
                        cv2.imshow('detect', resized_img)
                        SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                        cv2.waitKey(1)
        else:
            # 显示
            print("未能找到目标")
            if init[show]:
                img = cv2.resize(image, (400, 400))
                cv2.imshow('detect', img)
                SetWindowPos(FindWindow(None, 'detect'), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
                cv2.waitKey(1)


if __name__ == '__main__':
    loop()
