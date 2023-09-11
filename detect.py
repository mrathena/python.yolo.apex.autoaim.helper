import sys
from win32api import GetSystemMetrics
from win32con import SRCCOPY, SM_CXSCREEN, SM_CYSCREEN, DESKTOPHORZRES, DESKTOPVERTRES
from win32gui import GetDesktopWindow, GetWindowDC, DeleteObject, GetDC, ReleaseDC, FindWindow
from win32ui import CreateDCFromHandle, CreateBitmap
from win32print import GetDeviceCaps
import mss
import pyautogui
import easyocr
import time
import pandas as pd
import cv2
import numpy as np

center_point = 'center_point'
show = 'show'

init = {
    center_point: (644, 377),  # 中心点
    show: True,  # 显示

}


def extract_text_from_image(image_path):
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, model_storage_directory="model", download_enabled=True)
    start_time = time.time()
    info = reader.readtext(image=image_path)
    print('cpu need time', time.time() - start_time, 's')

    df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'text', 'proba'])

    for i, item in enumerate(info):
        # 保留左上和右下坐标
        ((x1, y1), _, (x2, y2), _), text, prob = item
        df.loc[i] = [x1, y1, x2, y2, text, prob]
    # 缓存结果，不重复识别
    df.to_csv('road-poetry.csv')
    data = pd.read_csv('road-poetry.csv')

    # 访问x1, y1, x2, y2字段
    x1_values = data['x1']
    y1_values = data['y1']
    x2_values = data['x2']
    y2_values = data['y2']

    # 计算中心点并添加到DataFrame中
    data['cx'] = x2_values - x1_values
    data['cy'] = y2_values - y1_values

    # 输出数据示例
    for i in range(len(data)):
        cx = data['cx'][i]
        cy = data['cy'][i]
        print(f"x1: {x1_values[i]}, y1: {y1_values[i]}, x2: {x2_values[i]}, y2: {y2_values[i]}, cx: {cx}, cy: {cy}")

    # 将中心点坐标返回为一个列表
    center_points = [(cx, cy) for cx, cy in zip(data['cx'], data['cy'])]

    return center_points


# 读取CSV数据文件
# Call the function and pass the image file path as an argument
# image_file_path = '111.jpg'
# extracted_data = extract_text_from_image(image_file_path)
# print(extracted_data)

def get_coordinate(size, x, y):
    a = init[center_point]
    # 假设原始图像在屏幕中的位置和大小，以及检测得到的图像中心点坐标
    screen_center_x, screen_center_y = a[0], a[1]
    image_width = size  # 图像的宽度
    image_height = size  # 图像的高度
    image_center_x = x  # 检测得到的图像中心点在图像坐标系中的x坐标
    image_center_y = y  # 检测得到的图像中心点在图像坐标系中的y坐标
    screen_x = screen_center_x - image_width + image_center_x
    screen_y = screen_center_y - image_height + image_center_y

    print("屏幕坐标 x:", screen_x)
    print("屏幕坐标 y:", screen_y)
    return screen_x, screen_y


def find_purple_points(image_path, target_point=(960, 540)):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义紫色的HSV范围
    lower_purple = np.array([135, 55, 55])
    upper_purple = np.array([160, 255, 255])

    # 提取紫色区域
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

    # 查找紫色区域中的点的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算紫色点的中心坐标
    purple_points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            purple_points.append((cX, cY))

    # 如果找不到紫色点，返回None
    if not purple_points:
        return None, None, None

    # 找到距离给定点最近的紫色点的中心坐标
    nearest_point = min(purple_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(target_point)))

    x, y = nearest_point[0], nearest_point[1]
    # 在图像上绘制红色圆点标记
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    return image, x, y


# 替换为您的图像文件路径


def capture_screenshot(top, left, right, height, output_file):
    width = right - left
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        sct_img = sct.shot(output=output_file, output_format="png", mon=monitor)
    return sct_img


def find_specific_purple_edges(image_path, show):
    # 读取图像
    img = cv2.imread(image_path)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义紫色的HSV范围
    lower_purple = np.array([135, 55, 55])
    upper_purple = np.array([160, 255, 255])
    # 创建紫色掩膜
    purple_mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

    # 进行边缘检测
    edges = cv2.Canny(purple_mask, 50, 150, apertureSize=3)

    # 查找紫色边缘的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最左端、最右端、最高处和最低处的位置
    leftmost = (img.shape[1], 0)
    rightmost = (0, 0)
    topmost = (0, img.shape[0])
    bottommost = (0, 0)

    # 遍历所有紫色边缘的轮廓
    for contour in contours:
        for point in contour:
            x, y = point[0]

            # 更新最左端、最右端、最高处和最低处的位置
            if x < leftmost[0]:
                leftmost = (x, y)
            if x > rightmost[0]:
                rightmost = (x, y)
            if y < topmost[1]:
                topmost = (x, y)
            if y > bottommost[1]:
                bottommost = (x, y)

    # 检查是否找到紫色边缘
    if not contours:
        return None, img

    # 计算中心点
    center_x = (leftmost[0] + rightmost[0]) // 2
    center_y = (topmost[1] + bottommost[1]) // 2
    # Calculate the new center_y at 40% of its original position
    height = bottommost[1] - topmost[1]
    new_height = height * 15 // 100
    new_center_y = center_y + new_height
    center = (center_x, new_center_y)

    if show:
        # 在图像上绘制中心点和轮廓点
        cv2.circle(img, center, 5, (0, 255, 0), -1)  # 绘制中心点，使用绿色填充
        for contour in contours:
            for point in contour:
                x, y = point[0]
                cv2.circle(img, (x, y), 2, (255, 0, 255), -1)  # 绘制紫色点，半径为2

    return center, img


# Example usage:
# center, output_img = find_specific_purple_edges("path/to/your/image.jpg", show=True)
# if center is None:
#     print("No purple edges found.")
# else:
#     cv2.imshow("Output Image", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




def capture_screen_around_center(s):
    center_x, center_y = init[center_point][0], init[center_point][1]
    left = center_x - s
    top = center_y - s
    right = center_x + s
    bottom = center_y + s

    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = screenshot
    img.save('detect.png')
    return img


from key_input.press_key import InputKey
from key_input import Keyboard

input_key = InputKey(0)
from PIL import Image, ImageGrab
import pytesseract


def ocr_digit(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 使用Tesseract进行OCR识别
    text = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    # 返回识别的数字文本
    return text.strip()




def capture_screen_around_centers(s):
    center_x, center_y = init[center_point][0], init[center_point][1]
    left = center_x - s
    top = center_y - s
    right = center_x + s
    bottom = center_y + s

    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = screenshot
    img.save('detect_full.png')
    return img


class Capturer:

    def __init__(self, title: str, region: tuple, interval=60):
        """
        title: 完整的窗体标题, 不支持模糊(因为没有必要)
        region: tuple, (left, top, width, height)
        """
        self.title = title
        self.region = region
        # 设置窗体句柄属性
        self.hwnd = None  # 截图的窗体句柄
        self.timestamp = None  # 上次成功设置句柄的时间戳
        self.interval = interval  # 秒, 更新间隔

    def grab(self):
        """
        还有优化空间, 比如把各个HDC缓存起来, 在截图方法中每次执行BitBlt, 但是考虑到比较麻烦, 而且提升的效果也有限, 就先这样了
        """
        # 检查并按需更新句柄等参数, 在以下时机更新句柄, 1. 句柄属性为空时; 2. 时间戳超过指定更新间隔时
        if (self.hwnd is None) or (
                self.timestamp is not None and time.perf_counter_ns() - self.timestamp > 1_000_000_000 * self.interval):
            hwnd = FindWindow(None, self.title)  # 找到第一个指定标题的窗体并返回其句柄
            if hwnd != 0:
                self.hwnd = hwnd
                self.timestamp = time.perf_counter_ns()
            else:
                Printer.warning(f'未找到标题为 [{self.title}] 的窗体')
                self.hwnd = None
                self.timestamp = None
        # 获取设备上下文
        left, top, width, height = self.region
        try:
            hWinDC = GetWindowDC(
                self.hwnd)  # 具有要检索的设备上下文的窗口的句柄。 如果此值为 NULL， GetWindowDC 将检索整个屏幕的设备上下文。等同于调用 GetDesktopWindow() 获得的句柄?
        except BaseException:  # pywintypes.error: (1400, 'GetWindowDC', '无效的窗口句柄。'). 可通过 BaseException 捕获, 通过如右方式判断, if e.args[0] == 1400: pass
            # 此时的句柄不能正常使用, 需要清空并重新获取句柄
            self.hwnd = None
            self.timestamp = None
            # 使用替代句柄
            hWinDC = GetWindowDC(GetDesktopWindow())
        try:
            srcDC = CreateDCFromHandle(hWinDC)
            memDC = srcDC.CreateCompatibleDC()
            bmp = CreateBitmap()
            bmp.CreateCompatibleBitmap(srcDC, width, height)
            memDC.SelectObject(bmp)
            memDC.BitBlt((0, 0), (width, height), srcDC, (left, top), SRCCOPY)
            array = bmp.GetBitmapBits(True)
            img = np.frombuffer(array, dtype='uint8')
            img.shape = (height, width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            DeleteObject(bmp.GetHandle())
            memDC.DeleteDC()
            srcDC.DeleteDC()
            ReleaseDC(self.hwnd, hWinDC)
            return img
        except BaseException:
            return None

    @staticmethod
    def backup(region):
        """
        region: tuple, (left, top, width, height)
        """
        left, top, width, height = region
        hWin = GetDesktopWindow()
        # hWin = FindWindow(完整类名, 完整窗体标题名)
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
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


class Monitor:
    class resolution:

        @staticmethod
        def show():
            """
            显示分辨率
            """
            w = GetSystemMetrics(SM_CXSCREEN)
            h = GetSystemMetrics(SM_CYSCREEN)
            return w, h

        @staticmethod
        def real():
            """
            物理分辨率
            """
            hDC = GetDC(None)
            w = GetDeviceCaps(hDC, DESKTOPHORZRES)
            h = GetDeviceCaps(hDC, DESKTOPVERTRES)
            ReleaseDC(None, hDC)
            return w, h

        @staticmethod
        def center():
            """
            物理屏幕中心点
            """
            w, h = Monitor.resolution.real()
            return w // 2, h // 2


class Timer:

    @staticmethod
    def cost(interval):
        """
        转换耗时, 输入纳秒间距, 转换为合适的单位
        """
        if interval < 1000:
            return f'{interval}ns'
        elif interval < 1_000_000:
            return f'{round(interval / 1000, 3)}us'
        elif interval < 1_000_000_000:
            return f'{round(interval / 1_000_000, 3)}ms'
        else:
            return f'{round(interval / 1_000_000_000, 3)}s'


class Printer:
    """
    开头部分：\033[显示方式;前景色;背景色m
    结尾部分：\033[0m
    显示方式: 0（默认值）、1（高亮，即加粗）、4（下划线）、7（反显）、
    前景色: 30（黑色）、31（红色）、32（绿色）、 33（黄色）、34（蓝色）、35（梅色）、36（青色）、37（白色）
    背景色: 40（黑色）、41（红色）、42（绿色）、 43（黄色）、44（蓝色）、45（梅色）、46（青色）、47（白色）
    """

    @staticmethod
    def danger(*args):
        sys.stdout.write('\033[0;31m')
        size = len(args)
        for i, item in enumerate(args):
            sys.stdout.write(str(item))
            if i < size - 1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[0m')
        print()

    @staticmethod
    def warning(*args):
        sys.stdout.write('\033[0;33m')
        size = len(args)
        for i, item in enumerate(args):
            sys.stdout.write(str(item))
            if i < size - 1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[0m')
        print()

    @staticmethod
    def info(*args):
        sys.stdout.write('\033[0;36m')
        size = len(args)
        for i, item in enumerate(args):
            sys.stdout.write(str(item))
            if i < size - 1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[0m')
        print()

    @staticmethod
    def success(*args):
        sys.stdout.write('\033[0;32m')
        size = len(args)
        for i, item in enumerate(args):
            sys.stdout.write(str(item))
            if i < size - 1:
                sys.stdout.write(' ')
        sys.stdout.write('\033[0m')
        print()


class Predictor:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, point):
        x, y = point
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        px, py = int(predicted[0]), int(predicted[1])
        return px, py


def find_names_gray(tp):
    # 定义HSV颜色范围
    lower_purple = np.array([130, 55, 55])
    upper_purple = np.array([160, 255, 255])

    # 读取图像
    image = cv2.imread(tp)  # 替换
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 创建一个掩码，显示在指定HSV范围内的颜色
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    # 通过在原始图像上应用掩码，只显示在指定HSV范围内的颜色
    cv2.bitwise_and(image, image, mask=mask)
    # 将掩膜应用于原始图像
    purple_masked_image = cv2.bitwise_and(image, image, mask=mask)
    # 将图像黑白化（灰度化）
    gray_image = cv2.cvtColor(purple_masked_image, cv2.COLOR_BGR2GRAY)
    output_path = 'ocr_detect.jpg'
    cv2.imwrite(output_path, gray_image)
    return gray_image
