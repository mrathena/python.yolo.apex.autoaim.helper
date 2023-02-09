import os.path
import sys
import time

import cv2
import numpy as np
import torch
from win32api import GetSystemMetrics
from win32con import SRCCOPY, SM_CXSCREEN, SM_CYSCREEN, DESKTOPHORZRES, DESKTOPVERTRES
from win32gui import GetDesktopWindow, GetWindowDC, DeleteObject, GetDC, ReleaseDC, EnumWindows, GetWindowText
from win32ui import CreateDCFromHandle, CreateBitmap
from win32print import GetDeviceCaps

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


class Capturer:

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
        windowNotFoundMessage = f'未找到标题中包含 [{self.title}] 的窗体'
        windowNotExplicitMessage = f'找到多个标题中包含 [{self.title}] 的窗体, 需确保根据给出的标题条件找到的窗体唯一'
        # 枚举窗体
        windowHandleList = []
        EnumWindows(lambda hwnd, param: param.append(hwnd), windowHandleList)
        filteredWindowHandleList = []
        for hwnd in windowHandleList:
            if self.title in GetWindowText(hwnd):
                filteredWindowHandleList.append(hwnd)
        size = len(filteredWindowHandleList)
        if size == 0:
            Printer.warning(windowNotFoundMessage)
            return None
        elif size > 1:
            message = windowNotExplicitMessage
            for i, hwnd in enumerate(filteredWindowHandleList):
                message += f'\r\n\t{i + 1}: {hwnd}, ' + GetWindowText(hwnd)
            Printer.warning(message)
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


class Detector:

    @smart_inference_mode()
    def __init__(self, weights, classes=None):
        self.weights = weights
        self.source = 'data/images'  # file/dir/URL/glob, 0 for webcam
        self.data = 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.nosave = False  # do not save images/videos
        self.classes = classes  # filter by class: --class 0, or --class 0 2 3, 数字, 需要自己将类别转成类别索引, None 检测全部标签
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False,  # update all models
        self.project = 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 2  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference

        # 加载模型
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        # print(f'设备:{self.device.type}, 模型:{self.model.weights}')
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        bs = 1
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup

    @smart_inference_mode()
    def detect(self, image, show=False):
        img0 = image
        # t1 = time.perf_counter_ns()
        aims = []
        im = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        det = pred[0]
        annotator = None
        if show:
            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            im0 = img0
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # if conf < confidence:
                #     continue
                c = int(cls)  # integer class
                clazz = self.names[c] if not self.weights.endswith('.engine') else str(c)  # 类别
                aims.append((c, clazz, float(conf), xyxy))  # 类别索引, 类别名称, 置信度, xyxy
                if show:
                    label = f'{c}:{clazz} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(0, True))
        # print(f'检测:{Timer.cost(time.perf_counter_ns() - t1)}, 数量:{len(aims)}/{len(det)}')
        return aims, annotator.result() if show else None

    def convert(self, aims, region):
        """
        将截屏坐标系下的 xyxy 转换为 屏幕坐标下下的 ltwhxy 和 截屏坐标系下的 ltwhxy
        """
        lst = []
        for item in aims:
            c, clazz, conf, xyxy = item
            # 屏幕坐标系下, 框的 ltwh 和 框的中心点 xy
            sl = int(region[0] + xyxy[0])
            st = int(region[1] + xyxy[1])
            sw = int(xyxy[2] - xyxy[0])
            sh = int(xyxy[3] - xyxy[1])
            sx = int(sl + sw / 2)
            sy = int(st + sh / 2)
            # 截图坐标系下, 框的 ltwh 和 框的中心点 xy
            gl = int(xyxy[0])
            gt = int(xyxy[1])
            gw = int(xyxy[2] - xyxy[0])
            gh = int(xyxy[3] - xyxy[1])
            gx = int((xyxy[0] + xyxy[2]) / 2)
            gy = int((xyxy[1] + xyxy[3]) / 2)
            lst.append((c, clazz, float(conf), (sx, sy), (gx, gy), (sl, st, sw, sh), (gl, gt, gw, gh)))
        return lst

    @smart_inference_mode()
    def backup(self, region, show=False, label=True, confidence=True):
        # 截图和转换
        t1 = time.perf_counter_ns()
        # 截屏范围 region = (left, top, width, height)
        img0 = Capturer.grab(win=True, region=region, convert=True)
        t2 = time.perf_counter_ns()
        # 检测
        aims = []
        im = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        det = pred[0]
        annotator = None
        if show:
            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            im0 = img0
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                clazz = self.names[c] if not self.weights.endswith('.engine') else str(c)  # 类别
                # 屏幕坐标系下, 框的 ltwh 和 框的中心点 xy
                sl = int(region[0] + xyxy[0])
                st = int(region[1] + xyxy[1])
                sw = int(xyxy[2] - xyxy[0])
                sh = int(xyxy[3] - xyxy[1])
                sx = int(sl + sw / 2)
                sy = int(st + sh / 2)
                # 截图坐标系下, 框的 ltwh 和 框的中心点 xy
                gl = int(xyxy[0])
                gt = int(xyxy[1])
                gw = int(xyxy[2] - xyxy[0])
                gh = int(xyxy[3] - xyxy[1])
                gx = int((xyxy[0] + xyxy[2]) / 2)
                gy = int((xyxy[1] + xyxy[3]) / 2)
                # confidence 置信度
                aims.append((c, clazz, float(conf), (sx, sy), (gx, gy), (sl, st, sw, sh), (gl, gt, gw, gh)))
                if show:
                    label2 = (f'{c}:{clazz} {conf:.2f}' if confidence else f'{clazz}') if label else None
                    annotator.box_label(xyxy, label2, color=colors(0, True))
                    # 下面是自己写的给框中心画点, 在 Annotator 类所在的 plots.py 中的 box_label 方法下添加如下方法
                    """
                    def circle(self, center, radius, color, thickness=None, lineType=None, shift=None):
                        cv2.circle(self.im, center, radius, color, thickness=thickness, lineType=lineType, shift=shift)
                    """
                    """
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    annotator.circle((cx, cy), 1, colors(6, True), 2)
                    """
        t3 = time.perf_counter_ns()
        # print(f'截图:{Timer.cost(t2 - t1)}, 检测:{Timer.cost(t3 - t2)}, 总计:{Timer.cost(t3 - t1)}, 数量:{len(aims)}/{len(det)}')
        return aims, annotator.result() if show else None

    @smart_inference_mode()
    def label(self, path):
        img0 = cv2.imread(path)
        im = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        det = pred[0]
        result = []
        if len(det):
            im0 = img0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                c = int(cls)  # integer class
                result.append((c, xywh))
        if result:
            directory = os.path.dirname(path)
            filename = os.path.basename(path)
            basename, ext = os.path.splitext(filename)
            name = os.path.join(directory, basename + '.txt')
            print(name)
            with open(name, 'w') as file:
                for item in result:
                    index, xywh = item
                    file.write(f'{index} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
