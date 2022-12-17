import os.path
import time

import cv2
import mss
import numpy as np
import torch
from win32api import GetSystemMetrics
from win32con import SRCCOPY, SM_CXSCREEN, SM_CYSCREEN
from win32gui import GetDesktopWindow, GetWindowDC, DeleteObject, ReleaseDC
from win32ui import CreateDCFromHandle, CreateBitmap

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


class Capturer:

    @staticmethod
    def grabWithWin(region):
        """
        region: tuple, (left, top, width, height)
        conda install pywin32, 用 pip 装的一直无法导入 win32ui 模块, 找遍各种办法都没用, 用 conda 装的一次成功
        """
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

    @staticmethod
    def getMssInstance():
        return mss.mss()

    @staticmethod
    def grabWithMss(instance, region):
        """
        region: tuple, (left, top, width, height)
        pip install mss
        """
        left, top, width, height = region
        return instance.grab(monitor={'left': left, 'top': top, 'width': width, 'height': height})

    @staticmethod
    def grab(win=False, mss=False, instance=None, region=None, convert=False):
        """
        win:
            region: tuple, (left, top, width, height)
        mss:
            instance: mss instance
            region: tuple, (left, top, width, height)
        convert: 是否转换为 opencv 需要的 numpy BGR 格式, 转换结果可直接用于 opencv
        """
        # 补全范围
        if not region:
            w, h = Monitor.resolution()
            region = 0, 0, w, h
        # 范围截图
        if win:
            img = Capturer.grabWithWin(region)
        elif mss:
            img = Capturer.grabWithMss(instance, region)
        else:
            win = True
            img = Capturer.grabWithWin(region)
        # 图片转换
        if convert:
            if win:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif mss:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        return img


class Monitor:

    @staticmethod
    def resolution():
        """
        显示分辨率
        """
        w = GetSystemMetrics(SM_CXSCREEN)
        h = GetSystemMetrics(SM_CYSCREEN)
        return w, h

    @staticmethod
    def center():
        """
        屏幕中心点
        """
        w, h = Monitor.resolution()
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
    def __init__(self, weights):
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
        self.classes = None  # filter by class: --class 0, or --class 0 2 3, 数字, 需要自己将类别转成类别索引
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
    def detect(self, region, classes=None, image=False, label=True, confidence=True):
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
        if image:
            annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            im0 = img0
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                clazz = self.names[c] if not self.weights.endswith('.engine') else str(c)  # 类别
                if classes and clazz not in classes:
                    continue
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
                aims.append((clazz, float(conf), (sx, sy), (gx, gy), (sl, st, sw, sh), (gl, gt, gw, gh)))
                if image:
                    label2 = (f'{clazz} {conf:.2f}' if confidence else f'{clazz}') if label else None
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
        print(f'截图:{Timer.cost(t2 - t1)}, 检测:{Timer.cost(t3 - t2)}, 总计:{Timer.cost(t3 - t1)}, 数量:{len(aims)}/{len(det)}')
        return aims, annotator.result() if image else None

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
