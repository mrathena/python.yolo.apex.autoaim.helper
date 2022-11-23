import os
import time

import cv2
import d3dshot  # pip install git+https://github.com/fauskanger/D3DShot#egg=D3DShot
import mss as pymss  # pip install mss
import numpy as np
import torch
from win32api import GetSystemMetrics  # conda install pywin32
from win32con import SRCCOPY, SM_CXSCREEN, SM_CYSCREEN
from win32gui import GetDesktopWindow, GetWindowDC, DeleteObject, ReleaseDC
from win32ui import CreateDCFromHandle, CreateBitmap
import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Capturer:

    @staticmethod
    def win(region):
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
    def mss(instance, region):
        """
        region: tuple, (left, top, width, height)
        pip install mss
        """
        left, top, width, height = region
        return instance.grab(monitor={'left': left, 'top': top, 'width': width, 'height': height})

    @staticmethod
    def d3d(instance, region=None):
        """
        DXGI 普通模式
        region: tuple, (left, top, width, height)
        因为 D3DShot 在 Python 3.9 里会和 pillow 版本冲突, 所以使用大佬修复过的版本来替代
        pip install git+https://github.com/fauskanger/D3DShot#egg=D3DShot
        """
        if region:
            left, top, width, height = region
            return instance.screenshot((left, top, left + width, top + height))
        else:
            return instance.screenshot()

    @staticmethod
    def d3d_latest_frame(instance):
        """
        DXGI 缓存帧模式
        """
        return instance.get_latest_frame()

    @staticmethod
    def instance(mss=False, d3d=False, buffer=False, frame_buffer_size=60, target_fps=60, region=None):
        if mss:
            return pymss.mss()
        elif d3d:
            """
            buffer: 是否使用缓存帧模式
                否: 适用于 dxgi.screenshot
                是: 适用于 dxgi.get_latest_frame, 需传入 frame_buffer_size, target_fps, region
            """
            if not buffer:
                return d3dshot.create(capture_output="numpy")
            else:
                dxgi = d3dshot.create(capture_output="numpy", frame_buffer_size=frame_buffer_size)
                left, top, width, height = region
                dxgi.capture(target_fps=target_fps, region=(left, top, left + width, top + height))  # region: left, top, right, bottom, 需要适配入参为 left, top, width, height 格式的 region
                return dxgi

    @staticmethod
    def grab(win=False, mss=False, d3d=False, instance=None, region=None, buffer=False, convert=False):
        """
        win:
            region: tuple, (left, top, width, height)
        mss:
            instance: mss instance
            region: tuple, (left, top, width, height)
        d3d:
            buffer: 是否为缓存帧模式
                否: 需要 region
                是: 不需要 region
            instance: d3d instance, 区分是否为缓存帧模式
            region: tuple, (left, top, width, height), 区分是否为缓存帧模式
        convert: 是否转换为 opencv 需要的 numpy BGR 格式, 转换结果可直接用于 opencv
        """
        # 补全范围
        if (win or mss or (d3d and not buffer)) and not region:
            w, h = Monitor.resolution()
            region = 0, 0, w, h
        # 范围截图
        if win:
            img = Capturer.win(region)
        elif mss:
            img = Capturer.mss(instance, region)
        elif d3d:
            if not buffer:
                img = Capturer.d3d(instance, region)
            else:
                img = Capturer.d3d_latest_frame(instance)
        else:
            img = Capturer.win(region)
            win = True
        # 图片转换
        if convert:
            if win:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif mss:
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
            elif d3d:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

    def __init__(self, weights):
        self.weights = weights
        self.source = 'inference/images'  # file/folder, 0 for webcam
        self.imgsz = 640  # inference size (pixels)
        self.conf_thres = 0.25  # object confidence threshold, 不能是0, 不然会检测出大量目标, 显存爆炸, 卡死进程
        self.iou_thres = 0  # IOU threshold for NMS
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # display results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.update = False,  # update all models
        self.project = 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.trace = False  # trace model, 要改成 False, 不然每次都生成 traced_model.pt, 是权重文件的两倍大, 伤固态
        # 加载模型
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def detect(self, region, classes=None, image=False, label=True, confidence=True):
        # 截图和转换
        t1 = time.perf_counter_ns()
        # 此 IMG 经过了转化, 和 cv2.read 读到的格式是一样的
        img0 = Capturer.grab(win=True, region=region, convert=True)
        t2 = time.perf_counter_ns()
        # 检测
        aims = []
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            for i in range(3):
                self.model(img, augment=self.augment)[0]
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        det = pred[0]
        im0 = img0
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                clazz = self.names[c] if not self.weights.endswith('.engine') else str(c)  # 类别
                if classes and clazz not in classes:
                    continue
                # 屏幕坐标系下, 框的 ltwh 和 xy
                sl = int(region[0] + xyxy[0])
                st = int(region[1] + xyxy[1])
                sw = int(xyxy[2] - xyxy[0])
                sh = int(xyxy[3] - xyxy[1])
                sx = int(sl + sw / 2)
                sy = int(st + sh / 2)
                # 截图坐标系下, 框的 ltwh 和 xy
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
                    plot_one_box(xyxy, im0, label=label2, color=self.colors[int(cls)], line_thickness=3)
        t3 = time.perf_counter_ns()
        # print(f'截图:{Timer.cost(t2 - t1)}, 检测:{Timer.cost(t3 - t2)}, 总计:{Timer.cost(t3 - t1)}, 数量:{len(aims)}/{len(det)}')
        return aims, img0 if image else None

    def label(self, path):
        img0 = cv2.imread(path)
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            for i in range(3):
                self.model(img, augment=self.augment)[0]
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        det = pred[0]
        result = []
        if len(det):
            im0 = img0
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
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
