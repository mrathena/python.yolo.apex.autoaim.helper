import time
from ultralytics import YOLO
import pyautogui
import pygetwindow
import torch
import win32con
import win32gui
import win32ui
import cv2
import numpy as np

import math

import pydirectinput as direct



weights = ("yolov8n.pt")


class_mapping = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


def convert(final_aims,region):

    try:
        lst = []
        for cls,name, conf, boxes_xywh in final_aims:
            print(final_aims)
            print("================================")
            name = class_mapping[int(cls)]
            # 屏幕坐标系下, 框的 ltwh 和 框的中心点 xy
            sl = int(region[0] + boxes_xywh[0])
            st = int(region[1] + boxes_xywh[1])
            sw = int(boxes_xywh[2] - boxes_xywh[0])
            sh = int(boxes_xywh[3] - boxes_xywh[1])
            sx = int(sl + sw / 2)
            sy = int(st + sh / 2)
            # 截图坐标系下, 框的 ltwh 和 框的中心点 xy
            gl = int(boxes_xywh[0])
            gt = int(boxes_xywh[1])
            gw = int(boxes_xywh[2] - boxes_xywh[0])
            gh = int(boxes_xywh[3] - boxes_xywh[1])
            gx = int((boxes_xywh[0] + boxes_xywh[2]) / 2)
            gy = int((boxes_xywh[1] + boxes_xywh[3]) / 2)
            lst.append((int(cls), name, conf,  (sx, sy), (gx, gy), (sl, st, sw, sh), (gl, gt, gw, gh)))
            print(lst)
            return lst
    except:
        print("error")



def detect(img):
    model = YOLO(weights)
    results = model(img,verbose=False)


    aims = []

    for result in results:
        boxes_xywh = list(result.boxes.xywh.cpu().numpy().tolist())
        cls = result.boxes.cls.cpu().numpy().tolist()
        conf = result.boxes.conf.cpu().numpy().tolist()

        aimx = [(cls, conf, boxes_xywh)]
        for sublist in aimx:
            sub_output_list = []
            for i in range(len(sublist[0])):
                sub_output_list.append((sublist[0][i], sublist[1][i], sublist[2][i]))

            aims.extend(sub_output_list)

    final_aims = []
    for cls, conf, boxes_xywh in aims:

        name = class_mapping[int(cls)]
        if name == 'airplane': #当名字为person才记录在表格
           final_aims.append((int(cls),name,conf,boxes_xywh))
    print(final_aims)
    return final_aims

