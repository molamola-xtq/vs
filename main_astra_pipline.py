import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

from openni import openni2
from pyorbbecsdk import *
# from utils import frame_to_bgr_image
import math
import yaml
import argparse
import os
import time
import numpy as np
import sys

import cv2

# class_names=[]

class YoloV5:
    def __init__(self, yolov5_yaml_path='/home/wildman/yolov5-astra-main/config/yolov5s.yaml'):
        global class_names
        '''初始化'''
        # 载入配置文件
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # class_names = self.yolov5['class_name']
        # 随机生成每个类别的颜色
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        # 模型初始化
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        # 设置日志输出
        set_logging()
        # 选择计算设备
        device = select_device(self.yolov5['device'])
        # 如果是GPU则使用半精度浮点数 F16
        is_half = device.type != 'cpu'
        # 载入模型
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  # 检查模型的尺寸
        if is_half:
            model.half()  # 将模型转换为半精度
        # 设置BenchMark，加速固定图像的尺寸的推理
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 图像缓冲区初始化
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # 创建模型
        # run once
        _ = model(img_torch.half()
                  if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half  # 是否开启半精度
        self.device = device  # 计算设备
        self.model = model  # Yolov5模型
        self.img_torch = img_torch  # 图像缓冲区

    def preprocessing(self, img):
        '''图像预处理'''
        # 图像缩放
        # 注: auto一定要设置为False -> 图像的宽高不同
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        # print("img resize shape: {}".format(img_resize.shape))
        # 增加一个维度
        img_arr = np.stack([img_resize], 0)
        # 图像转换 (Convert) BGR格式转换为RGB
        # 转换为 bs x 3 x 416 x
        # 0(图像i), 1(row行), 2(列), 3(RGB三通道)
        # ---> 0, 3, 1, 2
        # BGR to RGB, to bsx3x416x416
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # 数值归一化
        # img_arr =  img_arr.astype(np.float32) / 255.0
        # 将数组在内存的存放地址变成连续的(一维)， 行优先
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # https://zhuanlan.zhihu.com/p/59767914
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, depth_data):
        # 图像预处理
        img_resize = self.preprocessing(img)  # 图像缩放
        self.img_torch = torch.from_numpy(img_resize).to(self.device)  # 图像格式转换
        self.img_torch = self.img_torch.half() if self.is_half else self.img_torch.float()  # 格式转换 uint8-> 浮点数
        self.img_torch /= 255.0  # 图像归一化
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)

        # 模型推理
        pred = self.model(self.img_torch, augment=False)[0]

        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)

        # 初始化画布
        canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []

        det = pred[0]
        if det is not None and len(det):
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_coords(img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf.item())  # 将tensor转为标准Python数值
                class_id_list.append(class_id)

                # 获取中心点深度
                dist = get_mid_pos(img, xyxy, depth_data, 24)  # 假设你有一个函数来计算中心点的深度

                # 绘制矩形框与标签，包括深度信息
                label = f'{self.yolov5["class_name"][class_id]} {conf:.2f} {dist / 1000:.2f}m'
                self.plot_one_box(xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)

        return canvas, class_id_list, xyxy_list, conf_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        '''绘制矩形框+标签'''
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_mid_pos(frame,box,depth_data,randnum):
#这个函数就是简单的从给出的图片、框、深度数据、可以选择的迭代次数
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置左上角和右下角相加在/2
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)

        row_idx = int(mid_pos[1] + bias)  
        col_idx = int(mid_pos[0] + bias)  
        if 0 <= row_idx < depth_data.shape[0] and 0 <= col_idx < depth_data.shape[1]:  
            dist = depth_data[row_idx, col_idx]  
            cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        else:
            continue
        # dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        # cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    return np.mean(distance_list)


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

if __name__ == "__main__":
    MIN_DEPTH = 20  # 20mm
    MAX_DEPTH = 10000  # 10000mm

    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
        except OBError as e:
            print("Error: ", e)
            depth_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(depth_profile)
        pipeline.start(config)

        cap = cv2.VideoCapture(2)
        print("[INFO] YoloV5 Objective Detection - Program Start")
        print("[INFO] Loading YoloV5 Model")
        model = YoloV5(yolov5_yaml_path='/home/wildman/yolov5-astra-main/config/yolov5s.yaml')
        print("[INFO] YoloV5 Model Loaded")

        while True:
            frames = pipeline.wait_for_frames(10)
            depth_frame = frames.get_depth_frame()
            width = depth_profile.get_width()
            height = depth_profile.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))

            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            # Apply temporal filtering
            depth_data = temporal_filter.process(depth_data)


            # dpt = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dpt = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth', dpt)

            ret, color_frame = cap.read()

            # YoloV5 Object Detection
            canvas, class_id_list, xyxy_list, conf_list = model.detect(color_frame, depth_data)

            # 显示检测后的图像
            cv2.imshow('Detected Image', canvas)  # 显示处理后的图像

            # 组合深度图和原图显示（可选）
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(dpt, alpha=0.03), cv2.COLORMAP_JET)

            depth_scale_width = int(depth_colormap.shape[1] * (480 / depth_colormap.shape[0]))  
            depth_colormap_resized = cv2.resize(depth_colormap, (depth_scale_width, 480), interpolation=cv2.INTER_LINEAR)  
            images = np.hstack((color_frame, depth_colormap_resized))

            # images = np.hstack((color_frame, depth_colormap))
            cv2.imshow('RealSense', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()



