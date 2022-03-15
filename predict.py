# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import platform
import sys
import time
import fcntl
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_det=False,   # 检测到入侵后保存图片
        ):

    root_dir = str(Path('../../dataset/hostData/realTimeImg/').resolve())
    source_img_path = os.path.join(root_dir, 'realTimeImg.jpg')
    save_img_path = os.path.join(root_dir, 'showRealTimeImg.jpg')
    detect_object_img_flag = False
    detect_object_img_time_name_pre = ""

    # 打开相关文件
    source_img = open(source_img_path, 'r')
    save_img = open(save_img_path, 'r')
    # Directories

    # 初始化
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # 加载模型
    w = str(weights[0] if isinstance(weights, list) else weights)
    suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    while True:
        t1 = time_sync()
        # 加载图像
        fcntl.flock(source_img, fcntl.LOCK_EX)
        im0s = cv2.imread(source_img_path)  # BGR
        fcntl.flock(source_img, fcntl.LOCK_UN)

        assert im0s is not None, f'Image Not Found {source_img_path}'
        s = 'image: '
        # Padded resize
        img = letterbox(im0s, imgsz, stride=64, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=augment, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0, frame = im0s.copy(), 0

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                detect_object_img_flag = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            else:
                detect_object_img_flag = False




            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)

            fcntl.flock(save_img, fcntl.LOCK_EX)
            cv2.imwrite(save_img_path, im0)
            fcntl.flock(save_img, fcntl.LOCK_UN)

            # Rename
            if save_det and detect_object_img_flag:
                detect_object_img_time_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".jpg"
                if detect_object_img_time_name != detect_object_img_time_name_pre:
                    detect_object_img_time_name_pre = detect_object_img_time_name
                    detect_object_img_name = os.path.join(root_dir, "detectObjectImg", detect_object_img_time_name)
                    os.system("cp " + save_img_path + " " + detect_object_img_name)
        t2 = time_sync()
        t3 = int((t2 - t1) * 1000)
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3}ms)')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'model/yolov5m6.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '../../dataset/hostData/realTimeImg/realTimeImg.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default=ROOT / '../../dataset/hostData', help='save results to project/name')
    parser.add_argument('--name', default='detectResult', help='save results to project/name')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', default=[0, 2, 3, 5, 7], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')

    parser.add_argument('--exist-ok', default=True, action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--conf-thres', type=float, default=0.75, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512, 640], help='inference size h,w')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    parser.add_argument('--save_det', default=False, action='store_true', help='save detect Img')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
