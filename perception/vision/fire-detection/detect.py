# -*- coding: utf-8 -*-
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from temporal.tracker import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        temporal=None,
        area_thresh=0.05,
        window_size=20,
        task='test',
        persistence_thresh=0.5):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')

    # 🔹 Diretório base
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 NOVO: pasta para fire bounding boxes
    fire_dir = save_dir / "fire_labels"
    fire_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 Modelo
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    print("CLASSES:", names)  # IMPORTANTE PARA CONFIRMAR FIRE ID

    fire_class_id = 1

    # 🔹 Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=model.pt)

    model.warmup(imgsz=(1, 3, *imgsz), half=half)

    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):

            p = Path(path)
            im0 = im0s.copy()

            # 🔥 ficheiro por imagem
            txt_fire_path = fire_dir / f"{p.stem}.txt"

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):

                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 🔥 limpa ficheiro antes de escrever
                open(txt_fire_path, 'w').close()

                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)

                    # =========================
                    # 🔥 GUARDAR APENAS FIRE
                    # =========================
                    if c == fire_class_id:

                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        conf_val = float(conf)

                        with open(txt_fire_path, 'a') as f:
                            f.write(f"{xmin} {ymin} {xmax} {ymax} {conf_val}\n")

                    # 🔹 (mantém desenho normal)
                    if save_img or view_img:
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # 🔹 Mostrar
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 🔹 Guardar imagem (opcional)
            if save_img:
                cv2.imwrite(str(save_dir / p.name), im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--temporal', type=str, help='temporal analysis technique used after detection')
    parser.add_argument('--area-thresh', type=float, default = 0.05, help='suppression threshold when temporal analysis technique is tracker')
    parser.add_argument('--window-size', type=int, default = 20, help='sliding window size for temporal analysis technique')
    parser.add_argument('--task', type=str, default = 'test', help='perform validation or test')
    parser.add_argument('--persistence-thresh', type=float, default = 0.50, help='suppression threshold when temporal analysis technique is persistence')
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
