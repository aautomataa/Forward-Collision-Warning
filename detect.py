import time
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov10.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov10.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff

sys.path.insert(0, './yolov10')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height, *xyxy):
    """" 计算相对边界框的像素值。 """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def Estimated_speed(locations, fps, width):
    """
    估算车辆速度。
    :param locations: 车辆位置列表
    :param fps: 视频帧率
    :param width: 车辆宽度列表
    :return: 速度列表
    """
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])  # 获得当前帧中跟踪到车辆的ID
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])  # 获得前一帧中跟踪到车辆的ID
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:  # 将当前帧有效检测车辆的信息存入work_locations中
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:  # 将前一帧有效检测车辆的ID索引存入work_IDs_prev_index中
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:  # 将前一帧有效检测车辆的信息存入work_prev_locations中
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +  # 计算有效检测车辆的速度，采用线性的从像素距离到真实空间距离的映射
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *  # 当视频拍摄视角并不垂直于车辆移动轨迹时，测算出来的速度将比实际速度低
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]  # 将保留一位小数的单位为km/h的车辆速度及其ID存入speed二维列表中
    return speed


def detect(save_img=False):
    """
    检测函数，进行目标检测、跟踪和测速。
    :param save_img: 是否保存检测结果图像
    """
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # 是否保存推理图像
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 判断是否为网络摄像头

    # 目录设置
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 增量运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 获得视频的帧宽高
    capture = cv2.VideoCapture(source)
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # deepsort模块初始化
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    # 设置每种车型的真实车宽及人脑反应后的刹车时间
    width = [0, 0.2, 1.85, 0.5, 0, 2.3, 0, 2.5]  # bicycle、car、motorcycle、bus、truck的实际宽度，单位m
    time_person = 3   # 设置人脑反应后的刹车时间，单位为s，即从人反应后踩下刹车到车辆刹停的时间，这个时间与车辆本身的速度有关，后续可通过车机系统接口读取该速度，实现更好的预警效果
    locations = []
    speed = []
    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # 半精度仅支持CUDA

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步幅
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
    if half:
        model.half()  # 转换为FP16

    # 二阶段分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 设置数据加载器
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 设置为True以加速恒定图像大小的推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 改变显示图片大小（自定义函数）
    def cv_show(p, im0):
        height, width = im0.shape[:2]
        a = 1200 / width  # 宽为1200，计算比例
        size = (1200, int(height * a))

        img_resize = cv2.resize(im0, size, interpolation=cv2.INTER_AREA)
        cv2.imshow(p, img_resize)
        cv2.waitKey(1)  # 1毫秒

    # 运行推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次
    t0 = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # 应用NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # 应用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图像的检测结果
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 转为Path对象
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益whwh
            if len(det):
                # 将边界框从img_size缩放到im0大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每类检测结果数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # ---------------------------测速------------------------------------------#
                bbox_xywh = []
                confs = []
                classes = []
                img_h, img_w, _ = im0.shape
                # 把im0的检测结果调整至deepsort的输入数据类型
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                xywhs = torch.Tensor(bbox_xywh)  # 调用Tensor类的构造函数__init__，生成单精度浮点类型的张量
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                # 把调整好的检测结果输入deepsort，并进行跟踪
                outputs = deepsort.update(xywhs, confss, im0, classes)
                box_centers = []
                for i, each_box in enumerate(outputs):
                    # 求得每个框的中心点
                    if each_box[5] == 1 or each_box[5] == 2 or each_box[5] == 3 or each_box[5] == 5 or each_box[5] == 7:
                        box_centers.append([(each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2, each_box[4],
                                        each_box[5], each_box[2] - each_box[0]])
                location = box_centers
                # 将每帧检测出来的目标中心坐标和车辆ID写入txt中,实现轨迹跟踪
                if len(location) != 0:
                    with open('track.txt', 'a+') as track_record:
                        track_record.write('frame:%s\n' % str(frame_idx))
                        for j in range(len(location)):
                            track_record.write('id:%s,x:%s,y:%s\n' % (str(location[j][2]), str(location[j][0]), str(location[j][1])))
                    print('done!')
                locations.append(location)
                print(len(locations))
                # 每五帧写入一次测速的数据，进行测速
                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                    with open('speed.txt', 'a+') as speed_record:
                        for sp in speed:
                            speed_record.write('id:%s %skm/h\n' % (str(sp[1]), str(sp[0])))  # 将每辆车的速度写入项目根目录下的speed.txt中
                    locations = []

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    conf2 = float(f'{conf:.2f}')
                    if conf2 > 0.40:   # 置信度小于0.4时不显示检测框
                        # --------------------若检测出来的目标属于bicycle、car、motorcycle、bus、truck之一，进行测距，排除其他无关目标的干扰-------------------#
                        if names[int(cls)] == 'bicycle' or names[int(cls)] == 'car' or names[int(cls)] == 'motorcycle' or names[int(cls)] == 'bus' or names[int(cls)] == 'truck':
                            if save_txt:  # 写入文件
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # 归一化xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # 标签格式
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # 添加边界框到图像
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, speed, outputs, time_person, label=label, color=[0, 0, 255], line_thickness=3,
                                             name=names[int(cls)])  # 调用函数进行不同类别的测距，并绘制目标框



            #  打印时间（推理 + NMS）
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 显示结果
            if view_img:
                cv_show(str(p), im0)   # 该自定义函数有resize函数重构图片大小，注意不能在检测之前直接resize图像大小，会影响测距结果

            # 保存结果（带检测结果的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' 或 'stream'
                    if vid_path != save_path:  # 新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov10s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='行车记录仪视频.mp4', help='source')  #  file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')    # store_true为保存视频或者图片，路径为runs/detect
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')  # 结果视频的保存路径
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
