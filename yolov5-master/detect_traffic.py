import json
import redis
import cv2
import argparse
from PIL import Image
from utils.datasets import *
from utils.utils import *


# best_GQ_Classify_lastest
class detector:
    def __init__(self, weights='/media/yons/DATA/PycharmProjects/yolov5-master/weights/traffic_5l.pt'):
        self.weights = weights
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size = 640
        # 参数
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
        self.half = True
        self.view_img = True
        self.model = torch.load(self.weights, map_location=self.device)['model']
        self.model.to(self.device).eval()
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names

    def detect(self, source):
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        # Initialize
        # Load model
        google_utils.attempt_download(self.weights)

        # Half precision
        half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Set Dataloader
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.size)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=self.size)

        # Get names and colors
        names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference

        img = torch.zeros((1, 3, self.size, self.size), device=self.device)  # init img
        _ = self.model(img.half() if half else img.float()) if self.device.type != 'cpu' else None  # run once
        # print('dataset', dataset)
        for path, img, im0s, vid_cap in dataset:
            # count = 0
            # print(np.array(im0s).shape)
            h, w, d = np.array(im0s).shape
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            # t1 = torch_utils.time_synchronized()
            pred = self.model(img)[0]
            # to float
            if half:
                pred = pred.float()
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, fast=True)
            real_pred = []
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s
                # save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for i in det:
                        real_pred.append(i.tolist())
            if real_pred:
                labelData = {}
                annotation = []
                for b in real_pred:
                    cls = b[5]
                    anno = {"bbox": [b[0], b[1], b[2], b[3]], "class": '%s' % (names[int(cls)])}
                    annotation.append(anno)
                # time_stamp = ctime
                num = len(real_pred)
                width = w
                height = h
                labelData["annotation"] = annotation
                # labelData["time_stamp"] = time_stamp
                labelData["num"] = num
                labelData["width"] = width
                labelData["height"] = height
                jsonData = json.dumps(labelData)
                r = redis.Redis(connection_pool=self.pool)
                r.lpush("label", jsonData)
                # print('path', path)
                print(jsonData)
                return jsonData


if __name__ == '__main__':
    pic = '/media/yons/DATA/PycharmProjects/License-Plate-Recognition-master/back/UYC995/0.jpg'
    # im = cv2.imread(pic)
    detect = detector()
    pred = detect.detect(pic)
    print(pred)
