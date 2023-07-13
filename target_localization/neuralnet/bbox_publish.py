import numpy as np
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from threading import Thread

global header
global send_box
# global receive_array
global image
header = np.zeros((13, ), dtype=np.float32)
send_box = np.zeros((4, ), np.float32)
# receive_array = 0
image = np.zeros((480, 640, 3), np.uint8)
from ultralytics import YOLO
import time
import zmq

class ServerThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.context = zmq.Context()
        self.footage_socket = self.context.socket(zmq.PUB)
        self.footage_socket.bind('tcp://192.168.0.35:5557')
        # self.footage_socket.bind('tcp://192.168.0.13:5557')
    def run(self):
        while True:
            global send_box
            # encoded = base64.b64encode(send_box).decode("ascii")
            self.footage_socket.send(send_box)
            time.sleep(1/240)


class ClientThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.context = zmq.Context()
        self.footage_socket = self.context.socket(zmq.SUB)
        self.footage_socket.connect('tcp://192.168.0.59:5556')
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.compat.unicode(''))

    def run(self):
        global image
        while True:
            t0 = time.time()
            receive_array = self.footage_socket.recv()
            # print(source.shape)
            npary = np.frombuffer(receive_array, dtype=np.uint8)
            seq = header[-4] * 255 ** 2 + header[-3] * 255 + header[-2]
            new_seq = npary[-3] * 255 ** 2 + npary[-2] * 255 + npary[-1]
            if seq != new_seq:
                image = cv2.imdecode(npary[:-12], cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for i in range(0, 12):
                    header[i] = npary[i - 12]
            print(time.time()-t0)

class PredictThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        # self.image = np.zeros((480, 640, 3), np.uint8)
        self.model = YOLO("./runs/runs_tomato/yolov3u/geocut/train/weights/best.pt")

    def run(self):

        global image
        global send_box
        global header
        while True:

            # predict
            t1 = time.time()
            results = self.model.predict(source=image, stream=False, show=False, device='0')
            header[-1] = np.float32(time.time() - t1)
            # publish bounding box
            for result in results:
                bbox = result.boxes.xyxy.cpu().detach().numpy()
                cls = result.boxes.cls.cpu().detach().numpy()
                cls = np.expand_dims(cls, axis=0)
                conf = result.boxes.conf.cpu().detach().numpy()
                conf = np.expand_dims(conf, axis=0)
                try:
                    cls_bbox = np.concatenate((cls.T, bbox, conf.T), axis=1)
                    bbox_1d = cls_bbox.flatten()
                    send_box = np.append(bbox_1d, header)
                except:
                    print("break")
                    break

if __name__ == "__main__":
    newthread1 = ClientThread()
    newthread3 = ServerThread()
    newthread2 = PredictThread()
    newthread1.start()
    newthread1.join

    newthread2.start()
    newthread2.join

    newthread3.start()
    newthread3.join