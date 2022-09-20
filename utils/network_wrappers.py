import json
import logging as log
from collections import namedtuple
from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils.ie_tools import load_ie_model


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frame, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, trg_classes, conf=.6,
                 device='CPU', ext_path='', max_num_frames=1):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames)
        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def run(self, frame, only_target_class=True):
        outputs = self.net.forward(frame)

        all_detections = []
        detections = self.__decode_detections(outputs, frame.shape, only_target_class)
        all_detections.append(detections)
        return all_detections

    def run_async(self, frame):
        self.shapes = []

        self.shapes.append(frame.shape)
        self.net.forward_async(frame)

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.net.grab_all_async()

        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frame):
        """Returns all detections on frame"""
        self.run_async(frame)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, only_target_class):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            if only_target_class and detection[1] not in self.trg_classes:
                continue

            confidence = detection[2]
            if confidence < self.confidence:
                continue

            left = int(max(detection[3], 0) * frame_shape[1])
            top = int(max(detection[4], 0) * frame_shape[0])
            right = int(max(detection[5], 0) * frame_shape[1])
            bottom = int(max(detection[6], 0) * frame_shape[0])
            if self.expand_ratio != (1., 1.):
                w = (right - left)
                h = (bottom - top)
                dw = w * (self.expand_ratio[0] - 1.) / 2
                dh = h * (self.expand_ratio[1] - 1.) / 2
                left = max(int(left - dw), 0)
                right = int(right + dw)
                top = max(int(top - dh), 0)
                bottom = int(bottom + dh)

            detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


class VectorCNN:
    """Wrapper class for a network returning a vector"""

    def __init__(self, ie, model_path, device='CPU', ext_path='', max_reqs=100):
        self.max_reqs = max_reqs
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=self.max_reqs)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)
        outputs = self.net.grab_all_async()
        return outputs

    def forward_async(self, batch):
        """Performs async forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            self.net.forward_async(frame)

    def wait_and_grab(self):
        outputs = self.net.grab_all_async()
        return outputs
