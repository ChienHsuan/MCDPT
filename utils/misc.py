import cv2 as cv
import numpy as np
import logging as log
import os.path as osp
import sys
from importlib import import_module


class AverageEstimator(object):
    def __init__(self):
        self.val = 0  # previous feature
        self.avg = 0  # average feature
        self.count = 0

    def __len__(self):
        return self.count

    def update(self, val):
        self.val = val
        self.count += 1
        self.avg += (self.val - self.avg) / self.count

    def is_valid(self):
        return self.count > 0

    def merge(self, other):
        self.val = other.val
        self.count += other.count
        self.avg += (other.avg - self.avg) * other.count / self.count

    def get_avg(self):
        return self.avg


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def read_py_config(filename):
    filename = osp.abspath(osp.expanduser(filename))
    check_file_exist(filename)
    assert filename.endswith('.py')
    module_name = osp.basename(filename)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = osp.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }

    return cfg_dict


def check_pressed_keys(key):
    if key == 32:  # Pause
        while True:
            key = cv.waitKey(0)
            if key == 27 or key == 32 or key == 13:  # enter: resume, space: next frame, esc: exit
                break
    else:
        key = cv.waitKey(1)
    return key


def set_log_config():
    log.basicConfig(stream=sys.stdout, format='%(levelname)s: %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=log.DEBUG)


COLOR_PALETTE = [[0, 113, 188],
                 [216, 82, 24],
                 [236, 176, 31],
                 [125, 46, 141],
                 [118, 171, 47],
                 [76, 189, 237],
                 [161, 19, 46],
                 [76, 76, 76],
                 [153, 153, 153],
                 [255, 0, 0],
                 [255, 127, 0],
                 [190, 190, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [170, 0, 255],
                 [84, 84, 0],
                 [84, 170, 0],
                 [84, 255, 0],
                 [170, 84, 0],
                 [170, 170, 0],
                 [170, 255, 0],
                 [255, 84, 0],
                 [255, 170, 0],
                 [255, 255, 0],
                 [0, 84, 127],
                 [0, 170, 127],
                 [0, 255, 127],
                 [84, 0, 127],
                 [84, 84, 127],
                 [84, 170, 127],
                 [84, 255, 127],
                 [170, 0, 127],
                 [170, 84, 127],
                 [170, 170, 127],
                 [170, 255, 127],
                 [255, 0, 127],
                 [255, 84, 127],
                 [255, 170, 127],
                 [255, 255, 127],
                 [0, 84, 255],
                 [0, 170, 255],
                 [0, 255, 255],
                 [84, 0, 255],
                 [84, 84, 255],
                 [84, 170, 255],
                 [84, 255, 255],
                 [170, 0, 255],
                 [170, 84, 255],
                 [170, 170, 255],
                 [170, 255, 255],
                 [255, 0, 255],
                 [255, 84, 255],
                 [255, 170, 255],
                 [42, 0, 0],
                 [84, 0, 0],
                 [127, 0, 0],
                 [170, 0, 0],
                 [212, 0, 0],
                 [255, 0, 0],
                 [0, 42, 0],
                 [0, 84, 0],
                 [0, 127, 0],
                 [0, 170, 0],
                 [0, 212, 0],
                 [0, 255, 0],
                 [0, 0, 42],
                 [0, 0, 84],
                 [0, 0, 127],
                 [0, 0, 170],
                 [0, 0, 212],
                 [0, 0, 255],
                 [0, 0, 0],
                 [36, 36, 36],
                 [72, 72, 72],
                 [109, 109, 109],
                 [145, 145, 145],
                 [182, 182, 182],
                 [218, 218, 218],
                 [255, 255, 255]]