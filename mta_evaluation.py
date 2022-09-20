#!/usr/bin/env python3

import argparse
import time
import queue
import threading as th
import multiprocessing as mp
import logging as log
import os
import random
import sys

import numpy as np
import cv2
import pandas as pd

from utils.network_wrappers import VectorCNN
from mc_tracker.mct import MultiCameraTracker
from sort import Sort
from deep_sort import DeepSort
from deep_sort_plus import DeepSortPlus
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from utils.video import CamCapture
from utils.visualization import visualize_multicam_detections
from utils.osutils import ensure_dir
from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611


set_log_config()


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
            has_frame, frame = self.capture.get_frame()
            if not has_frame and self.frames_queue.empty():
                self.process = False
                break
            if has_frame:
                self.frames_queue.put(frame)
        self.capture.release()


class DetectionBBox:
    def __init__(self, root):
        self.dets_file = root
        self.all_dets = self._get_all_dets()
        self._filter_dets_data()

        self.frame_num = 0
        self.idx = 0

    def _get_all_dets(self):
        return pd.read_csv(self.dets_file)

    def _filter_dets_data(self):
        self.all_dets = self.all_dets.values.tolist()

    def get_bbox(self):
        detections = []
        while True:
            if self.idx < len(self.all_dets):
                det = self.all_dets[self.idx]
                if det[0] == self.frame_num:
                    left = det[2]
                    top = det[3]
                    right = det[4]
                    bottom = det[5]
                    detections.append((left, top, right, bottom))
                    self.idx += 1
                else:
                    break
            else:
                break
        self.frame_num += 1
        
        return detections


def DetectionFromFile(args, file_name, frame_buffer, bbox_buffer, process_stat):
    capture = CamCapture([file_name[0]])
    det_bbox = DetectionBBox(file_name[1])

    thread_body = FramesThreadBody(capture, max_queue_length=2)
    frame_thread = th.Thread(target=thread_body, daemon=True)
    frame_thread.start()

    prev_frame = thread_body.frames_queue.get()

    while True:
        if not bool(process_stat.value):
            break

        if frame_buffer.qsize() < 30 and bbox_buffer.qsize() < 30:
            try:
                frames = thread_body.frames_queue.get_nowait()
            except queue.Empty:
                frames = None
                if thread_body.process:
                    continue

            all_detections = det_bbox.get_bbox()

            frame_buffer.put(prev_frame)
            bbox_buffer.put([all_detections])
        else:
            time.sleep(0.1)

        if not thread_body.process and frames is None:
            break
        prev_frame = frames

    thread_body.process = False
    process_stat.value = 0
    sys.exit(0)


def Tracking(args, file_name, config, frame_buffer, bbox_buffer, process_stat):
    ie_feat = IECore()
    avg_latency = AverageEstimator()

    if args.method != 'sort':
        object_recognizer = VectorCNN(ie_feat, args.m_reid, args.device2, args.cpu_extension)

    config['cam_id']['id'] = int(file_name[1][-5])
    config['sct_config']['initial_id'] = args.initial_id

    if args.method == 'mtmct':
        tracker = MultiCameraTracker(object_recognizer, config['cam_id']['id'],
                                    config['sct_config'], **config['mct_config'],
                                    broker_url=args.broker_url)
    elif args.method == 'sort':
        tracker = Sort()
    elif args.method == 'deepsort':
        tracker = DeepSort(object_recognizer, **config['mct_config'])
    elif args.method == 'deepsortplus':
        tracker = DeepSortPlus(object_recognizer, config['cam_id']['id'],
                               config['sct_config'], **config['mct_config'],
                               broker_url=args.broker_url)
    else:
        raise NameError(f'Not supported method: {args.method}.')

    if len(args.output_video):
        frame_size, fps = config['visualization_config']['max_window_size'], config['visualization_config']['out_fps']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ensure_dir(args.logs_dir)
        output_video_path = os.path.join(args.logs_dir, args.output_video)
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    else:
        output_video = None

    if not args.no_txt_output:
        out_file = f"track_results_{config['cam_id']['id']}.txt"
        ensure_dir(args.logs_dir)
        out_file = os.path.join(args.logs_dir, out_file)
        with open(out_file, 'w') as file:
            file.write('frame_no_cam,cam_id,person_id,detection_idx,xtl,ytl,xbr,ybr\n')

    key = -1
    empty_times = 0
    while True:
        if not args.no_show:
            key = check_pressed_keys(key)
            if key == 27:
                process_stat.value = 0

        start = time.perf_counter()
        try:
            frames = frame_buffer.get_nowait()
            all_detections = bbox_buffer.get_nowait()
        except queue.Empty:
            frames = None
            all_detections = None
            empty_times += 1
            if bool(process_stat.value) or not frame_buffer.empty():
                continue
            else:
                if empty_times > 5:
                    break
                else:
                    continue
        empty_times = 0

        if args.method == 'sort':
            tracked_objects = tracker.process(all_detections)
        else:
            tracker.process(frames, all_detections)
            tracked_objects = tracker.get_tracked_objects()

        latency = max(time.perf_counter() - start, sys.float_info.epsilon)

        avg_latency.update(latency)
        fps = round(1. / latency)

        if not args.no_txt_output:
            with open(out_file, 'a') as file:
                for obj in tracked_objects:
                    left, top, right, bottom = obj.rect
                    cam_id = config['cam_id']['id']

                    if args.method == 'mtmct' or args.method == 'deepsortplus':
                        label = int(obj.label.split('-')[-1])
                    elif args.method == 'deepsort' or args.method == 'sort':
                        label = int(obj.label)
                    else:
                        raise NameError(f'Not supported method: {args.method}.')

                    file.write(f'{tracker.time-1},{cam_id},{label},{0},{left},{top},{right},{bottom}\n')

        vis = visualize_multicam_detections(frames, tracked_objects, fps,
                                            **config['visualization_config'])
        if not args.no_show:
            cv2.imshow('Output', vis)
        if output_video:
            output_video.write(cv2.resize(vis, frame_size))

        print(f'\rfps = {fps} (avg_fps = {round(1. / avg_latency.get_avg(), 1)})',
              end="")
        
        if args.method == 'deepsortplus' or args.method == 'mtmct':
            # to synchronize multi camera devices
            if tracker.sync_multi_cams:
                tracker.check_processed_frame_num()

    if args.method == 'mtmct' or args.method == 'deepsortplus':
        tracker.mqtt_client.end()
    cv2.destroyAllWindows()
    sys.exit(0)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    """MTA evaluation"""
    parser = argparse.ArgumentParser(description='Multi camera multi object \
                                                  tracking live demo script')
    parser.add_argument('-i', type=str, help='MTA_ext_short path', required=True)
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')
    parser.add_argument('-m', '--m_detector', type=str, required=False,
                        help='Path to the object detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the object detection model')
    parser.add_argument('--m_reid', type=str, required=True,
                        help='Path to the object re-identification model')
    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument('--history_file', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save results of the demo')
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')
    parser.add_argument('--no_txt_output', help="Optional. Don't write txt output", action='store_true')
    parser.add_argument('--device1', type=str, default='CPU')
    parser.add_argument('--device2', type=str, default='CPU')
    parser.add_argument('--broker_url', type=str, required=True, help='MQTT broker url')
    parser.add_argument('--method', type=str, default='mtmct')
    parser.add_argument('--cam_id', type=int, nargs='+', help='MTA_ext_short cam id', required=True)
    parser.add_argument('--initial_id', type=int, default=1, help='Initial ID of the first track')
    parser.add_argument('--logs_dir', type=str, help='Output logs dir', required=True)
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.',
                             type=str, default=None)
    args = parser.parse_args()

    if len(args.config):
        log.info('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error('No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    log.info("Creating Inference Engine")

    for cam_id in args.cam_id:
        file_paths = []
        video_path = os.path.join(args.i, f'test/cam_{cam_id}/cam_{cam_id}.mp4')
        annotation_path = os.path.join(args.i, f'test/cam_{cam_id}/coords_fib_cam_{cam_id}.csv')
        file_paths.append(video_path)
        file_paths.append(annotation_path)

        frame_buffer = mp.Queue()
        bbox_buffer = mp.Queue()
        process_stat = mp.Value('i', 1)

        detection = mp.Process(target=DetectionFromFile, args=(args, file_paths, frame_buffer, bbox_buffer, process_stat), daemon=True)
        tracking = mp.Process(target=Tracking, args=(args, file_paths, config, frame_buffer, bbox_buffer, process_stat), daemon=True)

        detection.start()
        tracking.start()

        detection.join()
        tracking.join()
        detection.terminate()
        tracking.terminate()
        print(f'Cam {cam_id} finished')

    log.info('Demo finished successfully')
    sys.exit(0)


if __name__ == '__main__':
    main()
