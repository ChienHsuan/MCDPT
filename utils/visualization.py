import cv2 as cv
import numpy as np
import os
from utils.misc import COLOR_PALETTE


def draw_detections(frame, detections, show_all_detections=True):
    """Draws detections and labels"""
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split('-')[-1]) if isinstance(label, str) else int(label)
        box_color = COLOR_PALETTE[id % len(COLOR_PALETTE)] if obj.display else (0, 0, 0)

        if show_all_detections or obj.display:
            cv.rectangle(frame, (left, top), (right, bottom), box_color, thickness=3)

        if obj.display:
            label = f'{label}' if not isinstance(label, str) else label
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            top = max(top, label_size[1])
            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def get_target_size(frame_sizes, vis=None, max_window_size=(1920, 1080), stack_frames='vertical', **kwargs):
    if vis is None:
        width = 0
        height = 0
        for size in frame_sizes:
            if width > 0 and height > 0:
                if stack_frames == 'vertical':
                    height += size[1]
                elif stack_frames == 'horizontal':
                    width += size[0]
            else:
                width, height = size
    else:
        height, width = vis.shape[:2]

    if stack_frames == 'vertical':
        target_height = max_window_size[1]
        target_ratio = target_height / height
        target_width = int(width * target_ratio)
    elif stack_frames == 'horizontal':
        target_width = max_window_size[0]
        target_ratio = target_width / width
        target_height = int(height * target_ratio)
    return target_width, target_height


def visualize_multicam_detections(frame, all_objects, fps='', show_all_detections=True,
                                  max_window_size=(1920, 1080), stack_frames='vertical', **kwargs):
    assert stack_frames in ['vertical', 'horizontal']
    vis = None

    draw_detections(frame, all_objects, show_all_detections)
    if vis is not None:
        if stack_frames == 'vertical':
            vis = np.vstack([vis, frame])
        elif stack_frames == 'horizontal':
            vis = np.hstack([vis, frame])
    else:
        vis = frame

    target_width, target_height = get_target_size(frame, vis, max_window_size, stack_frames)

    vis = cv.resize(vis, (target_width, target_height))

    label_size, base_line = cv.getTextSize(str(fps), cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv.putText(vis, str(fps), (base_line*2, base_line*3),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return vis


def draw_detection_range(frame, w_skip_ratio, h_skip_ratio):
    frame_shape = frame.shape
    left_b = int(frame_shape[1] * w_skip_ratio)
    top_b = int(frame_shape[0] * h_skip_ratio)
    right_b = int(frame_shape[1] - frame_shape[1] * w_skip_ratio)
    bottom_b = int(frame_shape[0] - frame_shape[0] * h_skip_ratio)
    cv.rectangle(frame, (left_b, top_b), (right_b, bottom_b), (0,255,0), thickness=1)
