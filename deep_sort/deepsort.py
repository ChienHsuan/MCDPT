from collections import namedtuple

import numpy as np

from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as dsTracker

TrackedObj = namedtuple('TrackedObj', ['rect', 'label', 'display'])


class DeepSort(object):

    def __init__(self, reid_model, time_window=10, match_threshold=0.25,
                 bbox_min_aspect_ratio=1.2, bbox_max_aspect_ratio=6,
                 w_skip_ratio=0.1, h_skip_ratio=0.125, ignore_edge_objects=False, **kwargs):
        self.reid_model = reid_model
        self.w_skip_ratio = w_skip_ratio
        self.h_skip_ratio = h_skip_ratio
        self.ignore_edge_objects = ignore_edge_objects
        self.bbox_min_aspect_ratio = bbox_min_aspect_ratio
        self.bbox_max_aspect_ratio = bbox_max_aspect_ratio
        self.time_window = time_window
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", match_threshold, 100)
        self.tracker = dsTracker(metric)
        self.finished_tracks = []
        self.time = 0

    def group_instances(self, detections, features):
        grouped_instances = []
        for obj_i in range(len(detections)):
            bbox = detections[obj_i]
            feature = features[obj_i].reshape(-1)
            detection = Detection(bbox, 1.0, feature, obj_i)
            grouped_instances.append(detection)
        return grouped_instances

    def get_tracked_objects(self):
        objs = []
        for track in self.tracker.tracks:
            if track.time_since_update > 0:
                continue

            objs.append(TrackedObj(track.to_tlbr().astype(int),
                                   str(track.track_id),
                                   track.hits > self.time_window
                                  )
                        )

        for track in self.tracker.deleted_tracks:
            self.finished_tracks.append(track.track_id)

        return objs

    def _filter_detections(self, detections, frame_shape):
        clean_detections = []
        screen_edge_objects = []

        left_b = int(frame_shape[1] * self.w_skip_ratio)
        top_b = int(frame_shape[0] * self.h_skip_ratio)
        right_b = int(frame_shape[1] - frame_shape[1] * self.w_skip_ratio)
        bottom_b = int(frame_shape[0] - frame_shape[0] * self.h_skip_ratio)
        boundary_coord = [left_b, top_b, right_b, bottom_b]

        for det in detections:
            if det[0] >=0 and det[1] >= 0 and det[2] < frame_shape[1] and det[3] < frame_shape[0]:
                w = det[2] - det[0]
                h = det[3] - det[1]
                ar = h / w
                if ar > self.bbox_min_aspect_ratio and ar < self.bbox_max_aspect_ratio:
                    center_x = int((det[2] + det[0]) / 2)
                    center_y = int((det[3] + det[1]) / 2)
                    if center_x < left_b or center_x > right_b or center_y < top_b \
                        or center_y > bottom_b:
                        if self.ignore_edge_objects:
                            pass
                        else:
                            clean_detections.append(det)
                            screen_edge_objects.append(True)
                    else:
                        clean_detections.append(det)
                        screen_edge_objects.append(False)

        return clean_detections, screen_edge_objects, boundary_coord

    def get_embeddings(self, frame, detections):
        rois = []
        embeddings = []

        for rect in detections:
            left, top, right, bottom = rect
            crop = frame[top:bottom, left:right]
            rois.append(crop)

        if rois:
            embeddings = self.reid_model.forward(rois)
            assert len(rois) == len(embeddings)

        return embeddings

    def process(self, frame, all_detections, return_track_object=True):
        all_detections, _, _ = self._filter_detections(all_detections[0], frame.shape)
        reid_features = self.get_embeddings(frame, all_detections)
        grouped_instances = self.group_instances(all_detections, reid_features)

        self.tracker.predict()
        self.tracker.update(grouped_instances)
        self.time += 1

        if return_track_object:
            objs = self.get_tracked_objects()
            return objs
        
        return
