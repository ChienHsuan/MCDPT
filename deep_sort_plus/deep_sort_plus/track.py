from enum import IntEnum

import numpy as np


THE_BIGGEST_DISTANCE = 2.

def euclidean_distance(x, y, squared=False, return_min_value=True):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x.shape) == 1 and len(y.shape) == 1:
        if squared:
            return np.sum((x-y)**2)
        else:
            return np.sqrt(np.sum((x-y)**2))
    else:
        xx = (x*x).sum(axis=1)[:, np.newaxis]
        yy = (y*y).sum(axis=1)[np.newaxis, :]
        squared_dist = xx + yy - 2 * x @ y.T
        squared_dist = np.maximum(squared_dist, 0)
        if return_min_value:
            squared_dist = np.amin(squared_dist)
        if squared:
            return squared_dist
        else:
            return np.sqrt(squared_dist)

def cosine_distance(a, b, data_is_normalized=False, return_min_value=True):
    if len(a.shape) == 1:
        a = a[np.newaxis, :]
    if len(b.shape) == 1:
        b = b[np.newaxis, :]

    if not data_is_normalized:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)

    if return_min_value:
        distance_matrix = 1. - np.dot(a, b.T)
        return np.amin(distance_matrix)
    else:
        return 1. - np.dot(a, b.T)


class TrackState(IntEnum):
    """
    Enumeration type for the single target track state. Newly Started tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class ClusterFeature:
    def __init__(self, feature_len, init_dis_thres=0.1):
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        self.init_dis_thres = init_dis_thres
        self.global_merge_weight = 0.2

    def update(self, feature_vec, num=1):
        if len(self.clusters) == 0:
            self.clusters.append(feature_vec)
            self.clusters_sizes.append(num)
        elif len(self.clusters) < self.feature_len:
            distances = cosine_distance(feature_vec.reshape(1, -1),
                                        np.array(self.clusters).reshape(len(self.clusters), -1),
                                        return_min_value=False)
            if np.amin(distances) > self.init_dis_thres:
                self.clusters.append(feature_vec)
                self.clusters_sizes.append(num)
            else:
                nearest_idx = np.argmin(distances)
                self.clusters_sizes[nearest_idx] += num
                self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) * num / \
                                              self.clusters_sizes[nearest_idx]

        else:
            distances = cosine_distance(feature_vec.reshape(1, -1),
                                        np.array(self.clusters).reshape(len(self.clusters), -1),
                                        return_min_value=False)
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += num
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) * num / \
                                           self.clusters_sizes[nearest_idx]

    def merge(self, other):
        for i, feature in enumerate(other.clusters):
            self.update(feature, other.clusters_sizes[i])

    def global_merge(self, global_feats):
        distances = cosine_distance(global_feats,
                                    np.array(self.clusters).reshape(len(self.clusters), -1),
                                    return_min_value=False)
        for i, feat in enumerate(global_feats):
            if len(self.clusters) < self.feature_len:
                if np.amin(distances[i]) > self.init_dis_thres:
                    self.clusters.append(feat)
                    self.clusters_sizes.append(1)
                else:
                    nearest_idx = np.argmin(distances[i])
                    self.clusters[nearest_idx] = self.global_merge_weight*feat \
                                                 + (1-self.global_merge_weight)*self.clusters[nearest_idx]
            else:
                nearest_idx = np.argmin(distances[i])
                self.clusters[nearest_idx] = self.global_merge_weight*feat \
                                             + (1-self.global_merge_weight)*self.clusters[nearest_idx]

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


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


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.

    """

    def __init__(self, mean, covariance, track_id, cam_id, n_init, max_age,
                 detection, start_time, budget=100, num_clusters=4, clust_init_dis_thresh=0.1,
                 stable_time_thresh=15, rectify_length_thresh=2):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.cam_id = cam_id
        self.trajectory = cam_id  # current camera ID
        self.start_time = start_time
        self.end_time = start_time
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.counts = 0

        self.state = TrackState.Tentative
        self.f_queue = []
        self.f_avg = AverageEstimator()  # average feature
        self.f_clust = ClusterFeature(num_clusters, init_dis_thres=clust_init_dis_thresh)  # cluster feature
        if detection.feature is not None:
            self.f_queue.append(detection.feature)
        self.current_detection = detection

        self.off = False
        self.feats_delivery_status = False
        self.pos_delivery_status = False
        self.cross_camera_track = False

        self.last_merge_dis = 1.
        self.stable_time_thresh = stable_time_thresh
        self.rectify_length_thresh=rectify_length_thresh

        self._n_init = n_init
        self._max_age = max_age
        self.budget = budget

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if self.time_since_update <= self._max_age:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, time, is_occluded):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        if detection.feature is not None:
            if self.is_confirmed() and not is_occluded:
                self.f_clust.update(detection.feature)
                self.f_avg.update(detection.feature)
                self.enqueue_dequeue(detection.feature)
            else:
                self.enqueue_dequeue(detection.feature)
        self.current_detection = detection

        self.end_time = time
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def enqueue_dequeue(self, feature):
        self.f_queue.append(feature)
        self.f_queue = self.f_queue[-self.budget:]

    def merge(self, other, dist):
        assert self.end_time < other.start_time

        self.mean = other.mean
        self.covariance = other.covariance
        self.end_time = other.end_time
        self.hits += other.hits
        self.age += other.age
        self.time_since_update = other.time_since_update
        self.f_queue = other.f_queue
        self.f_avg.merge(other.f_avg)
        self.f_clust.merge(other.f_clust)
        self.current_detection = other.current_detection
        self.last_merge_dis = dist

    def global_merge(self, track, dist):
        self.f_clust.global_merge(track.f_clust)
        if self.cross_camera_track:
            if track.start_time < self.start_time:
                self.cam_id = track.cam_id
                self.track_id = track.id
                self.start_time = track.start_time
        else:
            self.cam_id = track.cam_id
            self.track_id = track.id
            self.start_time = track.start_time
        self.cross_camera_track = True
        self.pos_delivery_status = True
        self.last_merge_dis = dist

    def mark_missed(self, length):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        if self.time_since_update >= self._max_age and len(self.f_avg) < length:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_stable(self):
        return self.counts >= self.stable_time_thresh \
               and len(self.f_avg) >= self.rectify_length_thresh
