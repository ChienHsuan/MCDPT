import numpy as np

from . import iou_matching, kalman_filter, linear_assignment
from .track import Track, cosine_distance, THE_BIGGEST_DISTANCE


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, cam_id, initial_id=1, max_iou_distance=0.7, max_age=30, n_init=3,
                 budget=100, num_clusters=4, clust_init_dis_thresh=0.1, continue_time_thresh=2,
                 time_window=10, stable_time_thresh=15, detection_occlusion_thresh=0.5,
                 rectify_time_thresh=10, rectify_length_thresh=4, rectify_thresh=0.25,
                 merge_thresh=0.25, gap=50):
        self.metric = metric
        self.cam_id = cam_id
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.budget = budget
        self.num_clusters = num_clusters
        self.clust_init_dis_thresh = clust_init_dis_thresh
        self.continue_time_thresh = continue_time_thresh
        self.time_window = time_window
        self.stable_time_thresh = stable_time_thresh
        self.detection_occlusion_thresh = detection_occlusion_thresh
        self.rectify_time_thresh = rectify_time_thresh
        self.rectify_length_thresh = rectify_length_thresh
        self.rectify_thresh = rectify_thresh
        self.merge_thresh = merge_thresh
        self.gap = gap

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = initial_id

        self.time = 0

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, screen_edge_objects, boundary_coord):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Check for the occluded and screen edge objects
        occluded_objects = np.full(len(detections), False)
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if (i != j and self._ios(det1, det2) > self.detection_occlusion_thresh) \
                    or screen_edge_objects[i]:
                    occluded_objects[i] = True
                    break

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], self.time, occluded_objects[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(self.rectify_length_thresh)
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.deleted_tracks = [t for t in self.tracks if t.is_deleted()]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Rectify tracks
        self._rectify_tracks()

        self._check_tracks_state(boundary_coord)

        # Merge tracks
        if self.time % self.time_window == 0:
            self._merge_tracks()

        self.time += 1

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            cost_matrix = self.metric.distance(tracks, features, track_indices)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) \
                if t.is_confirmed() and t.time_since_update <= self.max_age \
                and t.trajectory == self.cam_id]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) \
                if not t.is_confirmed() and t.trajectory == self.cam_id]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.cam_id, self.n_init, self.max_age,
            detection, self.time, budget=self.budget, num_clusters=self.num_clusters,
            clust_init_dis_thresh=self.clust_init_dis_thresh,
            stable_time_thresh=self.stable_time_thresh,
            rectify_length_thresh=self.rectify_length_thresh))
        self._next_id += 1

    def _rectify_tracks(self):
        active_tracks_idx = []
        non_active_tracks_idx = []

        # divided into active and non-active track
        for i, track in enumerate(self.tracks):
            if track.time_since_update <= self.rectify_time_thresh \
               and len(track.f_avg) >= self.rectify_length_thresh \
               and track.trajectory == self.cam_id:
                active_tracks_idx.append(i)
            elif len(track.f_avg) >= self.rectify_length_thresh and track.trajectory == self.cam_id:
                non_active_tracks_idx.append(i)

        distance_matrix = np.zeros((len(active_tracks_idx),
                                    len(non_active_tracks_idx)), dtype=np.float32)
        for i, idx1 in enumerate(active_tracks_idx):
            for j, idx2 in enumerate(non_active_tracks_idx):
                distance_matrix[i, j] = self._get_merge_distance(self.tracks[idx1], self.tracks[idx2])

        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.rectify_thresh:
                self._concatenate_tracks(active_tracks_idx[indices_rows[i]],
                                         non_active_tracks_idx[indices_cols[j]], dist)
                distance_matrix = np.delete(distance_matrix, i, 0)
                indices_rows = np.delete(indices_rows, i)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_cols = np.delete(indices_cols, j)
            else:
                break
        self.tracks = list(filter(None, self.tracks))

    def _check_tracks_state(self, boundary_coord):
        for track in self.tracks:
            if track.is_confirmed():
                if track.time_since_update <= self.continue_time_thresh:
                    track.counts += 1
                else:
                    track.counts = 0
                
                # check whether track left the camera or not
                track_last_box = track.current_detection.to_tlbr()
                if not track.off and (track.time_since_update > 2*self.time_window \
                    or (track.time_since_update > self.time_window // 2 and \
                    (track_last_box[0] - boundary_coord[0] <= self.gap or \
                    track_last_box[1] - boundary_coord[1] <= self.gap or \
                    boundary_coord[2] - track_last_box[2] <= self.gap or \
                    boundary_coord[3] - track_last_box[3] <= self.gap))):
                    track.off = True

                # check delivery and off status
                if track.is_stable():
                    if not track.feats_delivery_status:
                        track.feats_delivery_status = True
                    if track.off:
                        track.off = False
                else:
                    if not track.pos_delivery_status and track.cross_camera_track and track.off:
                        track.pos_delivery_status = True

    def _merge_tracks(self):
        distance_matrix, tracks_indices = self._get_merge_distance_matrix()

        while len(tracks_indices) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                kept_idx = self._concatenate_tracks(tracks_indices[i], tracks_indices[j], dist)
                deleted_idx = tracks_indices[i] if kept_idx == tracks_indices[j] else tracks_indices[j]
                assert self.tracks[deleted_idx] is None
                if deleted_idx == tracks_indices[i]:
                    idx_to_delete = i
                    idx_to_update = j
                else:
                    assert deleted_idx == tracks_indices[j]
                    idx_to_delete = j
                    idx_to_update = i
                updated_row = self._get_updated_merge_distance_matrix_row(kept_idx,
                                                                          deleted_idx,
                                                                          tracks_indices)
                distance_matrix[idx_to_update, :] = updated_row
                distance_matrix[:, idx_to_update] = updated_row
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 0)
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 1)
                tracks_indices = np.delete(tracks_indices, idx_to_delete)
            else:
                break

        self.tracks = list(filter(None, self.tracks))

    def _get_merge_distance_matrix(self):
        tracks_indices = []
        for i, track in enumerate(self.tracks):
            if track.trajectory == self.cam_id and len(track.f_avg) >= self.rectify_length_thresh:
                tracks_indices.append(i)

        distance_matrix = THE_BIGGEST_DISTANCE * np.eye(len(tracks_indices), dtype=np.float32)
        for i, idx_t1 in enumerate(tracks_indices):
            for j, idx_t2 in enumerate(tracks_indices):
                if i < j:
                    distance_matrix[i, j] = self._get_merge_distance(self.tracks[idx_t1], self.tracks[idx_t2])
        distance_matrix += np.transpose(distance_matrix)
        return distance_matrix, tracks_indices

    def _get_updated_merge_distance_matrix_row(self, update_idx, ignore_idx, alive_indices):
        distance_matrix = THE_BIGGEST_DISTANCE*np.ones(len(alive_indices), dtype=np.float32)
        for i, idx in enumerate(alive_indices):
            if idx != update_idx and idx != ignore_idx:
                distance_matrix[i] = self._get_merge_distance(self.tracks[update_idx], self.tracks[idx])
        return distance_matrix

    def _get_merge_distance(self, track1, track2):
        if (track1.start_time > track2.end_time
            or track2.start_time > track1.end_time) \
                and track1.f_avg.is_valid() and track2.f_avg.is_valid():
            f_clust_dist = cosine_distance(track1.f_clust.get_clusters_matrix(), track2.f_clust.get_clusters_matrix())
            f_avg_dist = cosine_distance(track1.f_avg.get_avg(), track2.f_avg.get_avg())
            return min(f_avg_dist, f_clust_dist)
        return THE_BIGGEST_DISTANCE

    def _concatenate_tracks(self, i, idx, dist):
        # merge track and check who appeared first
        if self.tracks[i].end_time < self.tracks[idx].start_time:
            self.tracks[i].merge(self.tracks[idx], dist)
            self.tracks[idx] = None
            return i
        else:
            assert self.tracks[idx].end_time < self.tracks[i].start_time
            self.tracks[idx].merge(self.tracks[i], dist)
            self.tracks[i] = None
            return idx

    def _area(self, box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)

    def _ios(self, det1, det2):
        b1 = det1.to_tlbr()
        b2 = det2.to_tlbr()

        # intersection over self
        a1 = self._area(b1)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])
        return intersection / a1 if a1 > 0 else 0
