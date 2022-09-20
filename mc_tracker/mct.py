import json
import time

import numpy as np
import paho.mqtt.client as mqtt

from .sct import SingleCameraTracker, cosine_distance, THE_BIGGEST_DISTANCE


class MCT_track(object):
    def __init__(self, id, cam_id, start_time, f_avg, f_clust):
        self.id = id
        self.cam_id = cam_id
        self.start_time = start_time
        self.f_avg = f_avg
        self.f_clust = f_clust


class MultiCameraTracker:
    def __init__(self, reid_model,
                 cam_id,
                 sct_config={},
                 time_window=20,
                 global_match_thresh=0.35,
                 bbox_min_aspect_ratio=1.2,
                 bbox_max_aspect_ratio=6,
                 w_skip_ratio=0.1,
                 h_skip_ratio=0.125,
                 ignore_edge_objects=False,
                 sync_multi_cams=False,
                 broker_url='127.0.0.1'
                 ):
        self.sct = None
        self.tracks = []
        self.trajectories = []
        self.time = 0
        self.time_window = time_window
        self.global_match_thresh = global_match_thresh
        self.bbox_min_aspect_ratio = bbox_min_aspect_ratio
        self.bbox_max_aspect_ratio = bbox_max_aspect_ratio
        self.w_skip_ratio = w_skip_ratio
        self.h_skip_ratio = h_skip_ratio
        self.ignore_edge_objects = ignore_edge_objects
        self.sync_multi_cams = sync_multi_cams

        self.sct = SingleCameraTracker(cam_id, reid_model, **sct_config)
        self.mqtt_client = Messages(f'lab314up{cam_id}',
                                    f'lab314/up{cam_id}',
                                    broker_url=broker_url
                                    )

    def process(self, frame, all_detections):
        all_detections, screen_edge_objects, boundary_coord = \
            self._filter_detections(all_detections[0], frame.shape)

        self.sct.process(frame, all_detections, screen_edge_objects, boundary_coord)

        # receive mqtt data
        self._receeive_info()

        # check trajectory constraint
        self._trajectory_constraint()

        # merge all camera tracks
        if self.time > 0:
            self._merge_all()

        # transmit track information
        self._transmit_info()

        self.time += 1

    def _merge_all(self):
        # check cross-camera tracks in current tracks
        for i, sct_track in enumerate(self.sct.tracks):
            for j, mct_track in enumerate(self.tracks):
                if sct_track.cam_id == mct_track.cam_id and sct_track.id == mct_track.id:
                    dist = self._get_global_clusters_distance(self.tracks[j], self.sct.tracks[i])
                    if dist <= self.global_match_thresh:
                        self.sct.tracks[i].f_clust.global_merge(self.tracks[j].f_clust)
                    self.sct.tracks[i].trajectory = self.sct.cam_id
                    self.tracks[j] = None
                    break
            self.tracks = list(filter(lambda x: x is not None, self.tracks))
    
        # compare cross-camera tracks and current tracks
        stable_sct_track_indices = []
        for i, track in enumerate(self.sct.tracks):
            if track.is_stable() and track.trajectory == self.sct.cam_id:
                stable_sct_track_indices.append(i)
        mct_track_indices = [i for i in range(len(self.tracks))]

        distance_matrix = np.zeros((len(mct_track_indices),
                                    len(stable_sct_track_indices)), dtype=np.float32)

        for i, mct_track_idx in enumerate(mct_track_indices):
            for j, sct_track_idx in enumerate(stable_sct_track_indices):
                distance_matrix[i, j] = self._get_global_rectification_distance(self.tracks[mct_track_idx],
                                                                                self.sct.tracks[sct_track_idx])

        while distance_matrix.shape[0] > 0 and distance_matrix.shape[1] > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.global_match_thresh:
                self.sct.tracks[stable_sct_track_indices[j]].global_merge(self.tracks[mct_track_indices[i]], dist)
                self.tracks[mct_track_indices[i]] = None

                distance_matrix = np.delete(distance_matrix, i, 0)
                distance_matrix = np.delete(distance_matrix, j, 1)
                mct_track_indices = np.delete(mct_track_indices, i)
                stable_sct_track_indices = np.delete(stable_sct_track_indices, j)
            else:
                break

        self.tracks = list(filter(lambda x: x is not None, self.tracks))

    def _trajectory_constraint(self):
        # check trajectory info in sct tracks
        for trajectory in self.trajectories:
            for i, track in enumerate(self.sct.tracks):
                if track.id == trajectory[1] and track.cam_id == trajectory[2]:
                    if not track.off and track.trajectory == self.sct.cam_id and track.cross_camera_track \
                        and track.last_merge_dis > trajectory[4]:
                        self.sct.tracks[i].trajectory = trajectory[3]
                        self.sct.tracks[i].off = True
                        self.sct.tracks[i].feats_delivery_status = False
                        self.sct.tracks[i].f_queue = []
                    break

    def _transmit_info(self):
        for i, track in enumerate(self.sct.tracks):
            if track.feats_delivery_status and track.off \
                and track.f_avg.is_valid() and len(track.f_clust) >= 1:
                self.mqtt_client.deliver_feats(track.id, track.cam_id, track.f_avg.get_avg(),
                                               track.f_clust.get_clusters_matrix())
                self.sct.tracks[i].feats_delivery_status = False
                self.sct.tracks[i].cross_camera_track = True
                for j, trajectory in enumerate(self.trajectories):
                    if track.id == trajectory[1] and track.cam_id == trajectory[2]:
                        self.trajectories[j] = None
                        break
                self.trajectories = list(filter(lambda x: x is not None, self.trajectories))

            if track.pos_delivery_status and not track.off and track.cross_camera_track:
                self.mqtt_client.deliver_pos(track.id, track.cam_id, self.sct.cam_id, track.last_merge_dis)
                self.sct.tracks[i].pos_delivery_status = False

    def _receeive_info(self):
        # trajectory info
        for i, cc_pos in enumerate(self.mqtt_client.pos_temp):
            for j, trajectory in enumerate(self.trajectories):
                if cc_pos[1] == trajectory[1] and cc_pos[2] == trajectory[2]:
                    if cc_pos[4] < trajectory[4]:
                        self.trajectories[j][3] = cc_pos[3]
                        self.trajectories[j][4] = cc_pos[4]
                    cc_pos[0] = None
                    break
                
            if cc_pos[0] is not None:
                self.trajectories.append(cc_pos)
            
            self.mqtt_client.pos_temp[i] = None
        self.mqtt_client.pos_temp = list(filter(lambda x: x is not None, self.mqtt_client.pos_temp))

        # feats info
        for i, cc_track in enumerate(self.mqtt_client.feats_temp):
            _, id, cam_id, f_avg, f_clust = cc_track
            f_avg = np.asarray(f_avg).reshape(-1)
            f_clust = np.asarray(f_clust).reshape(len(f_clust), -1)

            # update the same ID object in MCT tracks
            for j, track in enumerate(self.tracks):
                if track.id == id and track.cam_id == cam_id:
                    self.tracks[j].f_avg = f_avg
                    self.tracks[j].f_clust = f_clust
                    cc_track[0] = None
                    break

            if cc_track[0] is not None:
                # initialize the cross camera track
                self.tracks.append(MCT_track(id, cam_id, self.time, f_avg, f_clust))

            self.mqtt_client.feats_temp[i] = None

            # delete the trajectory info of same ID object
            for j, trajectory in enumerate(self.trajectories):
                if trajectory[1] == id and trajectory[2] == cam_id:
                    self.trajectories[j] = None
                    break
            self.trajectories = list(filter(lambda x: x is not None, self.trajectories))
        self.mqtt_client.feats_temp = list(filter(lambda x: x is not None, self.mqtt_client.feats_temp))

    def check_processed_frame_num(self):
        hit_times = 0
        while hit_times < 2:
            # deliver current frame num
            self.mqtt_client.deliver_frame_num(self.time)

            # receive frame num
            frame_num_min = self.time
            for i, frame_num in enumerate(self.mqtt_client.frame_num_temp):
                if frame_num[1] < frame_num_min:
                    frame_num_min = frame_num[1]
                self.mqtt_client.frame_num_temp[i] = None
            self.mqtt_client.frame_num_temp = list(filter(lambda x: x is not None, self.mqtt_client.frame_num_temp))

            diff_num = self.time - frame_num_min
            if diff_num <= 2:
                hit_times += 1
            else:
                hit_times = 0
            time.sleep(0.1)

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

    def _get_global_rectification_distance(self, track1, track2):
        if track1.f_clust.shape[0] > 0 and len(track2.f_clust) > 0 \
                and track1.f_avg.shape[0] > 0 and track2.f_avg.is_valid():
            f_complex_dist = cosine_distance(track1.f_clust,
                                             track2.f_clust.get_clusters_matrix())
            f_avg_dist = cosine_distance(track1.f_avg, track2.f_avg.get_avg())
            return min(f_avg_dist, f_complex_dist.min())

        else:
            return THE_BIGGEST_DISTANCE

    def _get_global_clusters_distance(self, track1, track2):
        if track1.f_clust.shape[0] > 0 and len(track2.f_clust) > 0:
            f_complex_dist = cosine_distance(track1.f_clust,
                                            track2.f_clust.get_clusters_matrix())
            return f_complex_dist.min()
        else:
            return THE_BIGGEST_DISTANCE

    def get_tracked_objects(self):
        return self.sct.get_tracked_objects()


class Messages:
    def __init__(self, clientname, topic,
                 broker_url='127.0.0.1',
                 broker_port=1883
                 ):
        self.pos_temp = []
        self.feats_temp = []
        self.frame_num_temp = []

        self.topic = topic

        self.client = mqtt.Client(client_id=clientname, protocol=mqtt.MQTTv311, transport='tcp')  # creating a client instance
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.connect(broker_url, port=broker_port, keepalive=60)  # connect the client to a broker
        self.client.loop_start()  # threaded loop

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print('Connection Established')
        else:
            print(f'Bad Connection Returned Code = {rc}')

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe('lab314/#', qos=0)  # subscription wildcards

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, message):
        if message.topic != self.topic:
            data = json.loads(message.payload.decode('utf-8'))  # decode
            if data[0] == 1:
                data[4] = float(data[4])
                self.pos_temp.append(data)
            elif data[0] == 2:
                self.feats_temp.append(data)
            elif data[0] == 3:
                self.frame_num_temp.append(data)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print('Subscription Complete')

    def deliver_pos(self, ID, cam_ID, device_ID, last_merge_dis):
        data = [1, ID, cam_ID, device_ID, str(last_merge_dis)]
        output = json.dumps(data)  # encode
        self.client.publish(topic=self.topic, payload=output, qos=0, retain=False)

    def deliver_feats(self, ID, cam_ID, f_avg, f_clust):
        data = [2, ID, cam_ID, f_avg, f_clust]
        output = json.dumps(data, cls=NumpyEncoder)  # encode
        self.client.publish(topic=self.topic, payload=output, qos=0, retain=False)

    def deliver_frame_num(self, frame_num):
        data = [3, frame_num]
        output = json.dumps(data)  # encode
        self.client.publish(topic=self.topic, payload=output, qos=0, retain=False)

    def end(self):
        print('Ending Connection')
        self.client.loop_stop()
        self.client.disconnect()


# json encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
