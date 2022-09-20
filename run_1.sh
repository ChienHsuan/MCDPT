#!/bin/bash

python multi_camera_multi_target_tracking.py \
	-i '../video/video_1.avi' \
	--m_detector '../models/person-detection-retail-0013/FP16/person-detection-retail-0013.xml' \
	--m_reid '../models/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml' \
	--config 'configs/person.py' \
	--device1 'CPU' \
	--device2 'CPU' \
	--broker_url '127.0.0.1' \
	--method 'mtmct' \
	--cam_id 1 \
	--no_show
