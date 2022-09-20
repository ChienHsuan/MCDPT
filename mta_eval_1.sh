conda activate hsuan_reid

python mta_evaluation.py \
	-i /home/lab314/MTA_ext_short/ \
	--m_detector ../models/person-detection-retail-0013/FP16/person-detection-retail-0013.xml \
	--m_reid ../models/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml \
	--config configs/person_evaluation.py \
	--device1 'CPU' \
	--device2 'CPU' \
	--broker_url '127.0.0.1' \
	--method 'mtmct' \
	--cam_id 0 \
	--initial_id 1 \
	--logs_dir logs/
