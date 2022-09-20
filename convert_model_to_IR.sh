cd /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer

# mean_values(BGR): [103.53,116.28,123.675]
# scale_values(BGR): [57.375,57.12,58.395]
python mo.py \
	--input_model /home/lab314/Hsuan/models/MTA/Clust/osnet_ain_192-96/osnet_ain_192-96.onnx \
	--output_dir /home/lab314/Hsuan/models/MTA/Clust/osnet_ain_192-96/ \
	--framework onnx \
	--input_shape [1,3,192,96] \
	--mean_values [103.53,116.28,123.675] \
	--scale_values [57.375,57.12,58.395] \
	--reverse_input_channels \
	--data_type FP16 \
	--progress
