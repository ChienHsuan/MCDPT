random_seed = 0

obj_det = dict(
    trg_classes=(1,)
)

mct_config = dict(
    time_window=10,
    match_threshold=0.22,
    global_match_thresh=0.37,
    bbox_min_aspect_ratio=1.2,
    bbox_max_aspect_ratio=6,
    w_skip_ratio=0.0,
    h_skip_ratio=0.0,
    ignore_edge_objects=False,
    sync_multi_cams=False
)

cam_id = dict(
    id=1
)

sct_config = dict(
    initial_id=1,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    budget=100,
    num_clusters=4,
    clust_init_dis_thresh=0.1,
    continue_time_thresh=3,
    time_window=10,
    stable_time_thresh=13,
    detection_occlusion_thresh=0.5,
    rectify_time_thresh=8,
    rectify_length_thresh=2,
    rectify_thresh=0.2,
    merge_thresh=0.2,
    gap=50
)

visualization_config = dict(
    show_all_detections=True,
    max_window_size=(640, 480),
    out_fps=20
)
