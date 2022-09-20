random_seed = 0

obj_det = dict(
    trg_classes=(1,)
)

mct_config = dict(
    time_window=10,
    global_match_thresh=0.4,
    bbox_min_aspect_ratio=1.2,
    bbox_max_aspect_ratio=6,
    w_skip_ratio=0.075,
    h_skip_ratio=0.125,
    ignore_edge_objects=True,
    sync_multi_cams=False
)

cam_id = dict(
    id=1
)

sct_config = dict(
    initial_id=1,
    time_window=10,
    continue_time_thresh=3,
    confirm_time_thresh=3,
    match_threshold=0.25,
    stable_time_thresh=15,
    detection_occlusion_thresh=0.5,
    iou_dist_thresh=1.4,
    n_clusters=4,
    clust_init_dis_thresh=0.1,
    budget=3,
    rectify_thresh=0.35,
    rectify_time_thresh=8,
    rectify_length_thresh=2,
    merge_thresh=0.35,
    gap=50
)

visualization_config = dict(
    show_all_detections=True,
    max_window_size=(640, 480),
    out_fps=20
)
