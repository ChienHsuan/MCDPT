import os
import sys
from datetime import datetime
import cv2


def cam_setup(cam_id, mode, dir_name='', file_name=''):
    cap_all = []
    output_video_all = []
    for id in cam_id:
        cap = cv2.VideoCapture(f'udpsrc port=5{str(id).zfill(4)} ! application/x-rtp,encoding-name=JPEG,payload=26 ! rtpjpegdepay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        cap_all.append(cap)
        if mode == '1':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(os.path.join(dir_name, f'video_{id}_{file_name}.mp4'), fourcc, 20, (640, 480))
            output_video_all.append(output_video)
    if mode == '0':
        return cap_all
    else:
        return cap_all, output_video_all

def cam_display(cap_all):
    while True:
        for i in range(len(cap_all)):
            ret, frame = cap_all[i].read()
            if ret == True :
                cv2.imshow(f'Camera {i+1}', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    print('break')
                    break
            else:
                print(f'Camera {i+1} error.')

    for i in range(len(cap_all)):
        cap_all[i].release()
    cv2.destroyAllWindows()

def cam_record(end_time, cap_all, output_video_all):
    while (datetime.now() <= end_time):
        for i in range(len(cap_all)):
            ret, frame = cap_all[i].read()
            if ret == True :
                output_video_all[i].write(frame)
                cv2.imshow(f'Camera {i+1}', frame)
                cv2.waitKey(1)
            else:
                print(f'Camera {i+1} error.')

    for i in range(len(cap_all)):
        cap_all[i].release()
        output_video_all[i].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ######### config settings #########
    cam_id = [1, 2, 3]
    video_dir = '/USER-DEFINED-PATH/'  # for recording videos only
    time_start = ["23_05_2022-13_00"]  # for recording videos only
    time_end = ["23_05_2022-18_00"]  # for recording videos only
    ###################################

    print('Mode 0: display videos; mode 1: record videos.')
    mode = input('Mode: ')
    assert mode == '0' or mode == '1'
    assert len(cam_id) > 0
    if mode == '0':
        print(f'Display videos for camera {cam_id} ...')
        cap_all = cam_setup(cam_id, mode)
        cam_display(cap_all)
    else:
        assert os.path.isdir(video_dir)
        assert len(time_start) == len(time_end)
        index = 0
        dt_start = datetime.strptime(time_start[index], "%d_%m_%Y-%H_%M")
        dt_end = datetime.strptime(time_end[index], "%d_%m_%Y-%H_%M")
        
        print(f'Record videos for {len(time_start)} time period(s) ...')
        while True:
            if datetime.now() >= dt_start:
                print(f'Stage {index} start at {time_start[index]}')
                file_name_time = datetime.now().strftime("%d_%m_%Y-%H_%M")
                cap_all, output_video_all = cam_setup(cam_id, mode, dir_name=video_dir, file_name=file_name_time)
                print('Stream videos ...')
                cam_record(dt_end, cap_all, output_video_all)
                print(f'Stage {index} end at {time_start[index]}')
                index += 1
                if index >= len(time_start):
                    break
                dt_start = datetime.strptime(time_start[index], "%d_%m_%Y-%H_%M")
                dt_end = datetime.strptime(time_end[index], "%d_%m_%Y-%H_%M")
    sys.exit()
