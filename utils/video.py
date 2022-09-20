import logging as log
import cv2 as cv


class CamCapture:
    def __init__(self, source):
        self.capture = None

        try:
            source = int(source[0])
            mode = 'cam'
        except ValueError:
            source = source[0]
            mode = 'video'

        if mode == 'cam':
            log.info('Connection  cam {}'.format(source))
            cap = cv.VideoCapture(source)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FPS, 30)
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
            assert cap.isOpened()
            self.capture = cap
        else:
            source = ''.join(source)
            log.info('Opening file {}'.format(source))
            cap = cv.VideoCapture(source)
            assert cap.isOpened()
            self.capture = cap

    def get_frame(self):
        has_frame, frame = self.capture.read()
        return has_frame, frame

    def get_source_parameters(self):
        frame_size = []
        fps = 0

        frame_size.append((int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                           int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))))
        fps = int(self.capture.get(cv.CAP_PROP_FPS))

        return frame_size, fps

    def release(self):
        self.capture.release()
