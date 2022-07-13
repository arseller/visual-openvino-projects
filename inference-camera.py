import pyrealsense2 as rs
import numpy as np
import cv2

from openvino.runtime import Core

# set-up realsense stream
rgb = False
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# load openvino model

try:
    while True:
        # get camera data
        frames = pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1)
        rgb1_frame = frames.get_color_frame()
        if not ir1_frame:
            continue
        if rgb:
            np_image = np.asanyarray(rgb1_frame.get_data())
            stream = 'RGB'
        else:
            np_image = np.asanyarray(ir1_frame.get_data())
            stream = 'IR'

        # display camera data
        cv2.namedWindow(stream + ' stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow(stream + ' stream', np_image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
