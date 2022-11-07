from openvino.runtime import Core
from scipy import stats

import pyrealsense2 as rs
import numpy as np
import collections
import time
import cv2

# settings
settings = {
    'rgb': True,
    'model_name': 'glasses',
    'force_cpu': False,
    'mean_horizon': 30,
    'stream_only': False
}

# set-up realsense stream
rgb = settings['rgb']
points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# images shape
frames = pipeline.wait_for_frames()
ir_frame = np.asanyarray(frames.get_infrared_frame(1).get_data())
rgb_frame = np.asanyarray(frames.get_color_frame().get_data())
print('\nFrames shape:')
print('IR: ', ir_frame.shape, '\nRGB: ', rgb_frame.shape)

# load openvino model
runtime_device = 'CPU'
ie_core = Core()
devices = ie_core.available_devices

print('\nAvailable devices:')
for device in devices:
    device_name = ie_core.get_property(device_name=device, name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

if 'GPU' in devices and not settings['force_cpu']:
    runtime_device = 'GPU'

print(f'Selected device: \n{runtime_device}: '
      f'{ie_core.get_property(device_name=runtime_device, name="FULL_DEVICE_NAME")}')

model_name = settings['model_name']
model_xml = f'model/{model_name}.xml'
model = ie_core.read_model(model=model_xml)
compiled_model = ie_core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
N, H, W, C = input_layer.shape
with open(f'model/labels_{model_name}.txt', 'r') as f:
    classes = f.read().splitlines()

print('\nModel loaded: --------->', model_name)
print('Input layer shape: ---->', input_layer.shape)
print('Output layer shape: --->', output_layer.shape)

try:
    processing_times = collections.deque()
    predictions = collections.deque()
    probs = collections.deque()
    while True:
        # get camera data
        frames = pipeline.wait_for_frames()
        ir1_frame = frames.get_infrared_frame(1)
        rgb1_frame = frames.get_color_frame()
        if rgb:
            np_image = np.asanyarray(rgb1_frame.get_data())
            stream = 'RGB'
        else:
            np_image = np.expand_dims(np.asanyarray(ir1_frame.get_data()), -1)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
            stream = 'IR'
        _, f_width = np_image.shape[:2]

        if not settings['stream_only']:
            # prepare input data
            resized_image = cv2.resize(src=np_image, dsize=(W, H))
            input_data = np.expand_dims(resized_image, 0).astype(np.float32)

            # model inference
            start = time.time()
            result = compiled_model([input_data])[output_layer]
            stop = time.time()

            # append prob and compute mean
            prob = np.max(result)
            probs.append(prob)
            probs.popleft() if len(probs) > settings['mean_horizon'] else {}
            mean_prob = np.mean(probs)
            # compute mean prediction
            result_index = np.argmax(result)
            prediction = classes[result_index]
            prediction = prediction.split()[-1]
            predictions.append(prediction)
            predictions.popleft() if len(predictions) > settings['mean_horizon'] else {}
            mean_prediction = stats.mode(predictions)[0]
            # get processing time and FPS
            processing_times.append(stop - start)
            processing_times.popleft() if len(processing_times) > 200 else {}
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time

        # display camera data
        cv2.namedWindow(stream + ' stream', cv2.WINDOW_AUTOSIZE)
        cv2.putText(
            img=np_image,
            text=f'Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)',
            org=(30, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        ) if not settings['stream_only'] else {}
        cv2.putText(
            img=np_image,
            text=f'Prediction: {mean_prediction} with prob {mean_prob:.2f}',
            org=(30, 60),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        ) if not settings['stream_only'] else {}
        cv2.imshow(stream + ' stream', np_image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

# ctrl-c
except KeyboardInterrupt:
    print('\nInterrupted')
# any error
except RuntimeError as e:
    print()
    print(e)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
