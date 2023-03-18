from __future__ import print_function

from firebase_admin import credentials, initialize_app, storage
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from openvino.runtime import Core
import torch
import cv2
import numpy as np

import os.path
import cv2
import numpy as np
from google.cloud import storage as store
from firebase_admin import credentials, initialize_app
import os
os.environ.setdefault("GCLOUD_PROJECT", "frick-dashcam")

cred = credentials.Certificate("frick-dashcam-918ac2a2ce16.json")
initialize_app(cred, {'storageBucket': 'frick-dashcam.appspot.com'})
ie_core = Core()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = 'best.pt'
model = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True)
# A directory where the model will be downloaded.
base_model_dir = "model"
# The name of the model from Open Model Zoo.
detection_model_name = "vehicle-detection-0200"
recognition_model_name = "vehicle-attributes-recognition-barrier-0039"
# Selected precision (FP32, FP16, FP16-INT8)
precision = "FP32"

# Check if the model exists.
detection_model_path = (
    f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
)
recognition_model_path = (
    f"model/intel/{recognition_model_name}/{precision}/{recognition_model_name}.xml"
)
def model_init(model_path: str) -> Tuple:
    """
    Read the network and weights from file, load the
    model on the CPU and get input and output names of nodes

    :param: model: model architecture path *.xml
    :retuns:
            input_key: Input node network
            output_key: Output node network
            exec_net: Encoder model network
            net: Model network
    """

    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=model_path)
    # Compile the model for CPU (you can use GPU or MYRIAD as well).
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    # Get input and output names of nodes.
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model


input_key_de, output_keys_de, compiled_model_de = model_init(detection_model_path)
# Recognition model initialization.
input_key_re, output_keys_re, compiled_model_re = model_init(recognition_model_path)
height_de, width_de = list(input_key_de.shape)[2:]
    # Get input size - Recognition.
height_re, width_re = list(input_key_re.shape)[2:]

def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    """
    Use bounding boxes from detection model to find the absolute car position

    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: car_position: car's absolute position
    """
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    boxes = boxes[:, 2:]
    car_position = []
    for box in boxes:
        conf = box[0]
        if conf > threshold:
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[1:])
            ]

            car_position.append([x_min, y_min, x_max, y_max])

    return car_position
def vehicle_recognition(compiled_model_re, input_size, raw_image):
    """
    Vehicle attributes recognition, input a single vehicle, return attributes
    :param: compiled_model_re: recognition net
    :param: input_size: recognition input size
    :param: raw_image: single vehicle image
    :returns: attr_color: predicted color
                       attr_type: predicted type
    """
    colors = ['White', 'Gray', 'Yellow', 'Red', 'Green', 'Blue', 'Black']
    types = ['Car', 'Bus', 'Truck', 'Van']


    resized_image_re = cv2.resize(raw_image, input_size)
    input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)

    predict_colors = compiled_model_re([input_image_re])[compiled_model_re.output(1)]

    predict_colors = np.squeeze(predict_colors, (2, 3))
    predict_types = compiled_model_re([input_image_re])[compiled_model_re.output(0)]
    predict_types = np.squeeze(predict_types, (2, 3))

    attr_color, attr_type = (colors[np.argmax(predict_colors)],
                             types[np.argmax(predict_types)])
    return attr_color, attr_type
def car_pred(file):
    image_de = file#file
    resized_image_de = cv2.resize(image_de, (width_de, height_de))
    # Expand the batch channel to [1, 3, 256, 256].
    input_image_de = np.expand_dims(resized_image_de.transpose(2, 0, 1), 0)
    boxes = compiled_model_de([input_image_de])[output_keys_de]
    boxes = np.squeeze(boxes, (0, 1))
    # Remove zero only boxes.
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    car_position = crop_images(image_de, resized_image_de, boxes)
    if len(car_position)>0:
        pos = car_position[0]
        test_car = image_de[pos[1]:pos[3], pos[0]:pos[2]]
        if not test_car.size==0:
            resized_image_re = cv2.resize(test_car, (width_re, height_re))
            input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)
            return vehicle_recognition(compiled_model_re, (72, 72), test_car)
    return "",""


client = store.Client()
list_of_filenames = []
NEW_FILE = False
for blob in client.list_blobs('frick-dashcam.appspot.com', prefix='recordings/'):
    list_of_filenames.append(blob.name)

while True:
    for blob in client.list_blobs('frick-dashcam.appspot.com', prefix='recordings/'):
      if not blob.name in list_of_filenames:
        list_of_filenames.append(blob.name)
        NEW_FILE = True
    print(1)
    if NEW_FILE == True:
        file_name = list_of_filenames[-1]
        bucket = storage.bucket()
        blob = bucket.blob(file_name)
        blob.download_to_filename(file_name[11:])#file name
        i = 0
        frames=0
        cap = cv2.VideoCapture(file_name[11:])

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Frame', frame)
                if frames == 40:
                    results = model(frame)
                    preds = results.pandas().xyxy[0] # im predictions (pandas)
                    car_color, car_type = car_pred(frame)
                    print(preds)
                    print(car_color+" "+ car_type)
                    if preds.empty or (car_color != 'Red' and car_type != 'Car'):
                        print(1)
                    else:
                        print("Confe: %d", preds.get('confidence')[0])
                        if (preds.get('confidence')[0] > .5):
                            i += 1
                            if (i == 2):
                                print("Pay Ticket")
                                break
                    frames=0
                frames+=1
            else:
                break

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cap.release()
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()
        NEW_FILE=False

