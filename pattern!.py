RESO = 320, 320

import cv2
import numpy as np
import sys
from time import sleep
from json import dump, load

# Geometry
from EasyROI import EasyROI
from shapely.geometry import Polygon, Point

# Opencv DNN
is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
net = cv2.dnn.readNet(r'dnn_model\yolov4-tiny.weights', r'dnn_model\yolov4-tiny.cfg')

if is_cuda:
    print("Attempty to use CUDA")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
else:
    print("Running on CPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=RESO, scale=1/255)

# Load class lists
classes = []
with open(r'dnn_model\classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize input stream
cap = cv2.VideoCapture('worker.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def ssd_out():
    pass

def inference(frame, confidence, filter):
    # Object detection
    result = []
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):    
        if score > confidence and str(class_id) in filter: # TODO Сделать по фильтру
            result.append((bbox, class_id, score))
            
    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)  # TODO Добавить для устранения оверлапов
    # result_class_ids = []
    # result_confidences = []
    # result_boxes = []

    # for i in indexes:
    #     result_confidences.append(confidences[i])
    #     result_class_ids.append(class_ids[i])
    #     result_boxes.append(boxes[i])    
             
    return result

def render(metadata, frame):
    shapes = np.zeros_like(frame, np.uint8)
    for item in metadata:
        bbox, class_id, score = item
        (x, y, w, h) = bbox
        cv2.putText(shapes, str(classes[class_id]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (200, 0, 50), 2)
        cv2.putText(shapes, str(score), (x, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 200), 2)
        cv2.line(shapes, (x, y + h), (x + w, y + h), (200, 0, 0), 10)
    return shapes
    

winname = "Output"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)


def main():
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        result = inference(frame, 0.5, ['0'])
        alpha = 0.05
        output = frame.copy()
        shapes = render(result, frame)
        mask = shapes.astype(bool)
        output[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        cv2.imshow(winname, output)
        cv2.waitKey(1)

cv2.destroyAllWindows()

if __name__ == '__main__':
    main()