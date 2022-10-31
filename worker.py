# TODO Контролировать направление камеры

import cv2
import numpy as np
import sys
from time import sleep
from configuration import init
from pathlib import Path

from shapely.geometry import Polygon, LineString

RED = (0,0,255)
GREEN = (0,255,0)

data = init()['default']

model_path = data['model_path']
data_path = data['data_path']

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
model.setInputParams(size=data['frame_size'], scale=1/255)

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


def ssd_out(results, zones):
    instanses = []
    for item in results:
        bbox, class_id, score = item
        (x, y, w, h) = bbox
        for index, polygons in enumerate(zones):
            verticles = np.array(zones[index])
            shapely_poly = Polygon(verticles)
            shapely_line = LineString([(x, y+h), (x+w, y+h)])
            instanses.append(bool(shapely_poly.intersects(shapely_line)))
        return any(instanses)


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
    for item in metadata:
        bbox, class_id, score = item
        (x, y, w, h) = bbox
        cv2.putText(frame, str(classes[class_id]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (200, 0, 50), 2)
        cv2.putText(frame, str(score), (x, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 200), 2)
        cv2.line(frame, (x, y + h), (x + w, y + h), (200, 0, 0), 10)
    return frame

def draw_polygons(data, frame, color=(255,255,255)):
    shapes = np.zeros_like(frame, np.uint8)
    for index, polygon in enumerate(data):
        verticles = np.array(data[index])
        cv2.fillPoly(shapes, pts=[verticles], color=color)
    return shapes

winname = "Output"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)



def main():
    cycles = 0
    while True:
        cycles += 1
        ret, frame = cap.read()
        
        if frame is None:
            break
        
        if cycles > 50 and cv2.getWindowProperty('Output',cv2.WND_PROP_VISIBLE) < 1:
            break # TODO проверять открыто ли окно и взводить влаг вместо cycles
        
        result = inference(frame, 0.5, ['0'])
        alpha = 0.7
        frame = render(result, frame)
        output = frame.copy()
        shapes = draw_polygons(data['polygons'], frame, GREEN)
        mask = shapes.astype(bool)
        output[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        if ssd_out(result, data['polygons']):
            cv2.putText(output, 'WARNING', (100, 175), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 200), 2)     
        cv2.imshow(winname, output)
        cv2.waitKey(1)

cv2.destroyAllWindows()

if __name__ == '__main__':
    main()