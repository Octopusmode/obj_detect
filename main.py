from turtle import width
import cv2
import numpy as np
import time

start_time = time.time()
display_time = 0.2
fc = 0
FPS = 0

#Opencv DNN
net = cv2.dnn.readNet(r'dnn_model\yolov4-tiny.weights', r'dnn_model\yolov4-tiny.cfg')
# net = cv2.dnn.readNet(r'dnn_model\yolov7-tiny.weights', r'dnn_model\yolov7-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

# Load class lists
classes = []
with open(r'D:\src\obj_detect\dnn_model\classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Initialize camer/stream
cap = cv2.VideoCapture('video_park.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
first = False


while True:
    ret, frame = cap.read()
    frame_net = cv2.bilateralFilter(frame, 5, 20, 20)
    
    # if not first:
    #     cv2.imshow('filter', frame_net)
    #     first = True
    
    fc+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()
        
    fps_disp = "FPS: "+str(round(FPS))[:5]
    
    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame_net)
    
    i = 0
    
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            i += 1
            # print(x, y, w, h)
            if score > 0.2 and class_id == 0:
                cv2.putText(frame, str(classes[class_id]) + str(i), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (200, 0, 50), 2)
                cv2.putText(frame, str(score)[:4], (x, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 200), 2)
                cv2.rectangle(frame, (x, y),(x+w, y + h), (200, 0, 0), 3)
    
    cv2.putText(frame, fps_disp, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    # print('class_ids', class_ids)
    # print('score', scores)
    # print('bboxes', bboxes)
    
    res_height = int(frame.shape[0] * 0.75)
    res_width = int(frame.shape[1] * 0.75)
    
    resized = cv2.resize(frame, (res_width, res_height), interpolation= cv2.INTER_CUBIC)
    cv2.imshow('Output', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break