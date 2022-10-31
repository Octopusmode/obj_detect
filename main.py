from turtle import width
import cv2
import numpy as np
import time

start_time = time.time()
display_time = 0.2
fc = 0
FPS = 0

#Opencv DNN
# net = cv2.dnn.readNet(r'dnn_model\yolov4-tiny.weights', r'dnn_model\yolov4-tiny.cfg')
# net = cv2.dnn.readNet(r'dnn_model\yolov7-tiny.weights', r'dnn_model\yolov7-tiny.cfg')

net = cv2.dnn.readNetFromONNX('fire_detector.onnx')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

# Load class lists
classes = []
with open(r'classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

IMG_WIDTH = 1024
IMG_HEIGHT = 768

# Initialize camer/stream
cap = cv2.VideoCapture('video_park.mp4')
# cap = cv2.VideoCapture('rtsp://vidanalitic:q23NYz9SXQ@192.168.103.48:554/RVi/1/1')
# cap = cv2.VideoCapture('rtsp://admin:Admin_admin0@192.168.150.32/cam/realmonitor?channel=1&subtype=0')

# cap = cv2.VideoCapture('rtsp://vidanalitic:q23NYz9SXQ@192.168.103.48:554/RVi/1/1')
# rtsp://vidanalitic:q23NYz9SXQ@192.168.103.48:554/PSIA/streaming/channels/101
# cap = cv2.VideoCapture(r'video_park.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
first = False
count = 1000

def contains(x1, y1, x2, y2, ox1=120, oy1=70, ox2=1580, oy2=880):
   return x1 < ox1 < ox2 < x2 and y1 < oy1 < oy2 < y2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1024,768))

while True:
    ret, frame = cap.read()
    if type(frame) == 'NoneType':
        frame = to_save.copy()
    else:
        to_save = frame.copy()
    frame_net = cv2.bilateralFilter(frame, 5, 20, 20)
    # if not first:
    #     cv2.imshow('filter', frame_net)
    #     first = True
    
    if ret==True:
        grab = cv2.flip(frame,0)

        # write the flipped frame
        out.write(grab)
    
    fc+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()
        
    fps_disp = "FPS: "+str(round(FPS))[:5]
    
    # Object detection
    # print(model.detect(frame_net))
    
    i = 0
    
    # for class_id, score, bbox in zip(class_ids, scores, bboxes):
    #         (x, y, w, h) = bbox
    #         i += 1
    #         # print(x, y, w, h)
    #         if score > 0.25: # and class_id == 56:
    #             cv2.putText(frame, str(classes[class_id]) + str(i), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (200, 0, 50), 2)
    #             cv2.putText(frame, str(score)[:4], (x, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 200), 2)
    #             cv2.rectangle(frame, (x, y),(x+w, y + h), (200, 0, 0), 3)
                
    #             # key_pressed = cv2.waitKey(1)
                
    #             # ih, iw = to_save.shape[:2]
    #             # # if count % 100 == 0 and x+w < 1500 and y+h < 900:
    #             # if key_pressed == 99:
    #             #     cv2.imwrite(r"out\frame%d.jpg" % count, to_save)     # save frame as JPEG file
    #             #     count += 1 
                    
    
    # cv2.putText(frame, fps_disp, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    
    # print('class_ids', class_ids)
    # print('score', scores)
    # print('bboxes', bboxes)
    
    res_height = int(frame.shape[0] * 0.75)
    res_width = int(frame.shape[1] * 0.75)
    
    resized = cv2.resize(frame, (res_width, res_height), interpolation= cv2.INTER_CUBIC)
    cv2.imshow('Output', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    out.release()