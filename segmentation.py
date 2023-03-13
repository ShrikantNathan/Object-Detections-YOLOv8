from ultralytics import YOLO
import cv2
import numpy as np
import os
import supervision as sv

model = YOLO("yolov8x-seg.pt")
video_scenes = np.random.choice([file for file in os.listdir(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video"))][2:-1])
video = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, "Cam_4.mp4"))

all_labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    results = model.predict(frame, conf=0.5, agnostic_nms=True)
    labels = []
    # for r in results:
    #     for box in r.boxes.boxes:
    #         labels.append(labels[int(box[-1]) + 1] + " " + str(round(100 * float(box[-2]), 1)) + "%")
    #
    #     detections = sv.Detections.from_yolov8(r)
    #     # labels = [f"{model.model.names[class_id]} {confidence:.2f}" for i, class_id, confidence, j in detections]
    #     box_annotator = sv.BoxAnnotator(text_scale=0.8, text_thickness=2)
    #     frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    for r in results:
        for box in r.boxes.boxes:
            labels.append(all_labels[int(box[-1]) + 1] + " " + str(round(100 * float(box[-2]), 1)) + "%")
        detections3 = sv.Detections.from_yolov8(r)
        box_annotator = sv.BoxAnnotator(text_scale=0.8, text_thickness=2)
        frame = box_annotator.annotate(scene=frame, detections=detections3, labels=labels)

    cv2.imshow('detections', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()