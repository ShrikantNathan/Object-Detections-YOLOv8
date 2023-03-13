from ultralytics import YOLO
import cv2
import numpy as np
import os
import supervision as sv
import ultralytics
import boto3

model = ultralytics.YOLO(r"C:\Users\ShrikantViswanathan\yolov8_detections\yolov8_custom_best.pt")
all_labels = {0: u'license plate'}

client = boto3.client('rekognition')
frame = cv2.imread(r"D:\PyCharm-Projects-2\Random_LPR_Test_Images\20.plates.jpg")

# results = model.predict(frame, conf=0.5, agnostic_nms=True,save_crop=True,hide_labels = True, hide_conf=True)
results = model.predict(frame, conf=0.5, agnostic_nms=True)
print("results:::::::::::::::::", results)
labels = []
for r in results:
    for box in r.boxes.boxes:
        labels.append(all_labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%")
        print("#########################",int(box[0]))
        x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_image = frame[y1:h, x1:w]
        img_bytes = cv2.imencode('.jpg', cropped_image)[1].tobytes()
        response = client.detect_text(Image={'Bytes': img_bytes})

        MAIN_TEXT = ""
        textDetections = response['TextDetections']

        for text in textDetections:
            if text['Type'] == 'LINE':
                MAIN_TEXT = MAIN_TEXT + " " + text['DetectedText']
        print("@@@@@@@@@@@@@", MAIN_TEXT)
        print("labels",labels)
        labels[0]=labels[0]+" - "+MAIN_TEXT
        # labels.append(MAIN_TEXT)

    detections3 = sv.Detections.from_yolov8(r)
    box_annotator = sv.BoxAnnotator(text_scale=0.8, text_thickness=2)
    frame = box_annotator.annotate(scene=frame, detections=detections3, labels=labels)

cv2.imshow('detections', frame)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(1)





