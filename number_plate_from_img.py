from ultralytics import YOLO
import cv2
import numpy as np
import os
import supervision as sv
import ultralytics
import boto3
from datetime import datetime

model = ultralytics.YOLO(r"C:\Users\ShrikantViswanathan\yolov8_detections\yolov8_custom_best.pt")
all_labels = {0: u'license plate'}

client = boto3.client('rekognition')
frame = cv2.imread(rf"C:\Users\ShrikantViswanathan\Downloads\{np.random.choice(['Car-Plate-1.png', 'Car-Plate-2.png', 'Car-Plate-3.png'])}")
# frame = cv2.imread(rf"C:\Users\ShrikantViswanathan\Downloads\Car-Plate-3.png")

# results = model.predict(frame, conf=0.5, agnostic_nms=True,save_crop=True,hide_labels = True, hide_conf=True)
results = model.predict(frame, conf=0.5, agnostic_nms=True)
print("results:", results)
labels = []
labels1 = []

for r in results:
    for box in r.boxes.boxes:
        labels.append(all_labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%")
        x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cropped_image = frame[y1:h, x1:w]
        img_bytes = cv2.imencode('.jpg', cropped_image)[1].tobytes()
        response = client.detect_text(Image={'Bytes': img_bytes})

        MAIN_TEXT = ""
        textDetections = response['TextDetections']

        for text in textDetections:
            if text['Type'] == 'LINE':
                MAIN_TEXT = str(text['DetectedText'])
        print("MAIN TEXT", MAIN_TEXT)
        labels1.append(MAIN_TEXT)
        print("labels", labels)
        labels[0] = MAIN_TEXT
        # labels.append(MAIN_TEXT)

    detections3 = sv.Detections.from_yolov8(r)
    box_annotator = sv.BoxAnnotator(text_scale=0.8, text_thickness=2)
    print("labels:", labels1[0])
    frame = box_annotator.annotate(scene=frame, detections=detections3, labels=labels)

cv2.imshow('detections', frame)
current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_dt = datetime.strptime(current_dt, "%Y-%m-%d %H:%M:%S")
today_date = f'{current_dt.day}-{current_dt.month}-{current_dt.year}'
today_time = f'{current_dt.hour}-{current_dt.minute}-{current_dt.second}'

if not os.path.exists("saved_detections"):
    os.mkdir("saved_detections")

cv2.imwrite(f"saved_detections/detected_output_{today_time}.jpg", frame)

if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(1)
