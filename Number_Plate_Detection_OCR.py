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

video_scenes = np.random.choice([file for file in os.listdir(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video"))][2:-1])
# video = cv2.VideoCapture(r"D:\PyCharm-Projects-2\Scene\BusStopArm_Video\Scene_7\Cam_4.mp4")
video = cv2.VideoCapture(r"C:\Users\ShrikantViswanathan\Videos\Singapore-LPR-Part-1.mp4")

output_path = os.path.join(os.getcwd(), "Output_Demo_Singapore.mp4")
crop_img_path = os.path.join(os.getcwd(), "crop_plate_images")
output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920 // 2, 1080 // 2))
while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    results = model.predict(frame, conf=0.5, agnostic_nms=True)
    # print("results:::::::::::::::::",results)
    labels = []
    for r in results:
        for box in r.boxes.boxes:
            print("box!!!!!!!!!!!!!!!!")
            # labels.append(all_labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%")
            # labels.append(str(round(100 * float(box[-2]), 1)) + "%")
            x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_image = frame[y1:h, x1:w]

            img_bytes = cv2.imencode('.jpg', cropped_image)[1].tobytes()
            response = client.detect_text(Image={'Bytes': img_bytes})

            MAIN_TEXT = ""
            textDetections = response['TextDetections']

            for text in textDetections:
                if text['Type'] == 'LINE':
                    MAIN_TEXT = MAIN_TEXT + " " + text['DetectedText']
            print("MAIN_TEXT", MAIN_TEXT)
            # labels[0] = labels[0] + " - " + MAIN_TEXT
            labels.append(MAIN_TEXT)
            print("labels", labels)
            cv2.imwrite("crop_plate_images/" + str(MAIN_TEXT) + ".jpg",cropped_image)

        detections3 = sv.Detections.from_yolov8(r)
        box_annotator = sv.BoxAnnotator(text_scale=0.8, text_thickness=2)
        frame = box_annotator.annotate(scene=frame, detections=detections3, labels=labels)

    cv2.imshow('detections', frame)
    output_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()





