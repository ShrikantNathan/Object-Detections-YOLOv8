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

video = cv2.VideoCapture(r"C:\Users\ShrikantViswanathan\Downloads\ai-cam6-POEtestHome.mp4")

output_path = os.path.join(os.getcwd(), "Output_Demo.mp4")
crop_img_path = os.path.join(os.getcwd(), "crop_plate_images")
output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))


def ocr_From_Img(results,frame):
    # print(frame.shape)
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
            # print("textDetections:::", textDetections)

            for text in textDetections:
                if text['Type'] == 'LINE' and text['Confidence'] > 75:
                    MAIN_TEXT = MAIN_TEXT + " " + text['DetectedText']
                    cv2.putText(frame, str(MAIN_TEXT), (x1-10, y1-10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0,
                                color=(0, 255, 255),thickness=2)
                    cv2.rectangle(frame,(x1,y1),(w,h),color=(0,0,255),thickness=3)
            print("MAIN_TEXT", MAIN_TEXT)
            # labels[0] = labels[0] + " - " + MAIN_TEXTq
            labels.append(MAIN_TEXT)
            print("labels", labels)

    return frame


while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # print('resized', frame.shape)
    results = model.predict(frame, conf=0.5, agnostic_nms=True)

    frame = ocr_From_Img(results,frame)

    cv2.imshow('detections', frame)
    output_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
