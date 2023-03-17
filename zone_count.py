import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import os, glob
from typing import Union, List, AnyStr
from datetime import datetime
import numpy as np

screen_resolutions: tuple = (1920, 1080)
output_path: Union[AnyStr, List[AnyStr]] = str()
# zone_polygon = np.array([[700, 520], [1280, 520], [700, 960], [0, 520]])
zone_polygon = np.array([[500, 520], [1280, 520], [1280, 1200], [500, 1200]])
current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_dt = datetime.strptime(current_dt, "%Y-%m-%d %H:%M:%S")
today_date = f'{current_dt.day}-{current_dt.month}-{current_dt.year}'
today_time = f'{current_dt.hour}-{current_dt.minute}-{current_dt.second}'

if not os.path.exists(f"YOLOv8/Outputs/{today_date}"):
    os.makedirs(f'YOLOv8/Outputs/{today_date}')
else:
    output_path = os.path.join(os.getcwd(), "YOLOv8", "Outputs", f"{today_date}", f"Output_{today_time}.mp4")

output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument("--webcam-resolution", default=[1920, 1080], nargs=2, type=int)
    args = parser.parse_args()
    return args


def main():
    video_channel = os.path.join(glob.glob(r"C:\Users\ShrikantViswanathan\Videos")[-1], "CityDrive2-New.mp4")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(video_channel)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO('yolov8x-seg.pt')

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=0.8)

    while True:
        ret, frame = cap.read()
        result = model.predict(frame, conf=0.8, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [f"{model.model.names[class_id]} {confidence:.2f}" for i, confidence, class_id, j in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=args.webcam_resolution)
        zone_annotator = sv.PolygonZoneAnnotator(zone, color=sv.Color.blue(), thickness=2, text_thickness=4,
                                                 text_scale=2, text_color=sv.Color.white())
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(frame)
        cv2.imshow('yolov8', frame)
        output_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()