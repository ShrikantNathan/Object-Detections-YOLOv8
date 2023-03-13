import numpy as np
import ultralytics
import os
import cv2
import supervision as sv
import glob
from datetime import datetime
from typing import Union, List, AnyStr
from extractor import detect_text

IMAGE_RES = DIMS = (1080, 1920)

model = ultralytics.YOLO("E:\yolov8_detections\yolov8x.pt")
segmodel = ultralytics.YOLO("yolov8x-seg.pt")
lpr_model = ultralytics.YOLO(r"E:\d\YOLOv8 Root\yolov8_custom_best.pt")

blank_screen = np.zeros(IMAGE_RES, dtype=np.uint8)
output_path: Union[AnyStr, List[AnyStr]] = str()

current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_dt = datetime.strptime(current_dt, "%Y-%m-%d %H:%M:%S")
today_date = f'{current_dt.day}-{current_dt.month}-{current_dt.year}'
today_time = f'{current_dt.hour}-{current_dt.minute}-{current_dt.second}'

if not os.path.exists(f"YOLOv8/Outputs/{today_date}"):
    os.makedirs(f'YOLOv8/Outputs/{today_date}')
else:
    output_path = os.path.join(os.getcwd(), "YOLOv8", "Outputs", f"{today_date}", f"Output_{today_time}.mp4")

# output_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (1920, 1080))

# images = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
# images2 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
# images3 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
# images4 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
#
# image1_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images)),
#               (np.floor_divide(IMAGE_RES[1], 2), np.floor_divide(IMAGE_RES[0], 2)), interpolation=cv2.INTER_LINEAR)
# image2_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images2)),
#               (np.floor_divide(IMAGE_RES[1], 2), np.floor_divide(IMAGE_RES[0], 2)), interpolation=cv2.INTER_LINEAR)
# image3_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images3)),
#               (np.floor_divide(IMAGE_RES[1], 2), np.floor_divide(IMAGE_RES[0], 2)), interpolation=cv2.INTER_LINEAR)
# image4_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images4)),
#               (np.floor_divide(IMAGE_RES[1], 2), np.floor_divide(IMAGE_RES[0], 2)), interpolation=cv2.INTER_LINEAR)
#
# results = model.predict(image1_reset, conf=0.5)
# results2 = model.predict(image2_reset, conf=0.5)
# results3 = model.predict(image3_reset, conf=0.5)
# results4 = model.predict(image4_reset, conf=0.5)
#
# results_seg = segmodel.predict(image1_reset, conf=0.5)
# results_seg2 = segmodel.predict(image2_reset, conf=0.5)
# results_seg3 = segmodel.predict(image3_reset, conf=0.5)
# results_seg4 = segmodel.predict(image4_reset, conf=0.5)
#
# detections = sv.Detections.from_yolov8(results)
# detections2 = sv.Detections.from_yolov8(results2)
# detections3 = sv.Detections.from_yolov8(results3)
# detections4 = sv.Detections.from_yolov8(results4)
#
# labels = [f"{model.names[class_id]} {confidence:.2f}" for i, confidence, class_id, j in detections]
# labels2 = [f"{model.names[class_id]} {confidence:.2f}" for k, confidence, class_id, l in detections2]
# labels3 = [f"{model.names[class_id]} {confidence:.2f}" for m, confidence, class_id, n in detections3]
# labels4 = [f"{model.names[class_id]} {confidence:.2f}" for x, confidence, class_id, y in detections4]


# def annotate_objects_inside_frame():
#     box_annotator = sv.BoxAnnotator(color=sv.Color.red(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
#     box_annotator2 = sv.BoxAnnotator(color=sv.Color.blue(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
#     box_annotator3 = sv.BoxAnnotator(color=sv.Color.green(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
#     box_annotator4 = sv.BoxAnnotator(color=sv.Color.white(), text_color=sv.Color.black(), text_scale=0.8, text_thickness=2)
#
#     frame1 = box_annotator.annotate(scene=image1_reset, detections=detections, labels=labels)
#     frame2 = box_annotator2.annotate(scene=image2_reset, detections=detections2, labels=labels2)
#     frame3 = box_annotator3.annotate(scene=image3_reset, detections=detections3, labels=labels3)
#     frame4 = box_annotator4.annotate(scene=image4_reset, detections=detections4, labels=labels4)
#
#     frame1, frame2 = Annotator(image1_reset).result(), Annotator(image2_reset).result()
#     frame3, frame4 = Annotator(image3_reset).result(), Annotator(image4_reset).result()
#
#     res_stacked_1 = np.hstack((image1_reset, image2_reset))
#     res_stacked_2 = np.hstack((image3_reset, image4_reset))
#     blank_screen[0: IMAGE_RES[0] // 2, 0: IMAGE_RES[1]] = res_stacked_1
#     blank_screen[IMAGE_RES[0] // 2: IMAGE_RES[0], 0: IMAGE_RES[1]] = res_stacked_2
#     cv2.imshow('results', blank_screen)
#     cv2.waitKey(0)


def segment_objects_inside_frame():
    pass


def force_resize_high_res_frames(frame: np.ndarray):
    if np.bitwise_and(np.greater(frame.shape[1], 1920), np.greater(frame.shape[0], 1080)):
        frame_resz = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resz = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return frame_resz


def run_instance_segmentation_inside_videos():
    blank_screen_seg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    video_scenes = np.random.choice([file for file in os.listdir(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video"))][2:])
    video1 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, "Cam_2.mp4"))
    video2 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, 'Cam_3.mp4'))
    video3 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, "Cam_4.mp4"))
    video4 = cv2.VideoCapture(r"C:\Users\Shrikant\Videos\CityDrive.mp4")
    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 22, (1920, 1080))

    while True:
        ret5, frame5 = video1.read()  # Instance Segmentation (4-6)
        ret6, frame6 = video2.read()
        ret7, frame7 = video3.read()
        ret8, frame8 = video4.read()

        frame_seg = cv2.resize(frame5, dsize=(0, 0), fx=0.5, fy=0.5)
        frame2_seg = cv2.resize(frame6, dsize=(0, 0), fx=0.5, fy=0.5)
        frame3_seg = cv2.resize(frame7, dsize=(0, 0), fx=0.5, fy=0.5)
        frame4_seg = cv2.resize(force_resize_high_res_frames(frame8), dsize=(0, 0), fx=0.5, fy=0.5)

        results_seg = segmodel.predict(frame_seg, conf=0.5, agnostic_nms=True, task="segment")[-1]
        results_seg2 = segmodel.predict(frame2_seg, conf=0.5, agnostic_nms=True, task="segment")[-1]
        results_seg3 = segmodel.predict(frame3_seg, conf=0.5, agnostic_nms=True, task="segment")[-1]
        results_seg4 = segmodel.predict(frame4_seg, conf=0.5, agnostic_nms=True, task="segment")[-1]

        detections_seg = sv.Detections.from_yolov8(results_seg)
        detections2_seg = sv.Detections.from_yolov8(results_seg2)
        detections3_seg = sv.Detections.from_yolov8(results_seg3)
        detections4_seg = sv.Detections.from_yolov8(results_seg4)

        labels_seg = [f"{model.model.names[class_id]} {confidence:.2f}" for i, confidence, class_id, j in detections_seg]
        labels_seg2 = [f"{model.model.names[class_id]} {confidence2:.2f}" for k, confidence2, class_id, l in detections2_seg]
        labels_seg3 = [f"{model.model.names[class_id]} {confidence3:.2f}" for m, confidence3, class_id, n in detections3_seg]
        labels_seg4 = [f"{model.model.names[class_id]} {confidence4:.2f}" for m, confidence4, class_id, n in detections4_seg]

        box_annotator_seg = sv.BoxAnnotator(text_scale=0.8, text_thickness=2, text_padding=2)
        box_annotator_seg2 = sv.BoxAnnotator(text_scale=0.8, text_thickness=2, text_padding=2)
        box_annotator_seg3 = sv.BoxAnnotator(text_scale=0.8, text_thickness=2, text_padding=2)
        box_annotator_seg4 = sv.BoxAnnotator(text_scale=0.8, text_thickness=2, text_padding=2)

        frame_seg = box_annotator_seg.annotate(scene=frame_seg, detections=detections_seg, labels=labels_seg)
        frame2_seg = box_annotator_seg2.annotate(scene=frame2_seg, detections=detections2_seg, labels=labels_seg2)
        frame3_seg = box_annotator_seg3.annotate(scene=frame3_seg, detections=detections3_seg, labels=labels_seg3)
        frame4_seg = box_annotator_seg4.annotate(scene=frame4_seg, detections=detections4_seg, labels=labels_seg4)

        blank_screen_seg[0: DIMS[0] // 2, 0: DIMS[1]] = np.hstack((frame_seg, frame2_seg))
        blank_screen_seg[DIMS[0] // 2: DIMS[0], 0: DIMS[1]] = np.hstack((frame3_seg, frame4_seg))
        cv2.imshow("segmentation", blank_screen_seg)
        vid_writer.write(blank_screen_seg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def extract_text_from_bounding_boxes(cropped_pic: np.ndarray):
    import boto3

    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')

    response = client.detect_text(Image=cropped_pic)

    textDetections = response['TextDetections']
    print('Detected text\n----------')
    for text in textDetections:
        print('Detected text:' + text['DetectedText'])
        print('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
        print('Id: {}'.format(text['Id']))
        if 'ParentId' in text:
            print('Parent Id: {}'.format(text['ParentId']))
        print('Type:' + text['Type'])
        print()
    return len(textDetections)


def annotate_objects_inside_videos():
    blank_screen2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    video_scenes = np.random.choice([file for file in os.listdir(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video"))][2:])
    video1 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, "Cam_2.mp4"))
    video2 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, 'Cam_3.mp4'))
    video3 = cv2.VideoCapture(os.path.join(os.getcwd(), "Scene", "BusStopArm_Video", video_scenes, "Cam_4.mp4"))
    video4 = cv2.VideoCapture(r"C:\Users\Shrikant\Videos\CityDrive.mp4")
    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 22, (1920, 1080))

    while True:
        ret, frame = video1.read()  # Object Detection (1-3)
        ret2, frame2 = video2.read()
        ret3, frame3 = video3.read()
        ret4, frame4 = video4.read()

        frame: np.ndarray = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
        frame2: np.ndarray = cv2.resize(frame2, dsize=(0, 0), fx=0.5, fy=0.5)
        frame3: np.ndarray = cv2.resize(frame3, dsize=(0, 0), fx=0.5, fy=0.5)
        frame4: np.ndarray = cv2.resize(force_resize_high_res_frames(frame4), dsize=(0, 0), fy=0.5, fx=0.5, interpolation=cv2.INTER_NEAREST)

        results = model.predict(frame, conf=0.5, agnostic_nms=True)[-1]
        results2 = model.predict(frame2, conf=0.5, agnostic_nms=True)[-1]
        results3 = model.predict(frame3, conf=0.5, agnostic_nms=True)[-1]
        results4 = model.predict(frame4, conf=0.5, agnostic_nms=True)[-1]
        lpr_results = lpr_model.predict(frame, conf=0.5, agnostic_nms=True, save_crop=True)[0]
        lpr_results2 = lpr_model.predict(frame2, conf=0.5, agnostic_nms=True)[0]
        lpr_results3 = lpr_model.predict(frame3, conf=0.5, agnostic_nms=True)[0]
        lpr_results4 = lpr_model.predict(frame4, conf=0.5, agnostic_nms=True)[0]

        detections = sv.Detections.from_yolov8(results)
        detections2 = sv.Detections.from_yolov8(results2)
        detections3 = sv.Detections.from_yolov8(results3)
        detections4 = sv.Detections.from_yolov8(results4)
        lpr_detections = sv.Detections.from_yolov8(lpr_results)
        lpr_detections2 = sv.Detections.from_yolov8(lpr_results2)
        lpr_detections3 = sv.Detections.from_yolov8(lpr_results3)
        lpr_detections4 = sv.Detections.from_yolov8(lpr_results4)

        for res in lpr_results:
            boxes = res.boxes
            for box in boxes:
                b = box.xyxy[0].numpy()
                print(b)

        if not os.path.exists("yolo_cropped/frame1") and not os.path.exists("yolo_cropped/frame2") \
        and not os.path.exists("yolo_cropped/frame3") and not os.path.exists("yolo_cropped/frame4"):
            os.makedirs("yolo_cropped/frame1")
            os.makedirs("yolo_cropped/frame2")
            os.makedirs("yolo_cropped/frame3")
            os.makedirs("yolo_cropped/frame4")

        # cv2.rectangle(frame, (0, 35), (frame.shape[1], frame.shape[0] - 40), color=(0, 0, 255), thickness=2)
        # cv2.rectangle(frame2, (0, 35), (frame2.shape[1], frame2.shape[0] - 40), color=(255, 0, 0), thickness=2)
        # cv2.rectangle(frame3, (0, 35), (frame3.shape[1], frame3.shape[0] - 40), color=(0, 255, 0), thickness=2)

        labels = [f"{model.model.names[class_id]} {confidence:.2f}" for i, confidence, class_id, j in detections]
        labels2 = [f"{model.model.names[class_id]} {confidence2:.2f}" for k, confidence2, class_id, l in detections2]
        labels3 = [f"{model.model.names[class_id]} {confidence3:.2f}" for m, confidence3, class_id, n in detections3]
        labels4 = [f"{model.model.names[class_id]} {confidence4:.2f}" for m, confidence4, class_id, n in detections4]
        lic_plate_labels = [f"{lpr_model.names[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in lpr_detections]
        lic_plate_labels2 = [f"{lpr_model.names[class_id]} {confidence2:.2f}" for _, confidence2, class_id, _ in lpr_detections2]
        lic_plate_labels3 = [f"{lpr_model.names[class_id]} {confidence3:.2f}" for _, confidence3, class_id, n in lpr_detections3]
        lic_plate_labels4 = [f"{lpr_model.names[class_id]} {confidence4:.2f}" for _, confidence4, class_id, _ in lpr_detections4]

        box_annotator = sv.BoxAnnotator(color=sv.Color.red(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
        box_annotator2 = sv.BoxAnnotator(color=sv.Color.blue(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
        box_annotator3 = sv.BoxAnnotator(color=sv.Color.green(), text_color=sv.Color.white(), text_scale=0.8, text_thickness=2)
        box_annotator4 = sv.BoxAnnotator(text_scale=0.8, text_thickness=2, text_padding=1)
        lpr_box_annotator = sv.BoxAnnotator(text_thickness=2, text_scale=0.8, text_color=sv.Color.white(), color=sv.Color.black())

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame2 = box_annotator2.annotate(scene=frame2, detections=detections2, labels=labels2)
        frame3 = box_annotator3.annotate(scene=frame3, detections=detections3, labels=labels3)
        frame4 = box_annotator4.annotate(scene=frame4, detections=detections4, labels=labels4)
        frame = lpr_box_annotator.annotate(scene=frame, detections=lpr_detections, labels=lic_plate_labels)
        frame2 = lpr_box_annotator.annotate(scene=frame2, detections=lpr_detections2, labels=lic_plate_labels2)
        frame3 = lpr_box_annotator.annotate(scene=frame3, detections=lpr_detections3, labels=lic_plate_labels3)
        frame4 = lpr_box_annotator.annotate(scene=frame4, detections=lpr_detections4, labels=lic_plate_labels4)

        blank_screen2[0: DIMS[0] // 2, 0: DIMS[1]] = np.hstack((frame, frame2))
        blank_screen2[DIMS[0] // 2: DIMS[0], 0: DIMS[1]] = np.hstack((frame3, frame4))

        cv2.imshow("detection", blank_screen2)
        vid_writer.write(blank_screen2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# annotate_objects_inside_frame()
annotate_objects_inside_videos()
# run_instance_segmentation_inside_videos()