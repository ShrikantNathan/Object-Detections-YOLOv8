import numpy as np
import ultralytics
import os
import cv2
from ultralytics.yolo.utils.plotting import Annotator

IMAGE_RES = (1080, 1920)
model = ultralytics.YOLO("yolov8x.pt")
segmodel = ultralytics.YOLO("yolov8x-seg.pt")
blank_screen = np.zeros(IMAGE_RES, dtype=np.uint8)

images = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
images2 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
images3 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))
images4 = np.random.choice(list(image for image in os.listdir(os.path.join(os.getcwd(), "Images"))))

image1_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images)), (0, 0), interpolation=cv2.INTER_LINEAR)
image2_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images2)), (0, 0), interpolation=cv2.INTER_LINEAR)
image3_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images3)), (0, 0), interpolation=cv2.INTER_LINEAR)
image4_reset = cv2.resize(cv2.imread(os.path.join(os.getcwd(), "Images", images4)), (0, 0), interpolation=cv2.INTER_LINEAR)

results = model.predict(image1_reset, save=True)
results2 = model.predict(image2_reset, save=True)
results3 = model.predict(image3_reset, save=True)
results4 = model.predict(image4_reset, save=True)

res_stacked_1 = np.hstack((results, results2))
res_stacked_2 = np.hstack((results3, results4))
blank_screen[0: IMAGE_RES[0] // 2, 0: IMAGE_RES[1]] = res_stacked_1
blank_screen[IMAGE_RES[0] // 2: IMAGE_RES[0], 0: IMAGE_RES[1]] = res_stacked_2
cv2.imshow('results', blank_screen)
cv2.waitKey(0)