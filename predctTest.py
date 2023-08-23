from ultralytics import YOLO
import cv2


model = YOLO('runs/detect/train/weights/best.pt')

# LOADING AN IMAGE
img1 = cv2.imread('test/img1.jpg')
img2 = cv2.imread('test/img2.jpg')
img3 = cv2.imread('test/img3.jpg')
#img4 = cv2.imread('test/img4.png')

results = model.predict(img1, show=True, verbose=False)

cv2.waitKey(0)
