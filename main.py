from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

""" Loading Models"""
anomaly_model = YOLO('runs/detect/train/weights/best.pt')
persons_model = YOLO('yolov8m.pt')


""" Function to get the number of each PPE"""
def get_number_of_each_ppe(results):
    Boxes = results[0].boxes

    helmet_number = 0
    goggle_number = 0
    vest_number = 0
    boots_number = 0
    for box in Boxes:
        cls = int(box.cls[0].item())
        if cls == 0:
            helmet_number += 1
        elif cls == 1:
            goggle_number += 1
        elif cls == 2:
            vest_number += 1
        else:
            boots_number += 1
    return {"Helmet": helmet_number, "Goggles": goggle_number, "Vest": vest_number, "Safety boots": boots_number}


""" Create windows and set positions"""
cv2.namedWindow('Easy-Detect')
cv2.namedWindow('Persons', )

display_width, display_height = 1536, 864
#resized_image = cv2.resize(image, (new_width, new_height))


cv2.moveWindow('Easy-Detect', x=display_width//2, y=0)
cv2.moveWindow('Persons', x=display_width//2, y=display_height//2)


""" Start detections on video"""
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

initial_time = datetime.now()
time_interval = timedelta(seconds=5)

while cap.isOpened():
    succes, frame = cap.read()
    if succes:
        frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
        frame = cv2.flip(frame, 1)
        anomaly_results = anomaly_model.predict(frame, conf=0.5, verbose=False)
        persons_results = persons_model.predict(frame, classes=[0], verbose=False)

        annotated_anomaly_frame = anomaly_results[0].plot()
        annotated_persons_frame = persons_results[0].plot()

        number_of_persons = len(persons_results[0].boxes)
        number_of_each_ppe = get_number_of_each_ppe(anomaly_results)

        print(f'{number_of_persons} Persons detected')
        print(f'Number of PPEs detected: {number_of_each_ppe}')

        current_time = datetime.now()
        if current_time >= initial_time + time_interval:
            if number_of_persons:
                anomaly_detection = False
                for ppe, ppe_number in number_of_each_ppe.items():
                    if ppe_number < number_of_persons:
                        anomaly_detection = True
                        print(f'{number_of_persons - ppe_number} person(s) are without {ppe}')
                        time_interval = timedelta(seconds=5)
                if anomaly_detection:
                    now_time = datetime.now().strftime("%Y-%m-%d %Hh%Mm")
                    cv2.imwrite(f'anomaly/detection/{now_time}.png', frame)
            else:
                time_interval = timedelta(seconds=0)
            initial_time = current_time

        cv2.imshow('Easy-Detect', annotated_anomaly_frame)
        cv2.imshow('Persons', annotated_persons_frame)

        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()