import pathlib
import platform
import torch
import cv2

current_system = platform.system()
if current_system == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best14.pt')

model.conf = 0.30  #odczytuje z wieksza pewnoscia
model.iou = 0.45 
model.multi_label = False 

import sys
camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0 #pobiera kamere urzadzenia
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"BŁĄD: Nie można otworzyć kamery o ID: {camera_id}")
    exit()

print(f"Klasy modelu: {model.names}")
print("Naciśnij 'q', aby zamknąć okno.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można pobrać klatki.")
        break

    results = model(frame)

    annotated_frame = results.render()[0]


    cv2.imshow('Wizja Projektu Julii', annotated_frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
