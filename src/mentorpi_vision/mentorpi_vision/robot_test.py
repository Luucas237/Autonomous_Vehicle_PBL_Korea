import torch
import cv2
import pathlib
import sys

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path = 'best14.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)

cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam_id)

print(f"Uruchamiam detekcję na kamerze ID: {cam_id}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można odebrać klatki z kamery.")
        break
        
    results = model(frame)
    results.conf = 0.30
    
    rendered_frame = results.render()[0]
    
    cv2.imshow('Detekcja Znakow - MentorPi', rendered_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
