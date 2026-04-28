import pathlib
import platform
import torch
import cv2

# Naprawa błędu ścieżek między Linuxem a Windowsem
current_system = platform.system()
if current_system == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath

# Ładowanie modelu
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best14.pt')

# Parametry detekcji
model.conf = 0.30  # Zwiększyłem z 0.20 na 0.30, żeby było stabilniej
model.iou = 0.45 
model.multi_label = False 

cap = cv2.VideoCapture(0)

# Sprawdzenie czy kamera w ogóle działa
if not cap.isOpened():
    print("BŁĄD: Nie można otworzyć kamery. Sprawdź czy inna aplikacja jej nie używa!")
    exit()

print(f"Klasy modelu: {model.names}")
print("Naciśnij 'q', aby zamknąć okno.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Nie można pobrać klatki.")
        break

    # Detekcja
    results = model(frame)

    # Rysowanie wyników na klatce
    # render() zwraca listę obrazów numpy, bierzemy pierwszy [0]
    annotated_frame = results.render()[0]

    # Wyświetlanie
    cv2.imshow('Wizja Projektu Julii', annotated_frame)

    # Wyjście ze skryptu po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()