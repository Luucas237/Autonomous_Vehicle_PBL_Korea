echo "--- Przygotowanie środowiska wizji ---"

pip install --upgrade pip
pip install torch torchvision opencv-python pandas requests psutil matplotlib

echo "--- Uruchamiam skrypt detekcji ---"
# Domyślnie uruchamia kamerę 0, ale można zmienić wpisując np. ./uruchom.sh 1
python3 robot_test.py $1