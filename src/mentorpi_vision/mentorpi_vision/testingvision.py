#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

class RealCameraTest(Node):
    def __init__(self):
        super().__init__('camera_test_node')
        
        # Otwieramy domyślną kamerę robota podpiętą pod USB / taśmę CSI
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error("BŁĄD: Nie można połączyć się z kamerą! Sprawdź taśmę/USB.")
            return

        self.get_logger().info("SUKCES: Połączono z kamerą sprzętową robota!")
        
        # Pętla sprawdzająca obraz co 0.5 sekundy
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            # Pobieramy wymiary klatki
            wysokosc, szerokosc, _ = frame.shape
            
            # Obliczamy średnią jasność (żeby udowodnić, że obraz nie jest tylko czarną plamą)
            srednia_jasnosc = int(np.mean(frame))
            
            self.get_logger().info(f"KAMERA DZIAŁA -> Rozdzielczość: {szerokosc}x{wysokosc} | Średnia jasność pikseli: {srednia_jasnosc}")
        else:
            self.get_logger().warning("Kamera podłączona, ale zgubiła klatkę!")

def main(args=None):
    rclpy.init(args=args)
    node = RealCameraTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Kulturalne zamykanie kamery
    if hasattr(node, 'cap') and node.cap.isOpened():
        node.cap.release()
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()