#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class RealCameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_test_node')
        
        # Narzędzie do tłumaczenia obrazu z ROS 2 na format OpenCV
        self.bridge = CvBridge()
        
        # Nazwa tematu (topic) - dostosujemy ją, jeśli producent nazwał go inaczej
        self.topic_name = '/ascamera/camera_publisher/rgb0/image' 
        
        # Tworzymy subskrybenta
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10)
        
        self.get_logger().info(f"SUKCES: Węzeł uruchomiony. Oczekuję na obraz na temacie: {self.topic_name}")
        
        # Zmienna do kontrolowania częstotliwości logów (żeby nie zaspamować terminala)
        self.last_log_time = time.time()

    def image_callback(self, msg):
        try:
            # Magia: konwertujemy wiadomość ROS 2 na klatkę OpenCV (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu: {e}")
            return

        # Logujemy dane tylko raz na sekundę, choć obraz odbieramy np. 30 razy na sekundę
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            wysokosc, szerokosc, _ = cv_image.shape
            srednia_jasnosc = int(np.mean(cv_image))
            
            self.get_logger().info(f"KAMERA DZIAŁA -> Rozdzielczość: {szerokosc}x{wysokosc} | Średnia jasność: {srednia_jasnosc}")
            self.last_log_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = RealCameraSubscriber()
    try:
        rclpy.spin(node) # Węzeł kręci się w nieskończoność i czeka na wiadomości
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()