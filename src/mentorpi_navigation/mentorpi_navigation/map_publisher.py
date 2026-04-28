#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

class MakietaMapPublisher(Node):
    def __init__(self):
        super().__init__('makieta_map_publisher')
        # Publikujemy na topic /makieta_map
        self.map_pub = self.create_publisher(OccupancyGrid, '/makieta_map', 10)
        
        try:
            package_dir = get_package_share_directory('mentorpi_navigation')
            self.image_path = os.path.join(package_dir, 'resource', 'makieta.png')
        except Exception as e:
            self.get_logger().error(f"Nie znaleziono paczki: {e}")
            return

        self.timer = self.create_timer(2.0, self.publish_map)
        self.get_logger().info("Publikuje dywan makiety 3D (Topic: /makieta_map)")

    def publish_map(self):
        if not os.path.exists(self.image_path):
            return

        # Wczytanie obrazka w skali szarości
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
            
        # Odwrócenie osi Y (ROS 2 liczy Y w drugą stronę niż grafika komputerowa)
        img = cv2.flip(img, 0)
        
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map" # Przyspawanie do głównej siatki RViz
        
        # SKALA MAKIETY: Ile metrów ma jeden piksel. 
        # Zmień tę wartość, aby powiększyć/pomniejszyć makietę na podłodze!
        map_msg.info.resolution = 0.005 # (1 piksel = 5 mm)
        map_msg.info.width = img.shape[1]
        map_msg.info.height = img.shape[0]
        
        # Centrowanie na środku siatki RViz (punkt 0,0)
        map_msg.info.origin.position.x = - (img.shape[1] * map_msg.info.resolution) / 2.0
        map_msg.info.origin.position.y = - (img.shape[0] * map_msg.info.resolution) / 2.0
        map_msg.info.origin.position.z = 0.0 
        
        # Zamiana pikseli na dane Mapy (0 = tło makiety, 100 = czarne linie)
        ros_data = np.zeros_like(img, dtype=np.int8)
        ros_data[img < 128] = 100  # Czarne linie obrazka
        ros_data[img >= 128] = 0   # Białe tło obrazka
        
        map_msg.data = ros_data.flatten().tolist()
        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MakietaMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()