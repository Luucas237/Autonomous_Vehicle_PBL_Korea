#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import math
import time

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider_node')

        # 1. Nasłuchiwanie wizji (to, co widzi kamera)
        self.vision_sub = self.create_subscription(Float32, '/vision/offset_raw', self.vision_callback, 10)
        
        # 2. Nasłuchiwanie LiDARa
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)

        # 3. Publikowanie DO SILNIKÓW (to, na co reaguje PID z mentorpi_core)
        self.offset_pub = self.create_publisher(Float32, 'offset_value', 10)
        
        # 4. Publikowanie do RViz (wizualizacja pachołka)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_marker', 10)

        # Parametry omijania
        self.trigger_distance = 0.25      # [metry] Dystans wykrycia przeszkody (25 cm)
        self.swerve_offset = -350.0       # Sztuczny offset wymuszający skręt (np. -350 to ostro w prawo)
        self.swerve_duration = 1.0        # Ile sekund skręcać w bok
        self.pass_duration = 0.8          # Ile sekund jechać prosto omijając
        self.return_duration = 1.0        # Ile sekund wracać na tor (przeciwny offset)

        # Maszyna stanów (State Machine)
        self.state = 'NORMAL'  # Stany: NORMAL, SWERVE_OUT, PASSING, RETURN
        self.state_start_time = 0.0
        self.last_vision_offset = 0.0

        self.get_logger().info("Obstacle Avoider Node started! State: NORMAL")

    def vision_callback(self, msg):
        # Aktualizujemy to, co widzi kamera
        self.last_vision_offset = msg.data

        # Jeśli nie omijamy przeszkody, natychmiast przekazujemy sygnał z kamery do kół
        if self.state == 'NORMAL':
            self.offset_pub.publish(msg)

    def scan_callback(self, msg):
        # Sprawdzamy przestrzeń TYLKO Z PRZODU robota (zakładamy, że 0 stopni to przód)
        # Skanujemy +/- 25 stopni od frontu
        min_distance = 999.0
        
        for i, range_val in enumerate(msg.ranges):
            # Obliczenie kąta dla danego pomiaru
            angle = msg.angle_min + i * msg.angle_increment
            
            # Normalizacja kąta do przedziału -pi do pi
            angle = math.atan2(math.sin(angle), math.cos(angle))
            
            # Jeśli kąt jest z przodu robota (między -0.4 a 0.4 radiana)
            if -0.4 < angle < 0.4:
                # Ignorujemy błędy pomiaru (0.0 lub nieskończoność)
                if 0.05 < range_val < min_distance and not math.isinf(range_val):
                    min_distance = range_val

        # Publikacja markera dla podglądu w RViz
        self.publish_rviz_marker(min_distance)

        # LOGIKA MASZYNY STANÓW
        current_time = time.time()

        if self.state == 'NORMAL':
            if min_distance < self.trigger_distance:
                self.get_logger().warn(f"PRZESZKODA WYKRYTA na {min_distance:.2f}m! Rozpoczynam omijanie.")
                self.state = 'SWERVE_OUT'
                self.state_start_time = current_time

        elif self.state == 'SWERVE_OUT':
            # Publikujemy sztuczny potężny offset, by wymusić zjazd z linii
            self.publish_fake_offset(self.swerve_offset)
            if current_time - self.state_start_time > self.swerve_duration:
                self.get_logger().info("Jazda na wprost wzdłuż przeszkody.")
                self.state = 'PASSING'
                self.state_start_time = current_time

        elif self.state == 'PASSING':
            # Wyzerowanie offsetu (jazda w miarę prosto po łuku)
            self.publish_fake_offset(0.0)
            if current_time - self.state_start_time > self.pass_duration:
                self.get_logger().info("Powrót na tor linii.")
                self.state = 'RETURN'
                self.state_start_time = current_time

        elif self.state == 'RETURN':
            # Odwrotny offset, by wrócić na kamerę
            self.publish_fake_offset(-self.swerve_offset)
            if current_time - self.state_start_time > self.return_duration:
                self.get_logger().info("Omijanie zakończone. Powrót do wizji kamery.")
                self.state = 'NORMAL'

    def publish_fake_offset(self, fake_value):
        msg = Float32()
        msg.data = fake_value
        self.offset_pub.publish(msg)

    def publish_rviz_marker(self, distance):
        marker = Marker()
        marker.header.frame_id = "lidar_frame" # LUB 'base_link' - zależy od ramy Twojego lidara!
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "obstacle"
        marker.id = 0
        marker.type = Marker.SPHERE
        
        # Jeśli nic nie ma, rysujemy szarą przezroczystą kulę daleko
        if distance > 2.0:
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD
            marker.pose.position.x = float(distance)
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Jeśli jest bliżej niż trigger_distance, niech świeci na CZERWONO, inaczej na ZIELONO
            marker.color.a = 1.0
            if distance < self.trigger_distance:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()