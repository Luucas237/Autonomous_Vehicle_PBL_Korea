#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import time

class SimpleDriveController(Node):
    def __init__(self):
        super().__init__('simple_drive_node')
        
        self.offset_sub = self.create_subscription(
            Float32,
            'offset_value',
            self.vision_callback,
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.base_speed = 0.10
        self.min_speed = 0.055
        
        # --- TUNING: spokojniejsze trzymanie pasa ---
        # kp mniejsze = mniej nerwowa reakcja na błąd pozycji.
        self.kp = 0.0038
        # kd mniejsze i liczone na przefiltrowanym błędzie = tłumienie, bez szarpania.
        self.kd = 0.0012
        
        self.last_offset = 0.0
        self.filtered_offset = 0.0
        self.offset_alpha = 0.25       # filtr offsetu z kamery
        self.max_offset_step = 18.0    # limit zmiany offsetu na wiadomość [px]
        self.max_angular = 0.75        # limit skrętu rad/s
        self.max_angular_step = 0.08   # limit zmiany skrętu między wiadomościami
        self.last_angular = 0.0

    def vision_callback(self, msg):
        twist = Twist()
        offset = msg.data 

        if offset == 999.0: # Awaryjny stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        # Bieg wsteczny
        if offset == 888.0: 
            twist.linear.x = -0.15 
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        # Ograniczamy nagłe skoki offsetu z kamery. To jest główna ochrona
        # przed przejściem z pełnego skrętu w prawo na pełny skręt w lewo.
        offset_delta = offset - self.filtered_offset
        if offset_delta > self.max_offset_step:
            offset_delta = self.max_offset_step
        elif offset_delta < -self.max_offset_step:
            offset_delta = -self.max_offset_step

        stepped_offset = self.filtered_offset + offset_delta
        self.filtered_offset = (1.0 - self.offset_alpha) * self.filtered_offset + self.offset_alpha * stepped_offset

        error_diff = self.filtered_offset - self.last_offset
        steering_output = (self.filtered_offset * self.kp) + (error_diff * self.kd)

        target_angular = -steering_output
        target_angular = max(-self.max_angular, min(self.max_angular, target_angular))

        # Rate limiter na samym skręcie robota. Nawet gdy offset skoczy, koła
        # dochodzą do nowego skrętu stopniowo.
        angular_delta = target_angular - self.last_angular
        if angular_delta > self.max_angular_step:
            angular_delta = self.max_angular_step
        elif angular_delta < -self.max_angular_step:
            angular_delta = -self.max_angular_step

        twist.angular.z = self.last_angular + angular_delta

        curve_factor = abs(self.filtered_offset) * 0.0009
        dynamic_speed = self.base_speed - curve_factor
        
        twist.linear.x = max(self.min_speed, dynamic_speed)

        self.last_offset = self.filtered_offset
        self.last_angular = twist.angular.z
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriveController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Wykryto Ctrl+C! Wymuszam zatrzymanie silników...")
        
        emergency_stop = Twist()
        emergency_stop.linear.x = 0.0
        emergency_stop.angular.z = 0.0 
        
        for _ in range(3):
            node.cmd_vel_pub.publish(emergency_stop)
            rclpy.spin_once(node, timeout_sec=0.1)
            
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()