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
        
        self.base_speed = 0.12
        self.min_speed = 0.06
        
        # --- TUNING PID (NAPRAWA WĘŻYKOWANIA) ---
        # kp obniżone z 0.008 na 0.0055 (Mniej nerwowy skręt)
        self.kp = 0.0055 
        # kd podbite z 0.003 na 0.007 (Mocny "amortyzator", kontruje przy rozbujaniu)
        self.kd = 0.007  
        
        self.last_offset = 0.0

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

        error_diff = offset - self.last_offset
        steering_output = (offset * self.kp) + (error_diff * self.kd)
        
        twist.angular.z = -steering_output 

        curve_factor = abs(offset) * 0.0012
        dynamic_speed = self.base_speed - curve_factor
        
        twist.linear.x = max(self.min_speed, dynamic_speed)

        self.last_offset = offset
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