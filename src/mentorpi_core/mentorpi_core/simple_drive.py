#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import time  # <-- Ważny import do opóźnienia przy wyłączaniu

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
        
        self.base_speed = 0.20
        self.min_speed = 0.06
        self.kp = 0.008
        self.kd = 0.003
        self.last_offset = 0.0

    def vision_callback(self, msg):
        twist = Twist()
        offset = msg.data 

        if offset == 999.0:
            twist.linear.x = 0.0
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
        # --- SEKWENCJA AWARYJNEGO ZATRZYMANIA ---
        node.get_logger().info("Stopping motors..")
        
        emergency_stop = Twist()
        emergency_stop.linear.x = 0.0
        emergency_stop.angular.z = 0.0 
        
        node.cmd_vel_pub.publish(emergency_stop)
        
        
        time.sleep(0.2) 
        
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()