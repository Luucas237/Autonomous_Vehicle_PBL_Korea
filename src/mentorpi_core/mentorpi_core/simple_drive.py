#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class SimpleDriveController(Node):
    def __init__(self):
        super().__init__('simple_drive_node')
        
        # Subskrybujemy dane z wizji
        self.offset_sub = self.create_subscription(
            Float32,
            'offset_value',
            self.vision_callback,
            10
        )
        
        # Publikujemy prędkość na standardowy temat ROS 2 do silników
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('Sterownik jazdy uruchomiony. Czekam na komendy z wizji...')

    def vision_callback(self, msg):
        twist = Twist()
        
        # Jeśli wizja opublikowała 999.0, to znaczy, że nie widzi linii
        if msg.data == 999.0:
            twist.linear.x = 0.0  # ZATRZYMAJ SIĘ
            twist.angular.z = 0.0 # NIE SKRĘCAJ
            self.get_logger().info('Brak linii -> STOP')
        else:
            # Widzi linię! Jedziemy powoli do przodu (0.15 m/s)
            twist.linear.x = 0.15  
            twist.angular.z = 0.0 # Na razie nie skręcamy, tak jak prosiłeś
            self.get_logger().info('Widzę linię -> JAZDA DO PRZODU')
            
        # Wysyłamy komendę do fizycznych silników
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Zatrzymanie awaryjne przy wyłączaniu skryptu (Ctrl+C)
        emergency_stop = Twist()
        emergency_stop.linear.x = 0.0
        emergency_stop.angular.z = 0.0
        node.cmd_vel_pub.publish(emergency_stop)
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()