#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data  
import math
import time
import threading
import sys
import termios
import tty

class LidarSmartAvoider(Node):
    def __init__(self):
        super().__init__('lidar_smart_avoider_node')

        # NASŁUCHIWANIE
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.gui_cmd_subscriber = self.create_subscription(Float32, '/vision/offset_raw', self.gui_cmd_callback, qos_profile_sensor_data)
        self.vision_sub = self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.telemetry_callback, qos_profile_sensor_data)
        
        # PUBLIKACJA DO KÓŁ (JEDYNY NADAJNIK W SYSTEMIE)
        self.offset_pub = self.create_publisher(Float32, 'offset_value', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        
        self.engine_enabled = False 
        self.last_engine_state = False
        
        # DANE Z KAMERY (Fuzja)
        self.camera_desired_offset = 0.0
        self.camera_bumper_active = False

        self.trigger_distance = 0.45      
        self.critical_distance = 0.16     
        self.steering_alpha = 0.25 
        self.current_steer = 0.0

        # --- NOWOŚĆ: SYSTEM POTWIERDZANIA PRZESZKODY (DEBOUNCING) ---
        self.obstacle_confirm_counter = 0
        self.required_confirmations = 4   # Przeszkoda musi być widoczna przez 4 skany z rzędu (ok. 0.3 - 0.4 sekundy)

        self.state = 'NORMAL' 
        self.state_start_time = time.time()
        self.swerve_dir = 0.0

        self.front_dist, self.left_dist, self.right_dist = 999.0, 999.0, 999.0
        self.emergency_stop = False
        self.has_keyboard = False
        
        try:
            self.old_attr = termios.tcgetattr(sys.stdin)
            self.has_keyboard = True
        except termios.error:
            self.get_logger().warn("Uruchomiono w tle.")

        if self.has_keyboard:
            self.key_thread = threading.Thread(target=self.keyboard_listener)
            self.key_thread.daemon = True
            self.key_thread.start()

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Avoider [MASTER]: DATA FUSION + DEBOUNCING AKTYWNY!")

    def gui_cmd_callback(self, msg):
        if msg.data == 999.0:
            self.engine_enabled = False
            self.current_steer = 0.0 
        else:
            self.engine_enabled = True

        if self.engine_enabled != self.last_engine_state:
            if self.engine_enabled: self.get_logger().info(">>> MASTER: ENGINES ON! <<<")
            else: self.get_logger().warn(">>> MASTER: ENGINES OFF (STOP) <<<")
            self.last_engine_state = self.engine_enabled

    def telemetry_callback(self, msg):
        self.camera_desired_offset = msg.data[8]
        self.camera_bumper_active = (msg.data[9] == 1.0)

    def keyboard_listener(self):
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    self.emergency_stop = True
                    break
        finally:
            if self.has_keyboard: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attr)

    def scan_callback(self, msg):
        if self.emergency_stop or not self.engine_enabled: return 
            
        clusters = []
        current_cluster = []
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or r < 0.05 or r > 1.5: continue
            angle = msg.angle_min + i * msg.angle_increment
            if not current_cluster: current_cluster.append((r, angle))
            else:
                last_r, last_angle = current_cluster[-1]
                if math.sqrt(r**2 + last_r**2 - 2*r*last_r*math.cos(angle - last_angle)) < 0.20:
                    current_cluster.append((r, angle))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(r, angle)]
        if current_cluster: clusters.append(current_cluster)

        valid_obstacles = []
        for c in clusters:
            # FILTR PRZESTRZENNY (Wymiarowy)
            if len(c) < 3: continue
            first_pt, last_pt = c[0], c[-1]
            if math.sqrt(first_pt[0]**2 + last_pt[0]**2 - 2*first_pt[0]*last_pt[0]*math.cos(first_pt[1] - last_pt[1])) < 0.04: continue
            valid_obstacles.append(min(c, key=lambda x: x[0]))
            
        self.publish_rviz_markers(valid_obstacles)

        raw_f, decide_l, decide_r = 999.0, 999.0, 999.0
        for r, angle in valid_obstacles:
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            if -0.35 < norm_angle < 0.35: raw_f = min(raw_f, r)
            if 0.1 < norm_angle < 0.8: decide_l = min(decide_l, r)
            if -0.8 < norm_angle < -0.1: decide_r = min(decide_r, r)

        self.front_dist = raw_f
        current_time = time.time()

        # ==========================================
        # MASZYNA STANÓW: Fuzja Danych i Zmiana Pasa
        # ==========================================
        if self.state in ['LANE_CHANGE_OUT', 'LANE_CHANGE_ALIGN']:
            if self.front_dist < self.critical_distance:
                self.get_logger().error("AWARYJNE COFANIE!")
                self.state = 'REVERSE'
                self.state_start_time = current_time
                return

        if self.state == 'NORMAL':
            # --- FILTR CZASOWY (Debouncing) ---
            if self.front_dist < self.trigger_distance:
                self.obstacle_confirm_counter += 1
                
                if self.obstacle_confirm_counter >= self.required_confirmations:
                    self.get_logger().warn(f"PRZESZKODA POTWIERDZONA ({self.required_confirmations} skanów)! Rozpoczynam ZMIANĘ PASA!")
                    
                    if decide_l > decide_r:
                        self.swerve_dir = 280.0  
                    else:
                        self.swerve_dir = -280.0 
                        
                    self.state = 'LANE_CHANGE_OUT'
                    self.state_start_time = current_time
                    self.obstacle_confirm_counter = 0 # Reset po udanym triggerze
            else:
                # Jeśli przeszkoda znika choć na chwilę, zerujemy licznik 
                # (eliminuje to kumulację pojedynczych artefaktów w czasie)
                if self.obstacle_confirm_counter > 0:
                    self.obstacle_confirm_counter = 0
                
        elif self.state == 'LANE_CHANGE_OUT':
            if current_time - self.state_start_time > 0.8:
                self.state = 'LANE_CHANGE_ALIGN'
                self.state_start_time = current_time
                
        elif self.state == 'LANE_CHANGE_ALIGN':
            if current_time - self.state_start_time > 0.5:
                self.get_logger().info("Manewr zakończony. Kamera przejmuje kontrolę w nowym pasie.")
                self.state = 'NORMAL'
                self.obstacle_confirm_counter = 0 # Zabezpieczenie zerowania
                
        elif self.state == 'REVERSE':
            if current_time - self.state_start_time > 1.5:
                self.state = 'NORMAL'
                self.obstacle_confirm_counter = 0

    def publish_rviz_markers(self, obstacles):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        for idx, (r, angle) in enumerate(obstacles):
            if r > 0.70: continue 
            m = Marker()
            m.header.frame_id = "lidar_frame"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "cones", idx + 1, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = r * math.cos(angle), r * math.sin(angle), 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.1
            m.color.a = 1.0
            if r <= 0.40: m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            else: m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    def control_loop(self):
        msg = Float32()
        target_steer = 0.0

        if self.emergency_stop or not self.engine_enabled:
            self.current_steer = 0.0 
            msg.data = 999.0
            self.offset_pub.publish(msg)
            return

        # ==========================================
        # WYKONANIE FUZJI (Master-Slave)
        # ==========================================
        if self.state == 'NORMAL':
            target_steer = self.camera_desired_offset
            
        elif self.state == 'REVERSE':
            target_steer = 888.0 
            
        elif self.state == 'LANE_CHANGE_OUT':
            target_steer = self.swerve_dir
            
        elif self.state == 'LANE_CHANGE_ALIGN':
            target_steer = -self.swerve_dir * 0.35

        if self.state != 'REVERSE':
            self.current_steer = self.current_steer + self.steering_alpha * (target_steer - self.current_steer)
        else:
            self.current_steer = 888.0

        msg.data = float(self.current_steer)
        self.offset_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarSmartAvoider()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if getattr(node, 'has_keyboard', False): termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.old_attr)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()