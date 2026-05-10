#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import math
import time
import threading
import sys
import termios
import tty

class LidarSmartAvoider(Node):
    def __init__(self):
        super().__init__('lidar_smart_avoider_node')

        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.offset_pub = self.create_publisher(Float32, 'offset_value', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        
        # --- ZAKOMENTOWANE NASŁUCHIWANIE GUI ---
        # self.vision_sub = self.create_subscription(Float32, '/vision/offset_raw', self.vision_callback, 10)
        self.engine_enabled = True # Robot jedzie autonomicznie od samego startu!
        
        # --- PARAMETRY MANEWRU ---
        self.trigger_distance = 0.45      
        self.critical_distance = 0.16     
        self.target_side_distance = 0.35  
        self.kp_wall = 250.0              

        self.memory_ttl = 1.2  

        # --- PARAMETRY PŁYNNOŚCI (Kierownica) ---
        self.current_steer = 0.0          
        self.steering_alpha = 0.10        # Zmniejszono z 0.15 do 0.10 -> serwo będzie jeszcze bardziej leniwe/płynne
        self.base_swerve = 180.0          

        self.state = 'NORMAL' 
        self.state_start_time = time.time()

        self.front_dist = 999.0
        self.left_dist = 999.0
        self.right_dist = 999.0
        
        self.mem_l = {'dist': 999.0, 'time': 0.0}
        self.mem_r = {'dist': 999.0, 'time': 0.0}
        self.mem_f = {'dist': 999.0, 'time': 0.0}
        
        self.track_side = None 
        self.swerve_sign = 0.0  
        
        self.emergency_stop = False
        self.has_keyboard = False
        
        try:
            self.old_attr = termios.tcgetattr(sys.stdin)
            self.has_keyboard = True
        except termios.error:
            self.get_logger().warn("Uruchomiono w tle. Klawisz 'q' wyłączony.")

        if self.has_keyboard:
            self.key_thread = threading.Thread(target=self.keyboard_listener)
            self.key_thread.daemon = True
            self.key_thread.start()

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Gotowy do ŚLEPEGO SLALOMU! (Aktywny filtr szerokości obiektów)")

    def keyboard_listener(self):
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    self.emergency_stop = True
                    break
        finally:
            if self.has_keyboard:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attr)

    def scan_callback(self, msg):
        if self.emergency_stop or not self.engine_enabled:
            return 
            
        current_time = time.time()
        clusters = []
        current_cluster = []
        
        # 1. KLASTEROWANIE
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or r < 0.05 or r > 1.5: 
                continue
                
            angle = msg.angle_min + i * msg.angle_increment
            
            if not current_cluster:
                current_cluster.append((r, angle))
            else:
                last_r, last_angle = current_cluster[-1]
                dist = math.sqrt(r**2 + last_r**2 - 2*r*last_r*math.cos(angle - last_angle))
                if dist < 0.20:
                    current_cluster.append((r, angle))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(r, angle)]
        
        if current_cluster:
            clusters.append(current_cluster)

        # 2. FILTROWANIE ARTEFAKTÓW (Sprawdzamy szerokość fizyczną)
        valid_obstacles = []
        for c in clusters:
            # Warunek 1: Minimum 3 punkty pomiarowe z LiDARa
            if len(c) < 3:
                continue
                
            # Obliczanie fizycznej szerokości obiektu
            first_pt = c[0]
            last_pt = c[-1]
            width = math.sqrt(first_pt[0]**2 + last_pt[0]**2 - 2*first_pt[0]*last_pt[0]*math.cos(first_pt[1] - last_pt[1]))
            
            # Warunek 2: Obiekt musi mieć min. 4 cm szerokości (odrzuca kable i szum powitrza)
            if width < 0.04:
                continue
                
            # Wybieramy najbliższy punkt z PRAWDZIWEGO obiektu
            closest_pt = min(c, key=lambda x: x[0])
            valid_obstacles.append(closest_pt)
            
        self.publish_rviz_markers(valid_obstacles)

        # 3. ODZYSKIWANIE SUROWYCH ODLEGŁOŚCI (Teraz tylko z prawdziwych obiektów!)
        raw_f, raw_l, raw_r = 999.0, 999.0, 999.0
        decide_l, decide_r = 999.0, 999.0
        
        for r, angle in valid_obstacles:
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            if -0.35 < norm_angle < 0.35: raw_f = min(raw_f, r)
            if 1.0 < norm_angle < 2.2: raw_l = min(raw_l, r)
            if -2.2 < norm_angle < -1.0: raw_r = min(raw_r, r)
            if 0.1 < norm_angle < 0.8: decide_l = min(decide_l, r)
            if -0.8 < norm_angle < -0.1: decide_r = min(decide_r, r)

        # 4. PAMIĘĆ KRÓTKOTRWAŁA (Histereza)
        if raw_r < 1.0:
            self.mem_r = {'dist': raw_r, 'time': current_time}
        elif (current_time - self.mem_r['time']) < self.memory_ttl:
            raw_r = self.mem_r['dist']

        if raw_l < 1.0:
            self.mem_l = {'dist': raw_l, 'time': current_time}
        elif (current_time - self.mem_l['time']) < self.memory_ttl:
            raw_l = self.mem_l['dist']
            
        if raw_f < 1.0:
            self.mem_f = {'dist': raw_f, 'time': current_time}
        elif (current_time - self.mem_f['time']) < self.memory_ttl:
            raw_f = self.mem_f['dist']

        self.front_dist = raw_f
        self.left_dist = raw_l
        self.right_dist = raw_r

        # --- DYNAMICZNA MASZYNA STANÓW ---
        if self.state in ['NORMAL', 'SWERVE', 'PASSING']:
            if self.front_dist < self.critical_distance:
                self.get_logger().error("ZBYT BLISKO! Cofam...")
                self.state = 'REVERSE'
                self.state_start_time = time.time()
                return

        if self.state in ['SWERVE', 'PASSING', 'RETURN']:
            if self.front_dist < self.trigger_distance and (time.time() - self.state_start_time > 0.6):
                self.get_logger().warn(f"SLALOM! Zmiana toru jazdy.")
                if self.track_side == 'RIGHT':
                    self.track_side = 'LEFT'
                    self.swerve_sign = 1.0 
                else:
                    self.track_side = 'RIGHT'
                    self.swerve_sign = -1.0 
                self.state = 'SWERVE'
                self.state_start_time = time.time()
                return 

        if self.state == 'NORMAL':
            if self.front_dist < self.trigger_distance:
                if decide_l > decide_r:
                    self.track_side = 'RIGHT' 
                    self.swerve_sign = -1.0 
                else:
                    self.track_side = 'LEFT'  
                    self.swerve_sign = 1.0  
                self.state = 'SWERVE'
                self.state_start_time = time.time()
                
        elif self.state == 'SWERVE':
            if self.front_dist > 0.45:
                if self.track_side == 'RIGHT' and self.right_dist < 0.8:
                    self.state = 'PASSING'
                elif self.track_side == 'LEFT' and self.left_dist < 0.8:
                    self.state = 'PASSING'
                    
            if time.time() - self.state_start_time > 2.5:
                self.state = 'RETURN'
                self.state_start_time = time.time()
                
        elif self.state == 'PASSING':
            if self.track_side == 'RIGHT' and self.right_dist > 0.9:
                self.state = 'RETURN'
                self.state_start_time = time.time()
            elif self.track_side == 'LEFT' and self.left_dist > 0.9:
                self.state = 'RETURN'
                self.state_start_time = time.time()
                
        elif self.state == 'RETURN':
            if time.time() - self.state_start_time > 1.2:
                self.state = 'NORMAL'
                
        elif self.state == 'REVERSE':
            if time.time() - self.state_start_time > 1.5:
                self.state = 'NORMAL'

    def publish_rviz_markers(self, obstacles):
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for idx, (r, angle) in enumerate(obstacles):
            if r > 0.70:
                continue 
                
            m = Marker()
            m.header.frame_id = "lidar_frame"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "cones"
            m.id = idx + 1 
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            
            m.pose.position.x = r * math.cos(angle)
            m.pose.position.y = r * math.sin(angle)
            m.pose.position.z = 0.0
            
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.a = 1.0
            
            if r <= 0.40:
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0
            else:
                m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
                
            marker_array.markers.append(m)
            
        self.marker_pub.publish(marker_array)

    def control_loop(self):
        msg = Float32()
        target_steer = 0.0

        if self.emergency_stop or not self.engine_enabled:
            msg.data = 999.0
            self.current_steer = 0.0 
            self.offset_pub.publish(msg)
            return

        if self.state == 'NORMAL':
            target_steer = 0.0 
        elif self.state == 'REVERSE':
            target_steer = 888.0 
        elif self.state == 'SWERVE':
            urgency = self.trigger_distance / max(self.front_dist, 0.05)
            urgency = min(urgency, 1.8) 
            target_steer = self.swerve_sign * self.base_swerve * urgency
        elif self.state == 'PASSING':
            if self.track_side == 'RIGHT':
                error = self.right_dist - self.target_side_distance
                correction = error * self.kp_wall 
            else:
                error = self.target_side_distance - self.left_dist
                correction = error * self.kp_wall 
            target_steer = max(-220.0, min(220.0, correction))
        elif self.state == 'RETURN':
            target_steer = -self.swerve_sign * (self.base_swerve * 0.5)

        # Filtr Dolnoprzepustowy (Jeszcze łagodniejszy powrót)
        if self.state != 'REVERSE':
            self.current_steer = self.current_steer + self.steering_alpha * (target_steer - self.current_steer)
        else:
            self.current_steer = 888.0

        msg.data = float(self.current_steer)
        self.offset_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarSmartAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, 'has_keyboard', False):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.old_attr)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()