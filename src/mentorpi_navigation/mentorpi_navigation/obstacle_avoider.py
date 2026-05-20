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

        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.vision_sub = self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.telemetry_callback, qos_profile_sensor_data)
        self.gui_cmd_subscriber = self.create_subscription(Float32, '/vision/offset_raw', self.gui_cmd_callback, qos_profile_sensor_data)
        
        self.offset_pub = self.create_publisher(Float32, 'offset_value', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        
        self.engine_enabled = True 
        self.last_engine_state = True
        self.camera_desired_offset = 0.0
        
        # --- ZMIENNE LINII CIĄGŁEJ (Odczyt z kamery) ---
        self.cam_left_px = 0.0
        self.cam_right_px = 0.0
        # PRÓG LINII CIĄGŁEJ: Jeśli pod kołem jest więcej niż 2800 białych pikseli, to znaczy że linia jest gruba/ciągła (zakaz przekraczania)
        self.solid_line_px = 2800.0 
        
        self.trigger_distance = 0.45      
        self.critical_distance = 0.16     
        self.target_side_distance = 0.35  
        self.kp_wall = 250.0              

        self.memory_ttl = 1.2  

        self.current_steer = 0.0          
        self.base_swerve = 220.0 # Trochę mocniejszy skręt początkowy

        self.state = 'NORMAL' 
        self.state_start_time = time.time()

        self.front_dist, self.left_dist, self.right_dist = 999.0, 999.0, 999.0
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
            self.get_logger().warn("Uruchomiono w tle.")

        if self.has_keyboard:
            self.key_thread = threading.Thread(target=self.keyboard_listener)
            self.key_thread.daemon = True
            self.key_thread.start()

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("MASTER GOTOWY: Detekcja Linii Ciągłej włączona!")

    def telemetry_callback(self, msg):
        # Pobieranie offsetu ORAZ liczby pikseli pod kołami (dane z zderzaka kamery)
        if len(msg.data) >= 12:
            self.camera_desired_offset = msg.data[8]
            self.cam_left_px = msg.data[10]
            self.cam_right_px = msg.data[11]

    def gui_cmd_callback(self, msg):
        if msg.data == 999.0:
            self.engine_enabled = False
            self.current_steer = 0.0 
        else:
            self.engine_enabled = True

        if self.engine_enabled != self.last_engine_state:
            if self.engine_enabled: self.get_logger().info(">>> ENGINES ON <<<")
            else: self.get_logger().warn(">>> ENGINES OFF <<<")
            self.last_engine_state = self.engine_enabled

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
            
        current_time = time.time()
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
            if len(c) < 3: continue
            first_pt, last_pt = c[0], c[-1]
            if math.sqrt(first_pt[0]**2 + last_pt[0]**2 - 2*first_pt[0]*last_pt[0]*math.cos(first_pt[1] - last_pt[1])) < 0.04: continue
            valid_obstacles.append(min(c, key=lambda x: x[0]))
            
        self.publish_rviz_markers(valid_obstacles)

        raw_f, raw_l, raw_r, decide_l, decide_r = 999.0, 999.0, 999.0, 999.0, 999.0
        
        for r, angle in valid_obstacles:
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            if -0.35 < norm_angle < 0.35: raw_f = min(raw_f, r)
            if 1.0 < norm_angle < 2.2: raw_l = min(raw_l, r)
            if -2.2 < norm_angle < -1.0: raw_r = min(raw_r, r)
            if 0.1 < norm_angle < 0.8: decide_l = min(decide_l, r)
            if -0.8 < norm_angle < -0.1: decide_r = min(decide_r, r)

        if raw_r < 1.0: self.mem_r = {'dist': raw_r, 'time': current_time}
        elif (current_time - self.mem_r['time']) < self.memory_ttl: raw_r = self.mem_r['dist']

        if raw_l < 1.0: self.mem_l = {'dist': raw_l, 'time': current_time}
        elif (current_time - self.mem_l['time']) < self.memory_ttl: raw_l = self.mem_l['dist']
            
        if raw_f < 1.0: self.mem_f = {'dist': raw_f, 'time': current_time}
        elif (current_time - self.mem_f['time']) < self.memory_ttl: raw_f = self.mem_f['dist']

        self.front_dist, self.left_dist, self.right_dist = raw_f, raw_l, raw_r

        if self.state in ['NORMAL', 'SWERVE', 'PASSING']:
            if self.front_dist < self.critical_distance:
                self.get_logger().error("ZBYT BLISKO! Cofam...")
                self.state = 'REVERSE'
                self.state_start_time = time.time()
                return

        # --- NOWA LOGIKA Z ZAKAZEM WYPRZEDZANIA ---
        if self.state == 'NORMAL':
            if self.front_dist < self.trigger_distance:
                
                # Sprawdzamy czy po prawej/lewej nie ma linii ciągłej
                forbidden_left = (self.cam_left_px > self.solid_line_px)
                forbidden_right = (self.cam_right_px > self.solid_line_px)

                # Możemy wyprzedzać z danej strony tylko, jeśli jest miejsce z LiDARa ORAZ nie ma tam ciągłej
                can_go_left = (decide_l > 0.4) and not forbidden_left
                can_go_right = (decide_r > 0.4) and not forbidden_right

                if can_go_left and not can_go_right:
                    self.track_side = 'LEFT'  
                    self.swerve_sign = 1.0  
                    self.get_logger().warn("Wymijam z LEWEJ (Prawa zablokowana)")
                elif can_go_right and not can_go_left:
                    self.track_side = 'RIGHT' 
                    self.swerve_sign = -1.0 
                    self.get_logger().warn("Wymijam z PRAWEJ (Lewa zablokowana)")
                elif can_go_left and can_go_right:
                    # Obie strony wolne (linia przerywana z obu stron), wybieramy szerszą
                    if decide_l > decide_r:
                        self.track_side = 'LEFT'; self.swerve_sign = 1.0
                    else:
                        self.track_side = 'RIGHT'; self.swerve_sign = -1.0
                    self.get_logger().warn("Obie strony dozwolone, wybieram optymalną.")
                else:
                    self.get_logger().error("Brak możliwości wyminięcia! Hamowanie!")
                    # Obie zablokowane linią ciągłą lub ścianą. Wymusza zatrzymanie (lub cofanie).
                    self.state = 'REVERSE'
                    self.state_start_time = time.time()
                    return
                    
                self.state = 'SWERVE'
                self.state_start_time = time.time()
                
        elif self.state == 'SWERVE':
            if self.front_dist > 0.45:
                if self.track_side == 'RIGHT' and self.right_dist < 0.8: self.state = 'PASSING'
                elif self.track_side == 'LEFT' and self.left_dist < 0.8: self.state = 'PASSING'
                    
            if time.time() - self.state_start_time > 2.5:
                self.state = 'RETURN'
                self.state_start_time = time.time()
                
        elif self.state == 'PASSING':
            # Zmieniamy warunek powrotu:
            # 1. Musimy być wystarczająco daleko od przeszkody (boczny dystans > 0.9m)
            # 2. SEKCJA BEZPIECZEŃSTWA: Sprawdzamy czy w sektorach bocznych (6-8 godzina) jest czysto
            
            # Sektor 6-8 godzina to kąty od -2.5 do -1.5 radiana (dla prawej strony)
            # lub od 1.5 do 2.5 radiana (dla lewej strony)
            
            clear_behind = True
            if self.track_side == 'RIGHT':
                # Jeśli omijamy prawą stroną, sprawdzamy czy na prawej burcie (sektor tył-bok) jest czysto
                # Szukamy czy coś jest bliżej niż 0.4m w strefie tyłu
                if self.right_dist < 0.4: clear_behind = False
            else:
                # Jeśli omijamy lewą stroną, sprawdzamy lewą burtę
                if self.left_dist < 0.4: clear_behind = False

            # Warunek powrotu: minęliśmy przeszkodę bocznie ORAZ tył jest czysty
            is_passed = (self.right_dist > 0.9) if self.track_side == 'RIGHT' else (self.left_dist > 0.9)
            
            if is_passed and clear_behind:
                self.get_logger().info("Droga za przeszkodą czysta - wracam na tor.")
                self.state = 'RETURN'
                self.state_start_time = time.time()
                
        elif self.state == 'RETURN':
            # ZMIANA: Skrócono czas powrotu z 1.2s na 0.5s! (Zapobiega zjawisku stanięcia w poprzek bandy)
            if time.time() - self.state_start_time > 0.5: 
                self.state = 'NORMAL'
                
        elif self.state == 'REVERSE':
            if time.time() - self.state_start_time > 1.5: self.state = 'NORMAL'

    def publish_rviz_markers(self, obstacles):
        marker_array = MarkerArray()
        delete_marker = Marker(); delete_marker.action = Marker.DELETEALL
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
            msg.data = 999.0
            self.current_steer = 0.0 
            self.offset_pub.publish(msg)
            return

        if self.state == 'NORMAL':
            target_steer = self.camera_desired_offset 
        elif self.state == 'REVERSE':
            target_steer = 888.0 
        elif self.state == 'SWERVE':
            urgency = min(self.trigger_distance / max(self.front_dist, 0.05), 1.8) 
            target_steer = self.swerve_sign * self.base_swerve * urgency
        elif self.state == 'PASSING':
            if self.track_side == 'RIGHT': target_steer = max(-220.0, min(220.0, (self.right_dist - self.target_side_distance) * self.kp_wall))
            else: target_steer = max(-220.0, min(220.0, (self.target_side_distance - self.left_dist) * self.kp_wall))
        elif self.state == 'RETURN':
            # ZMIANA: Mniejsza siła kontry przy powrocie
            target_steer = -self.swerve_sign * (self.base_swerve * 0.4)

        # ZMIANA W PŁYNNOŚCI (Naprawa Wężykowania z punktu 1):
        if self.state == 'NORMAL':
            # Gdy steruje kamera, chcemy BARDZO SZYBKIEJ reakcji (0.7) - likwiduje opóźnienie/lag
            self.current_steer = self.current_steer + 0.7 * (target_steer - self.current_steer)
        elif self.state != 'REVERSE':
            # Gdy manewruje LiDAR, chcemy powolnych i płynnych ruchów niczym limuzyna (0.15)
            self.current_steer = self.current_steer + 0.15 * (target_steer - self.current_steer)
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