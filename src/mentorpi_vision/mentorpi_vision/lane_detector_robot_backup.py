#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32MultiArray, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data  

import cv2 as cv
import numpy as np
from collections import deque
import time

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_robot_node')
        self.bridge = CvBridge()
        
        self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        
        self.left_history = deque(maxlen=7)
        self.right_history = deque(maxlen=7)
        
        self.missing_left = 0
        self.missing_right = 0
        self.last_offset = 0.0

        # --- FILTR DOLNOPRZEPUSOTWY (PŁYNNOŚĆ KIEROWNICY) ---
        self.current_steer = 0.0  # Faktyczna, aktualna wartość skrętu
        self.steering_alpha = 0.25 # Zwinność (1.0 = brak płynności, 0.1 = bardzo leniwy)

        self.lower_color = np.array([0, 0, 0], dtype="uint8")
        self.upper_color = np.array([180, 255, 80], dtype="uint8")
        self.current_curve_threshold = 0.0005 

        self.engines_on = False
        self.last_engine_state = False

        # --- MASZYNA STANÓW BUMPERA ---
        self.bumper_state = 'INACTIVE' 
        self.bumper_start_time = 0.0
        self.bumper_dir = 0.0          
        self.pixel_threshold = 1200 # Próg wykrycia koca/linii

        self.frame_subscriber = self.create_subscription(
            Image, 
            '/ascamera/camera_publisher/rgb0/image',  
            self.listener_callback,
            qos_profile_sensor_data     
        )

        self.color_subscriber = self.create_subscription(
            Int32MultiArray,
            '/mentorpi/vision/hsv_thresholds',
            self.color_callback,
            10
        )

        self.curve_subscriber = self.create_subscription(
            Float32,
            '/mentorpi/vision/curve_threshold',
            self.curve_callback,
            10
        )

        self.gui_cmd_subscriber = self.create_subscription(
            Float32,
            '/vision/offset_raw',
            self.gui_cmd_callback,
            qos_profile_sensor_data
        )

        self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)
        self.telemetry_publisher = self.create_publisher(Float32MultiArray, '/vision/lane_telemetry', 10)
        self.mask_publisher = self.create_publisher(Image, '/vision/robot_mask', qos_profile_sensor_data)

        self.last_time = time.time()
        self.fps = 0.0
        
        self.get_logger().info('Vision Robot Node: DYNAMIC BUMPER + LOW-PASS FILTER READY!')

    def gui_cmd_callback(self, msg):
        if msg.data == 999.0:
            self.engines_on = False
            # Wymuszenie wyśrodkowania wirtualnej kierownicy podczas postoju
            self.current_steer = 0.0 
        else:
            self.engines_on = True

        if self.engines_on != self.last_engine_state:
            if self.engines_on:
                self.get_logger().info(">>> ENGINES ON! <<<")
            else:
                self.get_logger().warn(">>> ENGINES OFF / CENTER (STOP) <<<")
            self.last_engine_state = self.engines_on

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            new_lower = np.array([data[0], data[1], data[2]], dtype="uint8")
            new_upper = np.array([data[3], data[4], data[5]], dtype="uint8")
            
            if not np.array_equal(self.lower_color, new_lower) or not np.array_equal(self.upper_color, new_upper):
                self.lower_color = new_lower
                self.upper_color = new_upper

    def curve_callback(self, msg):
        new_threshold = msg.data
        if abs(self.current_curve_threshold - new_threshold) > 0.00001:
            self.current_curve_threshold = new_threshold

    def listener_callback(self, msg):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time + 0.0001)
        self.last_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.perform_detection(frame)
        except Exception as e:
            pass

    def perform_detection(self, frame):
        if not self.engines_on:
            msg = Float32()
            msg.data = 999.0
            self.offset_value_publisher_.publish(msg)
            self.current_steer = 0.0

        left_lines, right_lines, roi_mask = self.detect_lines_core(frame)

        left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
        right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)

        self.calculate_and_log(frame.shape, left_poly, right_poly, roi_mask)

    def detect_lines_core(self, frame):
        height, width = frame.shape[:2]
        
        crop_y = int(height * 0.55) 
        cropped_frame = frame[crop_y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]

        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_color, self.upper_color)

        roi_mask = np.zeros_like(mask)
        x_top_left = int(crop_w * 0.2)   
        x_top_right = int(crop_w * 0.8)  
        side_height = 135
        
        vertices = np.array([[ 
            (0, crop_h),                           
            (crop_w, crop_h),                      
            (crop_w, crop_h - side_height),        
            (x_top_right, 0),                      
            (x_top_left, 0),                       
            (0, crop_h - side_height)              
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, vertices, 255)
        roi = cv.bitwise_and(mask, roi_mask)

        try:
            mask_msg = self.bridge.cv2_to_imgmsg(roi, encoding="mono8")
            self.mask_publisher.publish(mask_msg)
        except Exception:
            pass

        kernel = np.ones((5, 5), np.uint8)
        roi_clean = cv.erode(roi, kernel, iterations=1)
        roi_clean = cv.dilate(roi_clean, kernel, iterations=2)

        edges = cv.Canny(roi_clean, 30, 120)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=60)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    real_y1 = y1 + crop_y
                    real_y2 = y2 + crop_y
                    slope = (real_y2 - real_y1) / (x2 - x1 + 0.0001)
                    if abs(slope) < 0.15: continue
                    if (x1 + x2) / 2 < width / 2: 
                        left_lines.append((x1, real_y1, x2, real_y2))
                    else: 
                        right_lines.append((x1, real_y1, x2, real_y2))

        return left_lines, right_lines, roi_clean

    def fit_and_filter(self, lines, history, missing_counter):
        if len(lines) == 0:
            missing_counter += 1
            if missing_counter > 45:
                history.clear()
            elif len(history) > 0:
                history.append(history[-1]) 
            
            if len(history) > 0:
                return np.mean(history, axis=0), missing_counter
            else:
                return None, missing_counter

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        missing_counter = 0

        if len(np.unique(y_coords)) < 3:
            return None, missing_counter

        poly2 = np.polyfit(y_coords, x_coords, 2)
        curvature = poly2[0]

        if abs(curvature) < self.current_curve_threshold:
            poly1 = np.polyfit(y_coords, x_coords, 1)
            poly = np.array([0.0, poly1[0], poly1[1]])
        else:
            poly = poly2
            poly[0] = np.clip(poly[0], -0.003, 0.003)

        history.append(poly)

        if len(history) == 7:
            smoothed_poly = np.zeros(3)
            for i in range(7):
                smoothed_poly += self.fir_weights[i] * history[i]
            return smoothed_poly, missing_counter
        else:
            return np.mean(history, axis=0), missing_counter

    def calculate_and_log(self, frame_shape, left_poly, right_poly, roi_mask):
        current_time = time.time()
        height, width, _ = frame_shape
        target_offset = self.last_offset
        lines_detected = 0
        
        center_x = width / 2.0
        lookahead_y = int(height * 0.70) 

        LANE_WIDTH_PX = 380.0 

        # ==========================================
        # 1. WARSTWA NAWIGACJI (MATEMATYKA)
        # ==========================================
        left_x_look, right_x_look = None, None

        if left_poly is not None:
            left_x_look = left_poly[0]*(lookahead_y**2) + left_poly[1]*lookahead_y + left_poly[2]
            lines_detected += 1
            
        if right_poly is not None:
            right_x_look = right_poly[0]*(lookahead_y**2) + right_poly[1]*lookahead_y + right_poly[2]
            lines_detected += 1

        if lines_detected == 2:
            mid_x = (left_x_look + right_x_look) / 2.0
            target_offset = float(mid_x - center_x)
            
        elif lines_detected == 1:
            if left_poly is not None:
                mid_x = left_x_look + (LANE_WIDTH_PX / 2.0)
                target_offset = float(mid_x - center_x)
            elif right_poly is not None:
                mid_x = right_x_look - (LANE_WIDTH_PX / 2.0)
                target_offset = float(mid_x - center_x)

        # ==========================================
        # 2. WARSTWA PRZETRWANIA (DYNAMICZNY ZDERZAK)
        # ==========================================
        crop_h, crop_w = roi_mask.shape
        bumper_h = 70    
        bumper_w = 180   

        left_zone = roi_mask[crop_h - bumper_h : crop_h, 0 : bumper_w]
        right_zone = roi_mask[crop_h - bumper_h : crop_h, crop_w - bumper_w : crop_w]
        
        left_pixels = cv.countNonZero(left_zone)
        right_pixels = cv.countNonZero(right_zone)
        
        bumper_active_flag = False

        # --- Funkcja wyliczająca siłę od 100 do 300 ---
        def calculate_dynamic_force(pixels):
            base_force = 100.0
            max_force = 300.0
            excess = pixels - self.pixel_threshold
            # Zderzak osiąga pełną moc 300 przy 4200 pikselach, a zaczyna od 100.
            scale = min(1.0, max(0.0, excess / 3000.0)) 
            return base_force + (max_force - base_force) * scale

        if self.bumper_state == 'INACTIVE':
            if left_pixels > self.pixel_threshold:
                self.bumper_state = 'ESCAPING'
                self.bumper_start_time = current_time
                self.bumper_dir = calculate_dynamic_force(left_pixels)
                self.get_logger().warn(f"BUMPER LEWY! Moc: {self.bumper_dir:.1f} (Piksele: {left_pixels})")
                
            elif right_pixels > self.pixel_threshold:
                self.bumper_state = 'ESCAPING'
                self.bumper_start_time = current_time
                self.bumper_dir = -calculate_dynamic_force(right_pixels)
                self.get_logger().warn(f"BUMPER PRAWY! Moc: {self.bumper_dir:.1f} (Piksele: {right_pixels})")

        if self.bumper_state == 'ESCAPING':
            bumper_active_flag = True
            escape_time = current_time - self.bumper_start_time
            if escape_time < 0.6: 
                target_offset = self.bumper_dir
            else:
                self.bumper_state = 'ALIGNING'
                self.bumper_start_time = current_time

        elif self.bumper_state == 'ALIGNING':
            bumper_active_flag = True
            align_time = current_time - self.bumper_start_time
            if align_time < 0.4: 
                # Bardzo delikatna kontra na wyprostowanie (tylko 30% siły ucieczki)
                target_offset = -self.bumper_dir * 0.30 
            else:
                self.bumper_state = 'INACTIVE'

        self.last_offset = target_offset

        # ==========================================
        # 3. FILTR DOLNOPRZEPUSTOWY (PŁYNNOŚĆ)
        # ==========================================
        # Stopniowe dążenie aktualnego skrętu do wyliczonego celu
        self.current_steer = self.current_steer + self.steering_alpha * (target_offset - self.current_steer)

        if self.engines_on:
            msg = Float32()
            msg.data = float(self.current_steer)
            self.offset_value_publisher_.publish(msg)

        # --- TELEMETRIA DO GUI ---
        telemetry_array = [0.0] * 10
        if left_poly is not None:
            telemetry_array[0] = 1.0
            telemetry_array[1], telemetry_array[2], telemetry_array[3] = left_poly[0], left_poly[1], left_poly[2]
        if right_poly is not None:
            telemetry_array[4] = 1.0
            telemetry_array[5], telemetry_array[6], telemetry_array[7] = right_poly[0], right_poly[1], right_poly[2]
            
        # Wysyłamy do GUI nałożony filtr, aby zielona kropka poruszała się równie gładko co koła robota
        telemetry_array[8] = float(self.current_steer)
        telemetry_array[9] = 1.0 if bumper_active_flag else 0.0
        
        tel_msg = Float32MultiArray()
        tel_msg.data = telemetry_array
        self.telemetry_publisher.publish(tel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ProcessFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()