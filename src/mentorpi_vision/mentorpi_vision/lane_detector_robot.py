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

        self.lower_color = np.array([20, 80, 80], dtype="uint8")
        self.upper_color = np.array([40, 255, 255], dtype="uint8")
        
        self.p_erode = 1
        self.p_dilate = 1
        self.p_canny_min = 30
        self.p_canny_max = 120
        self.p_hough_thr = 15
        
        # [PARAMETRY LINII PRZERYWANEJ]
        # p_hough_min: jak krótka może być pojedyncza przerywana kreska, by ją zauważyć
        # p_hough_max: jak duża wyrwa (brak farby) między kreskami pozwala na ich połączenie w jedną linię
        self.p_hough_min = 15
        self.p_hough_max = 150 
        
        self.current_curve_threshold = 0.0005 

        self.bumper_state = 'INACTIVE' 
        self.bumper_start_time = 0.0
        self.bumper_dir = 0.0          
        self.pixel_threshold = 1200 

        self.frame_subscriber = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.listener_callback, qos_profile_sensor_data)
        self.color_subscriber = self.create_subscription(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', self.color_callback, 10)
        self.algo_params_subscriber = self.create_subscription(Float32MultiArray, '/mentorpi/vision/algo_params', self.algo_params_callback, 10)
        
        self.telemetry_publisher = self.create_publisher(Float32MultiArray, '/vision/lane_telemetry', 10)

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info('Vision Node [SLAVE] z detekcją prostopadłą i dominacją prawej strony gotowy.')

    def algo_params_callback(self, msg):
        data = msg.data
        if len(data) == 8:
            self.p_erode = int(data[0])
            self.p_dilate = int(data[1])
            self.p_canny_min = int(data[2])
            self.p_canny_max = int(data[3])
            self.p_hough_thr = int(data[4])
            self.p_hough_min = int(data[5])
            self.p_hough_max = int(data[6])
            self.current_curve_threshold = float(data[7])

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            self.lower_color = np.array([data[0], data[1], data[2]], dtype="uint8")
            self.upper_color = np.array([data[3], data[4], data[5]], dtype="uint8")

    def listener_callback(self, msg):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time + 0.0001)
        self.last_time = current_time
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.perform_detection(frame)
        except Exception:
            pass

    def perform_detection(self, frame):
        left_lines, right_lines, roi_mask, perp_warn = self.detect_lines_core(frame)
        left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
        right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)
        self.calculate_and_log(frame.shape, left_poly, right_poly, roi_mask, perp_warn)

    def detect_lines_core(self, frame):
        import math # Upewnij się, że masz import math na górze pliku
        height, width = frame.shape[:2]
        crop_y = int(height * 0.55) 
        cropped_frame = frame[crop_y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]

        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_color, self.upper_color)

        roi_mask = np.zeros_like(mask)
        x_top_left = int(crop_w * 0.3)   
        x_top_right = int(crop_w * 0.7)  
        side_height = 135
        
        vertices = np.array([[ 
            (0, crop_h), (crop_w, crop_h), (crop_w, crop_h - side_height),        
            (x_top_right, 0), (x_top_left, 0), (0, crop_h - side_height)              
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, vertices, 255)
        roi = cv.bitwise_and(mask, roi_mask)

        roi_blurred = cv.GaussianBlur(roi, (7, 7), 0)
        kernel = np.ones((5, 5), np.uint8)
        roi_clean = cv.erode(roi_blurred, kernel, iterations=self.p_erode)
        roi_clean = cv.dilate(roi_clean, kernel, iterations=self.p_dilate)

        edges = cv.Canny(roi_clean, self.p_canny_min, self.p_canny_max)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, self.p_hough_thr, minLineLength=self.p_hough_min, maxLineGap=self.p_hough_max)

        left_lines, right_lines = [], []
        perpendicular_warning = False

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    real_y1, real_y2 = y1 + crop_y, y2 + crop_y
                    slope = (real_y2 - real_y1) / (x2 - x1 + 0.0001)
                    
                    # 1. TWARDY WARUNEK MATEMATYCZNY (Eliminacja Spamu)
                    # Obliczamy długość linii w pikselach
                    line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if abs(slope) < 0.20: 
                        # Musi być dłuższa niż 140px i być na samym dole ekranu (ostatnie 60px)
                        if line_length > 140 and (real_y1 > height - 60 or real_y2 > height - 60):
                            perpendicular_warning = True
                        continue 

                    if slope < 0: 
                        left_lines.append((x1, real_y1, x2, real_y2))
                    else: 
                        right_lines.append((x1, real_y1, x2, real_y2))

        return left_lines, right_lines, roi_clean, perpendicular_warning

    def fit_and_filter(self, lines, history, missing_counter):
        if len(lines) == 0:
            missing_counter += 1
            if missing_counter > 15: history.clear()
            elif len(history) > 0: history.append(history[-1]) 
            
            if len(history) > 0: return history[-1], missing_counter
            else: return None, missing_counter

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        missing_counter = 0
        if len(np.unique(y_coords)) < 2: return None, missing_counter

        poly1 = np.polyfit(y_coords, x_coords, 1)
        poly = np.array([0.0, poly1[0], poly1[1]]) 

        if len(history) == 0:
            history.append(poly)
        else:
            smoothed_poly = 0.8 * poly + 0.2 * history[-1]
            history.append(smoothed_poly)
            
        return history[-1], missing_counter

    def calculate_and_log(self, frame_shape, left_poly, right_poly, roi_mask, perp_warn):
        current_time = time.time()
        height, width, _ = frame_shape
        target_offset = self.last_offset
        center_x = width / 2.0
        lookahead_y = int(height * 0.78) 
        
        LANE_SHIFT_PX = 190.0 

        if left_poly is not None and right_poly is not None:
            left_x_look = left_poly[0]*(lookahead_y**2) + left_poly[1]*lookahead_y + left_poly[2]
            right_x_look = right_poly[0]*(lookahead_y**2) + right_poly[1]*lookahead_y + right_poly[2]
            if abs(right_x_look - left_x_look) < 120:
                left_poly = None

        if perp_warn:
            # Wysyłamy sygnał 888.0 do Avoidera
            target_offset = 888.0
        elif right_poly is not None:
            right_x_look = right_poly[0]*(lookahead_y**2) + right_poly[1]*lookahead_y + right_poly[2]
            estimated_center_x = right_x_look - LANE_SHIFT_PX
            target_offset = float(estimated_center_x - center_x)
        elif left_poly is not None:
            left_x_look = left_poly[0]*(lookahead_y**2) + left_poly[1]*lookahead_y + left_poly[2]
            estimated_center_x = left_x_look + LANE_SHIFT_PX
            target_offset = float(estimated_center_x - center_x)
        else:
            target_offset = self.last_offset

        # 2. NAPRAWA ZDERZAKÓW (Zwężone i mniej czułe)
        crop_h, crop_w = roi_mask.shape
        bumper_h, bumper_w = 60, 90   # Było 70x180 - teraz to wąskie prostokąty brzegowe
        self.pixel_threshold = 1600   # Zwiększona tolerancja

        left_zone = roi_mask[crop_h - bumper_h : crop_h, 5 : bumper_w]
        right_zone = roi_mask[crop_h - bumper_h : crop_h, crop_w - bumper_w - 5 : crop_w]
        
        left_pixels = cv.countNonZero(left_zone)
        right_pixels = cv.countNonZero(right_zone)
        bumper_active_flag = False

        def calculate_dynamic_force(pixels):
            return 100.0 + (150.0) * min(1.0, max(0.0, (pixels - self.pixel_threshold) / 2000.0)) 

        if self.bumper_state == 'INACTIVE' and not perp_warn:
            if left_pixels > self.pixel_threshold:
                self.bumper_state = 'ESCAPING'
                self.bumper_start_time = current_time
                self.bumper_dir = calculate_dynamic_force(left_pixels)
            elif right_pixels > self.pixel_threshold:
                self.bumper_state = 'ESCAPING'
                self.bumper_start_time = current_time
                self.bumper_dir = -calculate_dynamic_force(right_pixels)

        if self.bumper_state == 'ESCAPING':
            bumper_active_flag = True
            if current_time - self.bumper_start_time < 0.6: target_offset = self.bumper_dir
            else:
                self.bumper_state = 'ALIGNING'
                self.bumper_start_time = current_time
        elif self.bumper_state == 'ALIGNING':
            bumper_active_flag = True
            if current_time - self.bumper_start_time < 0.4: target_offset = -self.bumper_dir * 0.30 
            else: self.bumper_state = 'INACTIVE'

        if not perp_warn:
            self.last_offset = target_offset

        telemetry_array = [0.0] * 12
        if left_poly is not None:
            telemetry_array[0] = 1.0
            telemetry_array[1:4] = left_poly
        if right_poly is not None:
            telemetry_array[4] = 1.0
            telemetry_array[5:8] = right_poly
            
        telemetry_array[8] = float(target_offset) 
        telemetry_array[9] = 1.0 if bumper_active_flag else 0.0
        telemetry_array[10] = float(left_pixels)
        telemetry_array[11] = float(right_pixels)
        
        tel_msg = Float32MultiArray()
        tel_msg.data = telemetry_array
        self.telemetry_publisher.publish(tel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ProcessFrame()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()