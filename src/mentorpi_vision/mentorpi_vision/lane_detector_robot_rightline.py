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
        self.filtered_offset = 0.0
        self.max_offset_step = 22.0   # maksymalna zmiana offsetu na jedną klatkę [px]
        self.offset_alpha = 0.22      # filtr dolnoprzepustowy offsetu; mniejsze = płynniej
        self.lane_width_px = 400.0    # realny rozstaw linii pasa [px]

        self.lower_color = np.array([20, 80, 80], dtype="uint8")
        self.upper_color = np.array([40, 255, 255], dtype="uint8")
        
        self.p_erode = 1
        self.p_dilate = 1
        self.p_canny_min = 30
        self.p_canny_max = 120
        self.p_hough_thr = 15
        self.p_hough_min = 15
        self.p_hough_max = 50
        self.current_curve_threshold = 0.0005 

        self.bumper_state = 'INACTIVE' 
        self.bumper_start_time = 0.0
        self.bumper_dir = 0.0          
        self.pixel_threshold = 1200 

        self.frame_subscriber = self.create_subscription(
            Image, '/ascamera/camera_publisher/rgb0/image', self.listener_callback, qos_profile_sensor_data)
        self.color_subscriber = self.create_subscription(
            Int32MultiArray, '/mentorpi/vision/hsv_thresholds', self.color_callback, 10)
        self.algo_params_subscriber = self.create_subscription(
            Float32MultiArray, '/mentorpi/vision/algo_params', self.algo_params_callback, 10)
        
        self.telemetry_publisher = self.create_publisher(Float32MultiArray, '/vision/lane_telemetry', 10)

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info('Vision Node [SLAVE] - jazda po prawej linii, pas 400 px, wygładzanie offsetu gotowe.')

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
            self.get_logger().info(f"==> Change: Erode={self.p_erode}, Dilate={self.p_dilate}, Canny={self.p_canny_min}-{self.p_canny_max}, HoughMin={self.p_hough_min}")

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            self.lower_color = np.array([data[0], data[1], data[2]], dtype="uint8")
            self.upper_color = np.array([data[3], data[4], data[5]], dtype="uint8")
            # PRZYWRÓCONY LOG
            self.get_logger().info(f"==> Change color: Min={self.lower_color}, Max={self.upper_color}")

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
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    real_y1, real_y2 = y1 + crop_y, y2 + crop_y
                    slope = (real_y2 - real_y1) / (x2 - x1 + 0.0001)
                    
                    if abs(slope) < 0.1: continue # Odrzucamy poziome śmieci

                    if slope < 0: 
                        left_lines.append((x1, real_y1, x2, real_y2))
                    else: 
                        right_lines.append((x1, real_y1, x2, real_y2))

        return left_lines, right_lines, roi_clean

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
        poly = np.array([0.0, poly1[0], poly1[1]]) # A=0 (brak łuku), B=kąt, C=pozycja

        if len(history) == 0:
            history.append(poly)
        else:
            smoothed_poly = 0.8 * poly + 0.2 * history[-1]
            history.append(smoothed_poly)
            
        return history[-1], missing_counter

    def calculate_and_log(self, frame_shape, left_poly, right_poly, roi_mask):
        current_time = time.time()
        height, width, _ = frame_shape
        target_offset = self.last_offset
        center_x = width / 2.0
        lookahead_y = int(height * 0.70) 
        
        # ============================================================
        # GŁÓWNA ZMIANA: jazda w oparciu o PRAWĄ linię pasa.
        # Zakładamy, że odległość między dwiema liniami pasa to 400 px,
        # czyli środek pasa jest 200 px w lewo od prawej linii.
        #
        # offset > 0  -> środek pasa jest po prawej od kamery
        # offset < 0  -> środek pasa jest po lewej od kamery
        # ============================================================
        raw_target_offset = None
        half_lane = self.lane_width_px / 2.0

        if right_poly is not None:
            # Priorytet: prawa linia. To stabilizuje jazdę, gdy lewa jest przerywana
            # albo chwilowo znika podczas wymijania/przejazdu między pasami.
            right_x_look = right_poly[0]*(lookahead_y**2) + right_poly[1]*lookahead_y + right_poly[2]
            desired_center_x = right_x_look - half_lane
            raw_target_offset = float(desired_center_x - center_x)

        elif left_poly is not None:
            # Awaryjnie, gdy prawej nie widać, rekonstruujemy środek z lewej linii.
            left_x_look = left_poly[0]*(lookahead_y**2) + left_poly[1]*lookahead_y + left_poly[2]
            desired_center_x = left_x_look + half_lane
            raw_target_offset = float(desired_center_x - center_x)

        if raw_target_offset is not None:
            # 1) ograniczenie skoku offsetu między klatkami, żeby nie było przeskoku
            #    z maksymalnego skrętu w prawo na maksymalny w lewo.
            delta = raw_target_offset - self.filtered_offset
            delta = float(np.clip(delta, -self.max_offset_step, self.max_offset_step))
            stepped_offset = self.filtered_offset + delta

            # 2) filtr dolnoprzepustowy, czyli miękkie dochodzenie do celu.
            self.filtered_offset = (1.0 - self.offset_alpha) * self.filtered_offset + self.offset_alpha * stepped_offset
            target_offset = self.filtered_offset
        else:
            # Gdy nie widzimy żadnej linii, nie wymyślamy nowego skrętu, tylko
            # spokojnie trzymamy poprzedni kierunek.
            target_offset = self.last_offset

        crop_h, crop_w = roi_mask.shape
        bumper_h, bumper_w = 70, 180   

        left_zone = roi_mask[crop_h - bumper_h : crop_h, 0 : bumper_w]
        right_zone = roi_mask[crop_h - bumper_h : crop_h, crop_w - bumper_w : crop_w]
        
        left_pixels = cv.countNonZero(left_zone)
        right_pixels = cv.countNonZero(right_zone)
        bumper_active_flag = False

        def calculate_dynamic_force(pixels):
            # Zderzak zostaje, ale bez agresywnego szarpnięcia kierownicą.
            return 60.0 + (90.0) * min(1.0, max(0.0, (pixels - self.pixel_threshold) / 3000.0)) 

        if self.bumper_state == 'INACTIVE':
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
            if current_time - self.bumper_start_time < 0.35: target_offset = self.bumper_dir
            else:
                self.bumper_state = 'ALIGNING'
                self.bumper_start_time = current_time
        elif self.bumper_state == 'ALIGNING':
            bumper_active_flag = True
            if current_time - self.bumper_start_time < 0.25: target_offset = -self.bumper_dir * 0.20 
            else: self.bumper_state = 'INACTIVE'

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