#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data  

import cv2 as cv
import numpy as np
from collections import deque
import time

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_robot_node')
        self.bridge = CvBridge()
        self.left_history = deque(maxlen=5)
        self.right_history = deque(maxlen=5)

        self.frame_subscriber = self.create_subscription(
            Image, 
            '/ascamera/camera_publisher/rgb0/image',  
            self.listener_callback,
            qos_profile_sensor_data     
        )

        self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info('Wizja odpalona (wersja bez GUI)! Czekam na strumień wideo...')

    def listener_callback(self, msg):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time + 0.0001)
        self.last_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.perform_detection(frame)
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu: {e}")

    def perform_detection(self, frame):
        left_lines, right_lines = self.detect_white_lines(frame)
        
        left_poly_raw = self.average_line(left_lines) if len(left_lines) > 0 else None
        right_poly_raw = self.average_line(right_lines) if len(right_lines) > 0 else None

        left_poly = self.smooth_poly(self.left_history, left_poly_raw)
        right_poly = self.smooth_poly(self.right_history, right_poly_raw)

        # Zamiast draw_guideline, wywołujemy funkcję liczącą offset i logującą do konsoli
        self.calculate_and_log(frame.shape, left_poly, right_poly)

    def detect_white_lines(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype="uint8")
        upper_white = np.array([180, 30, 255], dtype="uint8")
        mask = cv.inRange(hsv, lower_white, upper_white)

        height, width = mask.shape
        roi_mask = np.zeros_like(mask)

        top_width = int(width * 0.3)
        bottom_width = width
        top_y = int(height * 0.5)
        bottom_y = height

        vertices = np.array([[ 
            ((width - top_width) // 2, top_y),
            ((width + top_width) // 2, top_y),
            (bottom_width, bottom_y),
            (0, bottom_y)
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, vertices, 255)
        roi = cv.bitwise_and(mask, roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        roi = cv.erode(roi, kernel, iterations=1)
        roi = cv.dilate(roi, kernel, iterations=2)

        edges = cv.Canny(roi, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1 + 0.0001)
                    if abs(slope) < 0.5: continue
                    if (x1 + x2) / 2 < width / 2: left_lines.append((x1, y1, x2, y2))
                    else: right_lines.append((x1, y1, x2, y2))

        # Zwracamy tylko wykryte linie, ignorujemy roi_debug (bo nie rysujemy)
        return left_lines, right_lines

    def smooth_poly(self, history, new_poly):
        if new_poly is not None:
            history.append(new_poly)
            return np.mean(history, axis=0)
        return None

    def average_line(self, lines):
        if len(lines) == 0: return None
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]
        return np.polyfit(y_coords, x_coords, deg=1)

    def calculate_and_log(self, frame_shape, left_poly, right_poly):
        height, width, _ = frame_shape
        y1 = int(height * 0.5)
        y2 = height
        
        center_status = "BRAK"
        lines_detected = 0
        offset = 999.0

        if left_poly is not None and right_poly is not None:
            lines_detected = 2
            left_x2 = int(left_poly[0] * y2 + left_poly[1])
            right_x2 = int(right_poly[0] * y2 + right_poly[1])
            mid_x2 = (left_x2 + right_x2) // 2
            offset = float(mid_x2 - (width // 2))

            if abs(offset) < width * 0.05: center_status = "SRODEK"
            elif offset < 0: center_status = "LEWO"
            else: center_status = "PRAWO"

        elif left_poly is not None:
            lines_detected = 1
            center_status = "TYLKO_LEWA"
        elif right_poly is not None:
            lines_detected = 1
            center_status = "TYLKO_PRAWA"

        # Publikacja wartości offsetu dla innych węzłów (np. PC lub sterownika silników)
        msg = Float32()
        msg.data = offset
        self.offset_value_publisher_.publish(msg)

        # Logowanie w jednej linijce do terminala (np: [FPS: 28.5] Wykryto: 2 | Status: SRODEK | Offset: 12.0)
        log_msg = f"[FPS: {self.fps:5.1f}] Wykryto: {lines_detected} | Status: {center_status:12} | Offset: {offset:6.1f}"
        self.get_logger().info(log_msg)

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