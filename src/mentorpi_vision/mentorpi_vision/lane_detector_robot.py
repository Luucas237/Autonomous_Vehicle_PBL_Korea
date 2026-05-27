#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from rclpy.qos import qos_profile_sensor_data  
import cv2 as cv
import numpy as np
import os

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_robot_node')
        self.bridge = CvBridge()
        
        self.frame_subscriber = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.listener_callback, qos_profile_sensor_data)
        self.color_subscriber = self.create_subscription(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', self.color_callback, 10)
        self.telemetry_publisher = self.create_publisher(Float32MultiArray, '/vision/lane_telemetry', 10)

        self.yellow_lower = np.array([15, 50, 180], dtype="uint8")
        self.yellow_upper = np.array([35, 255, 255], dtype="uint8")
        self.white_lower = np.array([55, 0, 210], dtype="uint8")
        self.white_upper = np.array([179, 255, 255], dtype="uint8")

        # ŚCIEŻKA DO WZORCA NA MALINIE
        self.template_path = '/home/ubuntu/ros2_ws/src/mentorpi_core/config/pedestrian.png'
        self.pedestrian_contour = self.load_template()

        self.get_logger().info('Vision Node: Aktywna struktura KLEPSYDRY (Road + Pedestrian ROI).')

    def load_template(self):
        if not os.path.exists(self.template_path):
            self.get_logger().error(f"Brak pliku wzorca pieszego: {self.template_path}")
            return None
            
        temp_img = cv.imread(self.template_path, cv.IMREAD_GRAYSCALE)
        _, thresh = cv.threshold(temp_img, 127, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv.contourArea) 
        return None

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            self.yellow_lower = np.array([data[0], data[1], data[2]], dtype="uint8")
            self.yellow_upper = np.array([data[3], data[4], data[5]], dtype="uint8")

    def find_lane_lines_at_y(self, lines, target_y):
        left_lane_x = None
        right_lane_x = None
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0: continue  
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    x_at_y = (target_y - intercept) / slope
                    
                    if slope < -0.1: 
                        if left_lane_x is None or x_at_y < left_lane_x:
                            left_lane_x = int(x_at_y)
                    elif slope > 0.1: 
                        if right_lane_x is None or x_at_y > right_lane_x:
                            right_lane_x = int(x_at_y)
        return left_lane_x, right_lane_x

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        h, w = frame.shape[:2] # 480x640

        # =======================================================
        # 1. STRUKTURA KLEPSYDRY (Zarządzanie Maskami)
        # =======================================================
        
        # A) Maska drogi (Dolny Trapez)
        road_mask = np.zeros((h, w), dtype=np.uint8)
        road_vertices = np.array([[
            (0, h),                  # Lewy dół
            (w, h),                  # Prawy dół
            (int(w * 0.8), 240),     # Prawy horyzont
            (int(w * 0.2), 240)      # Lewy horyzont
        ]], dtype=np.int32)
        cv.fillPoly(road_mask, road_vertices, 255)

        # B) Maska pieszego (Górny odwrócony Trapez - stykający się z drogą na Y=240)
        pedestrian_mask = np.zeros((h, w), dtype=np.uint8)
        ped_vertices = np.array([[
            (0, 0),                  # Lewa góra
            (w, 0),                  # Prawa góra
            (int(w * 0.8), 240),     # Prawy środek (styk z drogą)
            (int(w * 0.2), 240)      # Lewy środek (styk z drogą)
        ]], dtype=np.int32)
        cv.fillPoly(pedestrian_mask, ped_vertices, 255)

        # C) Maska dla linii STOP (Najniższy pasek)
        stop_mask = np.zeros((h, w), dtype=np.uint8)
        cv.rectangle(stop_mask, (0, 400), (w, h), 255, -1)


        # =======================================================
        # 2. DETEKCJA PIESZEGO (MOMENTY HU) W GÓRNYM TRAPEZIE
        # =======================================================
        pedestrian_detected = 0.0
        if self.pedestrian_contour is not None:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh_full = cv.threshold(gray_frame, 100, 255, cv.THRESH_BINARY_INV)
            
            # Wyszukiwanie ograniczone wyłącznie do górnej połowy klepsydry
            pedestrian_thresh = cv.bitwise_and(thresh_full, thresh_full, mask=pedestrian_mask)
            
            contours, _ = cv.findContours(pedestrian_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv.contourArea(cnt)
                if area > 400:
                    match_val = cv.matchShapes(self.pedestrian_contour, cnt, cv.CONTOURS_MATCH_I1, 0)
                    if match_val < 0.15:
                        pedestrian_detected = 1.0
                        break

        # =======================================================
        # 3. DETEKCJA LINII W DROGOWYM ROI (Logika Korei)
        # =======================================================
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        yellow_mask_full = cv.inRange(hsv_frame, self.yellow_lower, self.yellow_upper)
        road_yellow_mask = cv.bitwise_and(yellow_mask_full, yellow_mask_full, mask=road_mask)
        
        edges_lines = cv.Canny(cv.GaussianBlur(road_yellow_mask, (5, 5), 0), 50, 150)
        lines = cv.HoughLinesP(edges_lines, 1, np.pi/180, 50, maxLineGap=50)

        lx, rx = self.find_lane_lines_at_y(lines, target_y=320)
        lx = float(lx) if (lx is not None and lx > 0) else 0.0
        rx = float(rx) if (rx is not None and rx > 0) else 640.0

        # =======================================================
        # 4. DETEKCJA LINII STOP W DOLNYM ROI
        # =======================================================
        white_mask_full = cv.inRange(hsv_frame, self.white_lower, self.white_upper)
        road_white_mask = cv.bitwise_and(white_mask_full, white_mask_full, mask=stop_mask)
        
        edges_stop = cv.Canny(cv.GaussianBlur(road_white_mask, (5, 5), 0), 50, 150)
        lines_stop = cv.HoughLinesP(edges_stop, 1, np.pi/180, 50, minLineLength=80, maxLineGap=10)
        
        stop_detected = 0.0
        if lines_stop is not None:
            for line in lines_stop:
                y1, y2 = line[0][1], line[0][3]
                if abs(y1 - y2) < 10: 
                    stop_detected = 1.0
                    break

        # =======================================================
        # 5. WYSYŁKA
        # =======================================================
        tel_msg = Float32MultiArray()
        tel_msg.data = [lx, rx, stop_detected, pedestrian_detected] 
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