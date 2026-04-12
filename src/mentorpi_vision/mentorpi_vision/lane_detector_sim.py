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
        super().__init__('lane_detector_sim_node')
        self.bridge = CvBridge()
        
        self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        self.left_history = deque(maxlen=7)
        self.right_history = deque(maxlen=7)
        self.missing_left = 0
        self.missing_right = 0
        self.last_offset = 0.0

        # === ZMIANA: Temat z wirtualnej kamery z Gazebo ===
        self.frame_subscriber = self.create_subscription(
            Image, 
            '/camera/image_raw',  
            self.listener_callback,
            qos_profile_sensor_data     
        )

        self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)
        # Aby wrzucić porysowany obraz np. do Foxglove'a
        self.annotated_image_publisher = self.create_publisher(Image, '/camera/image_annotated', 10)

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info('Wizja (Symulacja) odpalona!')

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

        left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
        right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)

        output = frame.copy()
        output, status = self.draw_guideline(output, left_poly, right_poly)
        
        # Publikacja porysowanego obrazu do Foxglove!
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
            self.annotated_image_publisher.publish(annotated_msg)
        except Exception as e:
            pass

    def detect_white_lines(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180], dtype="uint8")
        upper_white = np.array([180, 30, 255], dtype="uint8")
        mask = cv.inRange(hsv, lower_white, upper_white)

        blur = cv.GaussianBlur(mask, (5, 5), 0)
        edges = cv.Canny(blur, 50, 150)

        height, width = edges.shape
        roi_mask = np.zeros_like(edges)
        
        top_width = int(width * 0.4)
        bottom_width = width
        top_y = int(height * 0.4)
        bottom_y = height

        vertices = np.array([[ 
            ((width - top_width) // 2, top_y),
            ((width + top_width) // 2, top_y),
            (bottom_width, bottom_y),
            (0, bottom_y)
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, vertices, 255)
        roi_edges = cv.bitwise_and(edges, roi_mask)

        lines = cv.HoughLinesP(roi_edges, 1, np.pi/180, 15, minLineLength=7, maxLineGap=3)

        left_lines = []
        right_lines = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1 + 0.0001)
                    if abs(slope) < 0.15: continue
                    if (x1 + x2) / 2 < width / 2: 
                        left_lines.append((x1, y1, x2, y2))
                    else: 
                        right_lines.append((x1, y1, x2, y2))

        return left_lines, right_lines

    def fit_and_filter(self, lines, history, missing_counter):
        if len(lines) == 0:
            missing_counter += 1
            if missing_counter > 5:
                history.clear()
            elif len(history) > 0:
                history.append(history[-1]) 
            return None, missing_counter

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        missing_counter = 0
        if len(np.unique(y_coords)) < 3:
            return None, missing_counter

        poly = np.polyfit(y_coords, x_coords, 2)
        poly[0] = np.clip(poly[0], -0.002, 0.002)

        history.append(poly)

        if len(history) == 7:
            smoothed_poly = np.zeros(3)
            for i in range(7):
                smoothed_poly += self.fir_weights[i] * history[i]
            return smoothed_poly, missing_counter
        else:
            return np.mean(history, axis=0), missing_counter

    def get_fitx(self, poly, ploty):
        if poly is None: return None
        return poly[0]*ploty**2 + poly[1]*ploty + poly[2]

    def draw_guideline(self, frame, left_poly, right_poly):
        height, width, _ = frame.shape
        ploty = np.linspace(int(height * 0.4), height, num=20)
        
        left_fitx = self.get_fitx(left_poly, ploty)
        right_fitx = self.get_fitx(right_poly, ploty)
        
        center_status = "BRAK"
        lines_detected = 0
        offset = self.last_offset

        if left_fitx is not None and right_fitx is not None:
            lines_detected = 2
            left_pts = np.int32(np.column_stack((left_fitx, ploty))).reshape((-1, 1, 2))
            right_pts = np.int32(np.column_stack((right_fitx, ploty))).reshape((-1, 1, 2))

            cv.polylines(frame, [left_pts], isClosed=False, color=(255, 0, 0), thickness=4)
            cv.polylines(frame, [right_pts], isClosed=False, color=(255, 0, 0), thickness=4)

            mid_fitx = (left_fitx + right_fitx) / 2
            mid_pts = np.int32(np.column_stack((mid_fitx, ploty))).reshape((-1, 1, 2))
            cv.polylines(frame, [mid_pts], isClosed=False, color=(0, 0, 255), thickness=3)

            lookahead_idx = int(len(ploty) * 0.4) 
            mid_x = mid_fitx[lookahead_idx]
            cv.drawMarker(frame, (int(mid_x), int(ploty[lookahead_idx])), (0, 255, 255), cv.MARKER_CROSS, 20, 2)

            offset = float(mid_x - (width / 2.0))
            self.last_offset = offset

            if abs(offset) < width * 0.05: center_status = "SRODEK"
            elif offset < 0: center_status = "LEWO"
            else: center_status = "PRAWO"
            
        elif left_fitx is not None:
            lines_detected = 1
            center_status = "TYLKO_LEWA"
        elif right_fitx is not None:
            lines_detected = 1
            center_status = "TYLKO_PRAWA"

        msg = Float32()
        msg.data = offset
        self.offset_value_publisher_.publish(msg)

        cv.putText(frame, f"FPS: {self.fps:.1f} | Kierunek: {center_status}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        log_msg = f"[FPS: {self.fps:5.1f}] Wykryto: {lines_detected} | Status: {center_status:12} | Offset: {offset:6.1f}"
        self.get_logger().info(log_msg)
        
        return frame, center_status

def main(args=None):
    rclpy.init(args=args)
    node = ProcessFrame()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()