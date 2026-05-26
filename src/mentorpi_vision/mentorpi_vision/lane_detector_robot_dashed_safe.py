#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data  

import cv2 as cv
import numpy as np
import time

class VisionSlave(Node):
    def __init__(self):
        super().__init__('lane_detector_robot_node')
        self.bridge = CvBridge()
        
        # --- ZAKRESY KOLORÓW (Koreańskie domyślne) ---
        self.yellow_lower = np.array([15, 50, 180], dtype="uint8")
        self.yellow_upper = np.array([35, 255, 255], dtype="uint8")
        
        self.white_lower = np.array([55, 0, 210], dtype="uint8")
        self.white_upper = np.array([179, 255, 255], dtype="uint8")

        # --- PIESZY (Momenty Hu) ---
        self.SCIEZKA_WZORZEC = r'/home/ubuntu/shared/wzorzec.jpeg' # PODMIEŃ NA MALINIE!
        self.pedestrian_prog_czern = 70
        self.pedestrian_prog_dopasowania = 0.2
        self.pedestrian_enabled = False
        self.glowny_kontur_wzorca = None
        
        self.get_logger().info('Wczytywanie wzorca pieszego...')
        self.load_pedestrian_template()

        # --- SUBSKRYPCJE I PUBLIKACJE ---
        self.frame_subscriber = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.listener_callback, qos_profile_sensor_data)
        self.color_subscriber = self.create_subscription(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', self.color_callback, 10)
        
        # NOWY FORMAT TELEMETRII: [lx, rx, stop_line_flag, pedestrian_flag]
        self.telemetry_publisher = self.create_publisher(Float32MultiArray, '/vision/lane_telemetry', 10)

    def load_pedestrian_template(self):
        img_color = cv.imread(self.SCIEZKA_WZORZEC)
        if img_color is not None:
            img_color = cv.resize(img_color, (0, 0), fx=0.1, fy=0.1)
            img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(img_gray, self.pedestrian_prog_czern, 255, cv.THRESH_BINARY_INV)
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                self.glowny_kontur_wzorca = max(contours, key=cv.contourArea)
                self.pedestrian_enabled = True
                self.get_logger().info("Wzorzec pieszego gotowy.")
        else:
            self.get_logger().warn("Brak zdjęcia wzorca. Detekcja pieszego wyłączona.")

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            self.yellow_lower = np.array([data[0], data[1], data[2]], dtype="uint8")
            self.yellow_upper = np.array([data[3], data[4], data[5]], dtype="uint8")

    def detect_pedestrian(self, frame):
        if not self.pedestrian_enabled: return 0.0
        height, width = frame.shape[:2]
        crop_y = int(height * 0.55)
        mask = np.zeros((height, width), dtype=np.uint8)
        trap_bottom_width = int(width * 0.4)
        offset_x = int((width - trap_bottom_width) / 2)
        pts = np.array([[[0, 0], [width, 0], [offset_x + trap_bottom_width, crop_y], [offset_x, crop_y]]], dtype=np.int32)
        cv.fillPoly(mask, pts, 255)
        
        roi_top = cv.bitwise_and(frame, frame, mask=mask)[0:crop_y, 0:width]
        gray = cv.cvtColor(roi_top, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, self.pedestrian_prog_czern, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for kontur in contours:
            if cv.contourArea(kontur) < 500: continue
            wynik = cv.matchShapes(self.glowny_kontur_wzorca, kontur, cv.CONTOURS_MATCH_I1, 0.0)
            if wynik < self.pedestrian_prog_dopasowania: return 1.0
        return 0.0

    def find_lane_lines_at_y(self, lines, target_y):
        left_lane_x, right_lane_x = None, None
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0: continue  
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    x_at_y = (target_y - intercept) / slope
                    if slope < 0:
                        if left_lane_x is None or x_at_y < left_lane_x: left_lane_x = int(x_at_y)
                    elif slope > 0:
                        if right_lane_x is None or x_at_y > right_lane_x: right_lane_x = int(x_at_y)
        return left_lane_x, right_lane_x

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 1. PIESZY (Górny ROI)
            pedestrian_flag = self.detect_pedestrian(cv_image)

            # 2. LINIE JAZDY (Żółte - Dolny ROI)
            crop_img = cv_image[300:, :]
            hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
            yellow_mask = cv.inRange(hsv, self.yellow_lower, self.yellow_upper)

            # ---> DODAJ TE DWIE LINIJKI (Liczenie białych pikseli na masce):
            left_px = np.sum(yellow_mask[:, :320] == 255)
            right_px = np.sum(yellow_mask[:, 320:] == 255)
            # <---

            yellow_img = cv.bitwise_and(crop_img, crop_img, mask=yellow_mask)
            
            gray_y = cv.cvtColor(yellow_img, cv.COLOR_BGR2GRAY)
            blur_y = cv.GaussianBlur(gray_y, (5, 5), 0)
            edges_y = cv.Canny(blur_y, 50, 150)
            lines_y = cv.HoughLinesP(edges_y, 1, np.pi/180, 50, maxLineGap=50)
            
            lx, rx = self.find_lane_lines_at_y(lines_y, target_y=20)
            if lx is None or lx < 0: lx = 0
            if rx is None or rx < 0: rx = 640

            # 3. LINIA ZATRZYMANIA (Biała - Sam dół)
            stop_img = cv_image[400:, :]
            hsv_stop = cv.cvtColor(stop_img, cv.COLOR_BGR2HSV)
            white_mask = cv.inRange(hsv_stop, self.white_lower, self.white_upper)
            stop_color = cv.bitwise_and(stop_img, stop_img, mask=white_mask)
            
            gray_s = cv.cvtColor(stop_color, cv.COLOR_BGR2GRAY)
            blur_s = cv.GaussianBlur(gray_s, (5, 5), 0)
            edges_s = cv.Canny(blur_s, 50, 150)
            lines_stop = cv.HoughLinesP(edges_s, 1, np.pi/180, threshold=50, minLineLength=80, maxLineGap=10)
            
            stop_flag = 0.0
            if lines_stop is not None:
                for line in lines_stop:
                    _, y1, _, y2 = line[0]
                    if abs(y1 - y2) < 10:
                        stop_flag = 1.0
                        break

            # 4. PUBLIKACJA DO GUI I MÓZGU
            telemetry = Float32MultiArray()
            telemetry.data = [float(lx), float(rx), stop_flag, pedestrian_flag, float(left_px), float(right_px)]
            self.telemetry_publisher.publish(telemetry)

        except Exception as e:
            self.get_logger().error(f"Vision Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionSlave()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()