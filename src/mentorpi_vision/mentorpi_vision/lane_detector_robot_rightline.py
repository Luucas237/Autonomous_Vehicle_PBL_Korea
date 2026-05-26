#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32MultiArray, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data  

import cv2 as cv
import numpy as np
import time

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_master_gui')
        self.bridge = CvBridge()
        
        self.frame_subscriber = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.robot_listener_callback, qos_profile_sensor_data)
        self.telemetry_subscriber = self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.telemetry_callback, qos_profile_sensor_data)
        
        self.offset_value_publisher_ = self.create_publisher(Float32, '/vision/offset_raw', qos_profile_sensor_data)
        self.color_range_publisher = self.create_publisher(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', 10)

        self.latest_robot_frame = None
        self.last_robot_time = 0.0
        self.latest_telemetry = None
        
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.engines_on = True

        self.frame_w, self.frame_h = 640, 480
        self.grid_w, self.grid_h = 480, 360
        self.panel_w = 320 

        self.target_bgr = (0, 255, 255)
        # Domyślne wartości startowe GUI
        self.lower_color = np.array([15, 50, 180], dtype="uint8")
        self.upper_color = np.array([35, 255, 255], dtype="uint8")
        
        self.white_lower = np.array([55, 0, 210], dtype="uint8")
        self.white_upper = np.array([179, 255, 255], dtype="uint8")

        self.click_zones = {}

        self.window_name = "HiWonder Control Center (Korean Merged)"
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        self.gui_timer = self.create_timer(0.033, self.main_update_loop)
        self.get_logger().info('GUI loaded - Clean UI Version z logowaniem HSV do konsoli.')

    def telemetry_callback(self, msg):
        self.latest_telemetry = msg.data

    def robot_listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_robot_frame = cv.resize(frame, (self.frame_w, self.frame_h))
            self.last_robot_time = time.time()
        except Exception:
            pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Kliknięcie w obraz kamery - Pipeta HSV dla ZÓŁTYCH LINII
            if self.panel_w <= x < self.panel_w + self.grid_w and 0 <= y < self.grid_h:
                if self.latest_robot_frame is not None:
                    orig_x = int((x - self.panel_w) * self.frame_w / self.grid_w)
                    orig_y = int(y * self.frame_h / self.grid_h)
                    bgr_pixel = self.latest_robot_frame[orig_y, orig_x]
                    hsv_pixel = cv.cvtColor(np.uint8([[bgr_pixel]]), cv.COLOR_BGR2HSV)[0][0]
                    h, s, v = hsv_pixel
                    
                    self.lower_color = np.array([max(0, h - 10), max(0, s - 50), max(0, v - 50)], dtype="uint8")
                    self.upper_color = np.array([min(179, h + 10), 255, 255], dtype="uint8")
                    self.target_bgr = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
                    
                    # Wypisanie gotowego kodu do konsoli (Zmienione na get_logger)
                    raport = (
                        "\n============================================================\n"
                        "SKOPIUJ PONIŻSZY KOD DO lane_detector_robot.py (na robocie):\n"
                        f"self.yellow_lower = np.array([{self.lower_color[0]}, {self.lower_color[1]}, {self.lower_color[2]}], dtype='uint8')\n"
                        f"self.yellow_upper = np.array([{self.upper_color[0]}, {self.upper_color[1]}, {self.upper_color[2]}], dtype='uint8')\n"
                        "============================================================"
                    )
                    self.get_logger().info(raport)

                    msg_color = Int32MultiArray()
                    msg_color.data = [int(self.lower_color[0]), int(self.lower_color[1]), int(self.lower_color[2]), 
                                      int(self.upper_color[0]), int(self.upper_color[1]), int(self.upper_color[2])]
                    self.color_range_publisher.publish(msg_color)
            
            # Kliknięcie w lewy panel sterowania
            elif x < self.panel_w:
                for key, rect in self.click_zones.items():
                    rx1, ry1, rx2, ry2 = rect
                    if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                        if key == 'BTN_ON': self.engines_on = True
                        elif key == 'BTN_OFF': self.engines_on = False
                        elif key == 'BTN_STOP':
                            self.engines_on = False
                            msg = Float32(); msg.data = 999.0
                            for _ in range(3): self.offset_value_publisher_.publish(msg)
                        break

    def local_pipeline(self, frame):
        # 1. Maska Żółta (Linie)
        crop_img = frame[300:, :]
        hsv_y = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
        mask_yellow = cv.inRange(hsv_y, self.lower_color, self.upper_color)
        
        # 2. Maska Biała (Stop) - Wymagane przez koreańską logikę do skrzyżowań
        stop_img = frame[400:, :]
        hsv_s = cv.cvtColor(stop_img, cv.COLOR_BGR2HSV)
        mask_white = cv.inRange(hsv_s, self.white_lower, self.white_upper)
        
        mask_y_bgr = np.zeros_like(frame)
        mask_y_bgr[300:, :] = cv.cvtColor(mask_yellow, cv.COLOR_GRAY2BGR)
        
        mask_w_bgr = np.zeros_like(frame)
        mask_w_bgr[400:, :] = cv.cvtColor(mask_white, cv.COLOR_GRAY2BGR)

        edges = cv.Canny(cv.GaussianBlur(mask_yellow, (5, 5), 0), 50, 150)
        canny_bgr = np.zeros_like(frame)
        canny_bgr[300:, :] = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        return mask_y_bgr, mask_w_bgr, canny_bgr

    def draw_telemetry(self, frame, tel_data):
        if tel_data is not None and len(tel_data) >= 4:
            lx, rx, stop_line, pedestrian = tel_data[0], tel_data[1], tel_data[2], tel_data[3]
            
            cv.circle(frame, (int(lx), 320), 8, (255, 0, 0), -1)
            cv.circle(frame, (int(rx), 320), 8, (0, 0, 255), -1)
            cv.putText(frame, f"LX: {int(lx)}", (int(lx)-20, 310), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(frame, f"RX: {int(rx)}", (int(rx)-20, 310), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if stop_line == 1.0:
                cv.line(frame, (0, 440), (640, 440), (0, 255, 255), 4)
                cv.putText(frame, "STOP LINE DETECTED", (180, 430), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
                
            if pedestrian == 1.0:
                cv.putText(frame, "!!! PEDESTRIAN !!!", (180, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        return frame

    def main_update_loop(self):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_fps_time + 0.0001)
        self.last_fps_time = current_time
        cv.waitKey(1)

        if not self.engines_on:
            msg = Float32(); msg.data = 999.0
            self.offset_value_publisher_.publish(msg)

        grid_view = np.zeros((self.grid_h * 2, self.grid_w * 2, 3), dtype=np.uint8)

        if self.latest_robot_frame is not None and (current_time - self.last_robot_time) < 1.0:
            base_frame = self.latest_robot_frame.copy()
            mask_y_bgr, mask_w_bgr, canny_bgr = self.local_pipeline(base_frame)
            rgb_bgr = self.draw_telemetry(base_frame, self.latest_telemetry)

            cv.putText(rgb_bgr, "1. RGB + TELEMETRY", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            tl = cv.resize(rgb_bgr, (self.grid_w, self.grid_h))

            cv.putText(mask_y_bgr, "2. YELLOW MASK (Lines)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tr = cv.resize(mask_y_bgr, (self.grid_w, self.grid_h))

            cv.putText(mask_w_bgr, "3. WHITE MASK (Stop Line)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            bl = cv.resize(mask_w_bgr, (self.grid_w, self.grid_h))

            cv.putText(canny_bgr, "4. CANNY EDGES", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            br = cv.resize(canny_bgr, (self.grid_w, self.grid_h))

            grid_view[0:self.grid_h, 0:self.grid_w] = tl
            grid_view[0:self.grid_h, self.grid_w:self.grid_w*2] = tr
            grid_view[self.grid_h:self.grid_h*2, 0:self.grid_w] = bl
            grid_view[self.grid_h:self.grid_h*2, self.grid_w:self.grid_w*2] = br
        else:
            cv.putText(grid_view, "ROBOT OFFLINE", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        left_panel = np.zeros((self.grid_h * 2, self.panel_w, 3), dtype=np.uint8)
        left_panel[:] = (35, 35, 35)

        cv.putText(left_panel, "QUAD CONTROL", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.line(left_panel, (10, 45), (310, 45), (100, 100, 100), 2)

        cv.putText(left_panel, "TARGET COLOR (LClick Pipeta):", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.rectangle(left_panel, (230, 60), (300, 95), self.target_bgr, -1)
        cv.rectangle(left_panel, (230, 60), (300, 95), (255, 255, 255), 2) 

        self.click_zones.clear()
        mot_y = 130
        cv.putText(left_panel, "MOTOR CONTROL:", (10, mot_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        on_col = (0, 200, 0) if self.engines_on else (50, 50, 50)
        cv.rectangle(left_panel, (10, mot_y + 10), (150, mot_y + 50), on_col, -1)
        cv.rectangle(left_panel, (10, mot_y + 10), (150, mot_y + 50), (255, 255, 255), 2)
        cv.putText(left_panel, "ON", (65, mot_y + 36), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.click_zones['BTN_ON'] = (10, mot_y + 10, 150, mot_y + 50)

        off_col = (0, 0, 200) if not self.engines_on else (50, 50, 50)
        cv.rectangle(left_panel, (160, mot_y + 10), (310, mot_y + 50), off_col, -1)
        cv.rectangle(left_panel, (160, mot_y + 10), (310, mot_y + 50), (255, 255, 255), 2)
        cv.putText(left_panel, "OFF", (215, mot_y + 36), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.click_zones['BTN_OFF'] = (160, mot_y + 10, 310, mot_y + 50)

        cv.rectangle(left_panel, (10, mot_y + 60), (310, mot_y + 100), (0, 100, 200), -1) 
        cv.rectangle(left_panel, (10, mot_y + 60), (310, mot_y + 100), (255, 255, 255), 2)
        cv.putText(left_panel, "CENTER & STOP", (70, mot_y + 86), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.click_zones['BTN_STOP'] = (10, mot_y + 60, 310, mot_y + 100)

        cv.putText(left_panel, f"FPS: {self.fps:.1f}", (10, mot_y + 140), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.latest_telemetry is not None and len(self.latest_telemetry) >= 4:
            lx, rx = self.latest_telemetry[0], self.latest_telemetry[1]
            cv.putText(left_panel, f"LX: {int(lx)} | RX: {int(rx)}", (10, mot_y + 170), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        if not self.engines_on:
            cv.putText(left_panel, "ENGINES STOPPED", (10, mot_y + 210), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        final_ui = np.hstack((left_panel, grid_view))
        cv.imshow(self.window_name, final_ui)

def main(args=None):
    rclpy.init(args=args)
    node = ProcessFrame()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()