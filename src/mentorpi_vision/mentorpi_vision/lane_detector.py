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
import math

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_master_node')
        self.bridge = CvBridge()
        
        self.frame_subscriber = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.robot_listener_callback, qos_profile_sensor_data)
        self.telemetry_subscriber = self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.telemetry_callback, qos_profile_sensor_data)
        
        self.offset_value_publisher_ = self.create_publisher(Float32, '/vision/offset_raw', qos_profile_sensor_data)
        self.color_range_publisher = self.create_publisher(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', 10)
        self.algo_params_publisher = self.create_publisher(Float32MultiArray, '/mentorpi/vision/algo_params', 10)

        self.latest_robot_frame = None
        self.last_robot_time = 0.0
        self.latest_telemetry = None
        
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.current_offset = 0.0
        self.engines_on = True

        self.frame_w, self.frame_h = 640, 480
        self.grid_w, self.grid_h = 480, 360
        self.panel_w = 320 

        self.target_bgr = (0, 255, 255)
        self.lower_color = np.array([20, 80, 80], dtype="uint8")
        self.upper_color = np.array([40, 255, 255], dtype="uint8")

        self.params = {
            'erode': 1.0, 'dilate': 1.0, 
            'canny_min': 30.0, 'canny_max': 120.0,
            'hough_thr': 15.0, 'hough_min': 15.0, 'hough_max': 50.0,
            'curve_thr': 0.0005
        }
        self.input_texts = {k: str(v) for k, v in self.params.items()}
        self.active_input = None
        self.click_zones = {}
        
        self.measure_pt1 = None
        self.measure_dist = 0.0

        self.window_name = "Vision Quad Control Center"
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)

        self.gui_timer = self.create_timer(0.033, self.main_update_loop)
        self.get_logger().info('Quad GUI loaded.')

    def telemetry_callback(self, msg):
        self.latest_telemetry = msg.data

    def robot_listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_robot_frame = cv.resize(frame, (self.frame_w, self.frame_h))
            self.last_robot_time = time.time()
        except Exception:
            pass

    def send_algo_params(self):
        try:
            for k in self.params.keys():
                self.params[k] = float(self.input_texts[k])
            msg = Float32MultiArray()
            msg.data = [self.params['erode'], self.params['dilate'], self.params['canny_min'], self.params['canny_max'],
                        self.params['hough_thr'], self.params['hough_min'], self.params['hough_max'], self.params['curve_thr']]
            self.algo_params_publisher.publish(msg)
            self.active_input = None
        except ValueError:
            pass

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
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
                    
                    msg_color = Int32MultiArray()
                    msg_color.data = [int(self.lower_color[0]), int(self.lower_color[1]), int(self.lower_color[2]), 
                                      int(self.upper_color[0]), int(self.upper_color[1]), int(self.upper_color[2])]
                    self.color_range_publisher.publish(msg_color)
            elif x < self.panel_w:
                self.active_input = None
                for key, rect in self.click_zones.items():
                    rx1, ry1, rx2, ry2 = rect
                    if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                        if key == 'BTN_SEND': self.send_algo_params()
                        elif key == 'BTN_ON': self.engines_on = True
                        elif key == 'BTN_OFF': self.engines_on = False
                        elif key == 'BTN_STOP':
                            self.engines_on = False
                            msg = Float32(); msg.data = 999.0
                            for _ in range(3): self.offset_value_publisher_.publish(msg)
                        else: self.active_input = key 
                        break
        
        elif event == cv.EVENT_RBUTTONDOWN:
            if self.panel_w <= x < self.panel_w + self.grid_w and 0 <= y < self.grid_h:
                orig_x = int((x - self.panel_w) * self.frame_w / self.grid_w)
                orig_y = int(y * self.frame_h / self.grid_h)
                if self.measure_pt1 is None:
                    self.measure_pt1 = (orig_x, orig_y)
                    self.get_logger().info("Pomiar: Pkt 1 ustawiony.")
                else:
                    self.measure_dist = math.sqrt((orig_x - self.measure_pt1[0])**2 + (orig_y - self.measure_pt1[1])**2)
                    self.get_logger().info(f"Pomiar ZAKOŃCZONY: {self.measure_dist:.1f} px")
                    self.measure_pt1 = None

    def local_pipeline(self, frame):
        height, width = frame.shape[:2]
        crop_y = int(height * 0.55) 
        cropped_frame = frame[crop_y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]

        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_color, self.upper_color)

        roi_mask = np.zeros_like(mask)
        x_top_left, x_top_right = int(crop_w * 0.3), int(crop_w * 0.7)  
        side_height = 135
        vertices = np.array([[ 
            (0, crop_h), (crop_w, crop_h), (crop_w, crop_h - side_height),        
            (x_top_right, 0), (x_top_left, 0), (0, crop_h - side_height)              
        ]], dtype=np.int32)
        cv.fillPoly(roi_mask, vertices, 255)
        roi = cv.bitwise_and(mask, roi_mask)
        
        mask_roi_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        mask_roi_cropped = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        cv.polylines(mask_roi_cropped, [vertices], True, (50, 50, 50), 2)
        mask_roi_bgr[crop_y:, :] = mask_roi_cropped

        try: e_it, d_it = int(self.params['erode']), int(self.params['dilate'])
        except: e_it, d_it = 1, 1
        kernel = np.ones((5, 5), np.uint8) 
        roi_morph = cv.erode(roi, kernel, iterations=e_it)
        roi_morph = cv.dilate(roi_morph, kernel, iterations=d_it) 
        morph_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        morph_bgr[crop_y:, :] = cv.cvtColor(roi_morph, cv.COLOR_GRAY2BGR)

        try: c_min, c_max = int(self.params['canny_min']), int(self.params['canny_max'])
        except: c_min, c_max = 30, 120
        edges = cv.Canny(roi_morph, c_min, c_max)
        canny_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        canny_bgr[crop_y:, :] = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        try: h_thr, h_min, h_max = int(self.params['hough_thr']), int(self.params['hough_min']), int(self.params['hough_max'])
        except: h_thr, h_min, h_max = 15, 15, 50
        lines = cv.HoughLinesP(edges, 1, np.pi/180, h_thr, minLineLength=h_min, maxLineGap=h_max)
        hough_count = len(lines) if lines is not None else 0

        return mask_roi_bgr, morph_bgr, canny_bgr, hough_count

    def draw_telemetry_lines(self, frame, tel_data):
        height, width = frame.shape[:2]
        lookahead_y = int(height * 0.70)
        offset = 0.0
        LANE_WIDTH_PX = 320.0 

        if tel_data is not None and len(tel_data) >= 12:
            ploty = np.linspace(int(height * 0.55), height, num=30)
            l_poly, r_poly, mid_poly = None, None, None
            
            if tel_data[0] == 1.0:
                l_poly = np.array([tel_data[1], tel_data[2], tel_data[3]])
                l_fitx = l_poly[0]*ploty**2 + l_poly[1]*ploty + l_poly[2]
                cv.polylines(frame, [np.array([np.transpose(np.vstack([l_fitx, ploty]))], np.int32)], False, (255, 0, 0), 4)

            if tel_data[4] == 1.0:
                r_poly = np.array([tel_data[5], tel_data[6], tel_data[7]])
                r_fitx = r_poly[0]*ploty**2 + r_poly[1]*ploty + r_poly[2]
                cv.polylines(frame, [np.array([np.transpose(np.vstack([r_fitx, ploty]))], np.int32)], False, (255, 0, 0), 4)

            if l_poly is not None and r_poly is not None:
                mid_poly = (l_poly + r_poly) / 2.0
            elif l_poly is not None:
                mid_poly = np.copy(l_poly)
                mid_poly[2] += (LANE_WIDTH_PX / 2.0)
            elif r_poly is not None:
                mid_poly = np.copy(r_poly)
                mid_poly[2] -= (LANE_WIDTH_PX / 2.0)

            if mid_poly is not None:
                mid_fitx = mid_poly[0]*ploty**2 + mid_poly[1]*ploty + mid_poly[2]
                cv.polylines(frame, [np.array([np.transpose(np.vstack([mid_fitx, ploty]))], np.int32)], False, (0, 0, 255), 3)

            offset = tel_data[8]
            bumper_active = (tel_data[9] == 1.0)
            left_px = int(tel_data[10])
            right_px = int(tel_data[11])
            
            mid_x_lookahead = offset + (width / 2.0)
            cv.circle(frame, (int(mid_x_lookahead), lookahead_y), 10, (0, 255, 255), -1)
            
            left_box_color = (0, 0, 255) if left_px > 1200 else (0, 255, 0)
            cv.rectangle(frame, (0, height - 70), (180, height), left_box_color, 2)
            cv.putText(frame, f"L: {left_px}", (10, height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, left_box_color, 2)

            right_box_color = (0, 0, 255) if right_px > 1200 else (0, 255, 0)
            cv.rectangle(frame, (width - 180, height - 70), (width, height), right_box_color, 2)
            cv.putText(frame, f"R: {right_px}", (width - 170, height - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, right_box_color, 2)
            
            if bumper_active:
                cv.putText(frame, "BUMPER ACTIVE!", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        if self.measure_dist > 0:
            cv.putText(frame, f"Width: {self.measure_dist:.1f} px", (width - 250, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 3)

        return frame, offset

    def main_update_loop(self):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_fps_time + 0.0001)
        self.last_fps_time = current_time

        key = cv.waitKey(1) & 0xFF
        if key != 255 and self.active_input is not None:
            if key == 8 or key == 127: self.input_texts[self.active_input] = self.input_texts[self.active_input][:-1]
            elif key == 13 or key == 10: self.send_algo_params()
            elif chr(key) in "0123456789.-": self.input_texts[self.active_input] += chr(key)

        if not self.engines_on:
            msg = Float32(); msg.data = 999.0
            self.offset_value_publisher_.publish(msg)

        grid_view = np.zeros((self.grid_h * 2, self.grid_w * 2, 3), dtype=np.uint8)

        if self.latest_robot_frame is not None and (current_time - self.last_robot_time) < 1.0:
            base_frame = self.latest_robot_frame.copy()
            mask_bgr, morph_bgr, canny_bgr, lines_count = self.local_pipeline(base_frame)
            rgb_bgr, self.current_offset = self.draw_telemetry_lines(base_frame, self.latest_telemetry)

            cv.putText(rgb_bgr, "1. RGB + TELEMETRY", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            tl = cv.resize(rgb_bgr, (self.grid_w, self.grid_h))

            cv.putText(mask_bgr, f"2. ROI MASK (Lines: {lines_count})", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tr = cv.resize(mask_bgr, (self.grid_w, self.grid_h))

            cv.putText(morph_bgr, "3. MORPHOLOGY (Erode->Dilate)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            bl = cv.resize(morph_bgr, (self.grid_w, self.grid_h))

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

        cv.putText(left_panel, "TARGET COLOR:", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv.rectangle(left_panel, (160, 60), (250, 95), self.target_bgr, -1)
        cv.rectangle(left_panel, (160, 60), (250, 95), (255, 255, 255), 2) 
        cv.putText(left_panel, "(LClick: Color | RClick: Measure)", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        self.click_zones.clear()
        labels = [
            ('ERODE ITER:', 'erode'), ('DILATE ITER:', 'dilate'),
            ('CANNY MIN:', 'canny_min'), ('CANNY MAX:', 'canny_max'),
            ('HOUGH THR:', 'hough_thr'), ('HOUGH MIN LEN:', 'hough_min'),
            ('HOUGH MAX GAP:', 'hough_max'), ('CURVE THR:', 'curve_thr')
        ]
        
        start_y = 150
        for i, (label_text, dict_key) in enumerate(labels):
            y_pos = start_y + (i * 45)
            cv.putText(left_panel, label_text, (10, y_pos + 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            
            box_x, box_y, box_w, box_h = 160, y_pos, 140, 30
            is_active = (self.active_input == dict_key)
            color_border = (0, 255, 0) if is_active else (255, 255, 255)
            color_bg = (50, 100, 50) if is_active else (50, 50, 50)
            
            cv.rectangle(left_panel, (box_x, box_y), (box_x + box_w, box_y + box_h), color_bg, -1)
            cv.rectangle(left_panel, (box_x, box_y), (box_x + box_w, box_y + box_h), color_border, 1)
            
            display_text = self.input_texts[dict_key]
            if is_active and int(time.time() * 2) % 2 == 0: display_text += "_"
            cv.putText(left_panel, display_text, (box_x + 5, box_y + 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.click_zones[dict_key] = (box_x, box_y, box_x + box_w, box_y + box_h)

        btn_y = start_y + (len(labels) * 45) + 10
        cv.rectangle(left_panel, (20, btn_y), (300, btn_y + 40), (0, 150, 0), -1)
        cv.rectangle(left_panel, (20, btn_y), (300, btn_y + 40), (255, 255, 255), 2)
        cv.putText(left_panel, "SEND PARAMS TO ROBOT", (45, btn_y + 26), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.click_zones['BTN_SEND'] = (20, btn_y, 300, btn_y + 40)

        mot_y = btn_y + 65
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
        cv.putText(left_panel, f"Offset: {self.current_offset:.1f} px", (10, mot_y + 170), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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