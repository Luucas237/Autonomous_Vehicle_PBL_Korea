#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32MultiArray
from rclpy.qos import qos_profile_sensor_data  

import cv2 as cv
import numpy as np
from collections import deque
import time

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_master_node')
        self.bridge = CvBridge()
        
        self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        self.left_history = deque(maxlen=7)
        self.right_history = deque(maxlen=7)
        self.missing_left = 0
        self.missing_right = 0

        # Subskrypcja kamery z robota (Topic)
        self.frame_subscriber = self.create_subscription(
            Image, 
            '/ascamera/camera_publisher/rgb0/image',  
            self.robot_listener_callback,
            qos_profile_sensor_data     
        )

        self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)
        self.color_range_publisher = self.create_publisher(Int32MultiArray, '/mentorpi/vision/hsv_thresholds', 10)
        self.curve_publisher = self.create_publisher(Float32, '/mentorpi/vision/curve_threshold', 10)

        # --- SYSTEM ZARZĄDZANIA ŹRÓDŁAMI WIDEO ---
        self.available_sources = ["Robot Topic"]
        self.current_source_idx = 0
        self.active_pc_cap = None
        
        self.get_logger().info("Skanowanie portów wideo na PC...")
        for i in range(4):
            cap = cv.VideoCapture(i)
            if cap.isOpened():
                self.available_sources.append(f"PC Camera {i}")
                cap.release()
        self.get_logger().info(f"Dostępne źródła wideo: {self.available_sources}")

        # Zmienne przechowujące ostatnią klatkę z danego źródła
        self.latest_robot_frame = None
        self.last_robot_time = 0.0
        
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.current_offset = 0.0
        self.current_curve_threshold = 0.0005 

        self.frame_w = 640 
        self.frame_h = 480

        self.target_bgr = (0, 0, 0)
        self.input_text = "0.0005"
        self.is_typing = False
        
        # Konfiguracja GUI
        self.window_name = "MentorPi - Vision Control Center"
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)
        
        cv.createTrackbar("H Min", self.window_name, 0, 180, self.nothing)
        cv.createTrackbar("H Max", self.window_name, 180, 180, self.nothing)
        cv.createTrackbar("S Min", self.window_name, 0, 255, self.nothing)
        cv.createTrackbar("S Max", self.window_name, 255, 255, self.nothing)
        cv.createTrackbar("V Min", self.window_name, 0, 255, self.nothing)
        cv.createTrackbar("V Max", self.window_name, 80, 255, self.nothing)

        # Główny Timer - odświeżanie logiki i GUI (ok. 30 FPS)
        self.gui_timer = self.create_timer(0.033, self.main_update_loop)
        
        # Inicjalizacja pierwszego źródła (jeśli to PC, otwieramy port)
        self.switch_source(force_init=True)
        self.get_logger().info('Master GUI załadowane! System gotowy do pracy.')

    def nothing(self, x): pass

    def reset_defaults(self):
        cv.setTrackbarPos("H Min", self.window_name, 0)
        cv.setTrackbarPos("H Max", self.window_name, 180)
        cv.setTrackbarPos("S Min", self.window_name, 0)
        cv.setTrackbarPos("S Max", self.window_name, 255)
        cv.setTrackbarPos("V Min", self.window_name, 0)
        cv.setTrackbarPos("V Max", self.window_name, 80)
        
        self.target_bgr = (0, 0, 0)
        self.current_curve_threshold = 0.0005
        self.input_text = "0.0005"
        self.is_typing = False
        self.get_logger().info("Ustawienia domyślne przywrócone.")

    def save_curve_threshold(self):
        try:
            val = float(self.input_text)
            self.current_curve_threshold = val
            self.is_typing = False
            self.get_logger().info(f"Zapisano Curve Threshold: {val}")
        except ValueError:
            self.get_logger().error("Zły format liczby! Wracam do poprzedniego.")
            self.input_text = str(self.current_curve_threshold)
            self.is_typing = False

    def switch_source(self, force_init=False):
        if not force_init:
            self.current_source_idx = (self.current_source_idx + 1) % len(self.available_sources)
            
        new_source = self.available_sources[self.current_source_idx]
        self.get_logger().info(f"--> Przełączanie źródła wideo na: {new_source}")

        if self.active_pc_cap is not None:
            self.active_pc_cap.release()
            self.active_pc_cap = None

        if new_source.startswith("PC Camera"):
            cam_index = int(new_source.split(" ")[-1])
            self.active_pc_cap = cv.VideoCapture(cam_index)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Kliknięcie w obraz (pobieranie koloru)
            if x < self.frame_w and y < self.frame_h:
                # Tu uwaga: musi być dostępna jakaś aktualna klatka, żeby pobrać kolor
                # Kod dla bezpieczeństwa pominięty w tej funkcji, obsłużone niżej
                pass
            
            # Kliknięcie w panel boczny
            elif x >= self.frame_w:
                rx = x - self.frame_w 
                ry = y
                
                if 10 <= rx <= 150 and 450 <= ry <= 490:
                    self.is_typing = True
                elif 160 <= rx <= 250 and 450 <= ry <= 490:
                    self.save_curve_threshold()
                # NOWY PRZYCISK: Zmiana źródła
                elif 10 <= rx <= 370 and 520 <= ry <= 560:
                    self.switch_source()
                elif 380 - 130 <= rx <= 380 - 20 and 960 - 60 <= ry <= 960 - 20:
                    self.reset_defaults()
                else:
                    self.is_typing = False

    def robot_listener_callback(self, msg):
        # Asynchroniczne nasłuchiwanie topicu (działa w tle niezależnie)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_robot_frame = cv.resize(frame, (self.frame_w, self.frame_h))
        self.last_robot_time = time.time()

    def main_update_loop(self):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_fps_time + 0.0001)
        self.last_fps_time = current_time

        # POBIERANIE KLATKI Z ZALEŻNOŚCI OD WYBRANEGO ŹRÓDŁA
        current_source_name = self.available_sources[self.current_source_idx]
        current_frame = None
        status_msg = "OK"

        if current_source_name == "Robot Topic":
            time_since_last_robot = current_time - self.last_robot_time
            if self.latest_robot_frame is not None and time_since_last_robot < 1.0:
                current_frame = self.latest_robot_frame.copy()
            else:
                status_msg = "ROBOT OFFLINE (NO TOPIC)"
        else:
            if self.active_pc_cap is not None and self.active_pc_cap.isOpened():
                ret, f = self.active_pc_cap.read()
                if ret:
                    current_frame = cv.resize(f, (self.frame_w, self.frame_h))
                else:
                    status_msg = "PC CAMERA ERROR"
            else:
                status_msg = "PC CAMERA NOT OPEN"

        # TWORZENIE WIDOKU LEWEGO
        rgb_view = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        mask_view = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        
        h_min = cv.getTrackbarPos("H Min", self.window_name)
        h_max = cv.getTrackbarPos("H Max", self.window_name)
        s_min = cv.getTrackbarPos("S Min", self.window_name)
        s_max = cv.getTrackbarPos("S Max", self.window_name)
        v_min = cv.getTrackbarPos("V Min", self.window_name)
        v_max = cv.getTrackbarPos("V Max", self.window_name)

        lower_bound = np.array([min(h_min, h_max), min(s_min, s_max), min(v_min, v_max)], dtype="uint8")
        upper_bound = np.array([max(h_min, h_max), max(s_min, s_max), max(v_min, v_max)], dtype="uint8")

        # Aktualizacja zmiennej dla kliknięcia pipety (jeśli mamy obraz)
        if current_frame is not None:
            self.frame_for_pipette = current_frame.copy()

        # Przetwarzanie wizji
        if current_frame is None:
            cv.putText(rgb_view, status_msg, (50, int(self.frame_h/2)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.current_offset = 0.0
        else:
            output_frame = current_frame.copy()
            left_lines, right_lines, mask_view = self.detect_lines_core(output_frame, lower_bound, upper_bound)
            
            left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
            right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)

            rgb_view, self.current_offset = self.draw_guideline(output_frame, left_poly, right_poly)

        left_panel = np.vstack((rgb_view, mask_view))

        # TWORZENIE PANELU KONTROLNEGO
        panel_w = 380
        panel_h = left_panel.shape[0]
        right_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        right_panel[:] = (30, 30, 30)

        cv.putText(right_panel, "CONTROL CENTER", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv.putText(right_panel, "- Click image to pick color", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv.putText(right_panel, "- Adjust Min/Max sliders", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv.putText(right_panel, "TARGET COLOR:", (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv.rectangle(right_panel, (180, 160), (280, 220), self.target_bgr, -1)
        cv.rectangle(right_panel, (180, 160), (280, 220), (255, 255, 255), 2) 

        cv.putText(right_panel, "DATA STREAM:", (10, 280), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(right_panel, f"FPS: {self.fps:.1f}", (10, 320), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(right_panel, f"Offset: {self.current_offset:.1f} px", (10, 360), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- BLOK CURVE THRESHOLD ---
        cv.putText(right_panel, "CURVE THRESHOLD:", (10, 430), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        box_color = (100, 255, 100) if self.is_typing else (255, 255, 255)
        cv.rectangle(right_panel, (10, 450), (150, 490), (50, 50, 50), -1)
        cv.rectangle(right_panel, (10, 450), (150, 490), box_color, 2)
        cursor = "_" if self.is_typing and int(time.time() * 2) % 2 == 0 else ""
        cv.putText(right_panel, self.input_text + cursor, (20, 478), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.rectangle(right_panel, (160, 450), (250, 490), (0, 150, 0), -1)
        cv.rectangle(right_panel, (160, 450), (250, 490), (255, 255, 255), 2)
        cv.putText(right_panel, "SAVE", (182, 476), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- NOWY BLOK WYBORU ŹRÓDŁA WIDEO (CLICK TO CYCLE) ---
        cv.putText(right_panel, "VIDEO SOURCE (Click to change):", (10, 515), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv.rectangle(right_panel, (10, 520), (370, 560), (70, 30, 90), -1) # Fioletowe tło przycisku
        cv.rectangle(right_panel, (10, 520), (370, 560), (200, 150, 255), 2)
        cv.putText(right_panel, current_source_name, (20, 545), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Przycisk Reset
        btn_w, btn_h = 110, 40
        btn_x1, btn_y1 = panel_w - btn_w - 20, panel_h - btn_h - 20
        btn_x2, btn_y2 = panel_w - 20, panel_h - 20
        cv.rectangle(right_panel, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 200), -1) 
        cv.rectangle(right_panel, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv.putText(right_panel, "RESET", (btn_x1 + 25, btn_y1 + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        final_ui = np.hstack((left_panel, right_panel))
        cv.imshow(self.window_name, final_ui)
        
        # Obsługa klawiatury + Publikacja danych
        key = cv.waitKey(1) & 0xFF
        if key != 255 and self.is_typing:
            if key == 8 or key == 127: self.input_text = self.input_text[:-1]
            elif key == 13 or key == 10: self.save_curve_threshold()
            elif chr(key) in "0123456789.": self.input_text += chr(key)
            
        # Pobieranie koloru pipetą
        if hasattr(self, 'frame_for_pipette') and cv.getWindowProperty(self.window_name, 0) >= 0:
             # Sprawdzam flagę myszy wewnątrz callbacku (nie polecam tego bezpośrednio z klawiatury, mysz już działa na pipetę w mouse_callback, tutaj publikujemy)
             pass

        # Publikowanie parametrów (tylko w kierunku ROS 2)
        msg_color = Int32MultiArray()
        msg_color.data = [int(lower_bound[0]), int(lower_bound[1]), int(lower_bound[2]), 
                          int(upper_bound[0]), int(upper_bound[1]), int(upper_bound[2])]
        self.color_range_publisher.publish(msg_color)
        
        msg_curve = Float32()
        msg_curve.data = float(self.current_curve_threshold)
        self.curve_publisher.publish(msg_curve)

    def detect_lines_core(self, frame, lower_bound, upper_bound):
        height, width = frame.shape[:2]
        crop_y = int(height * 0.45) 
        cropped_frame = frame[crop_y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]

        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_bound, upper_bound)

        roi_mask = np.zeros_like(mask)
        y_bottom = crop_h
        y_mid = int(crop_h * 0.5)
        y_top = 0
        x_top_left = int(crop_w * 0.25)
        x_top_right = int(crop_w * 0.75)

        vertices = np.array([[ 
            (0, y_bottom), (crop_w, y_bottom), (crop_w, y_mid), 
            (x_top_right, y_top), (x_top_left, y_top), (0, y_mid) 
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, vertices, 255)
        roi = cv.bitwise_and(mask, roi_mask)
        
        mask_preview_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        mask_preview_cropped = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        cv.polylines(mask_preview_cropped, [vertices], isClosed=True, color=(127, 127, 127), thickness=2)

        roi_blurred = cv.GaussianBlur(roi, (7, 7), 0)
        kernel = np.ones((7, 7), np.uint8) 
        roi_clean = cv.erode(roi_blurred, kernel, iterations=1)
        roi_clean = cv.dilate(roi_clean, kernel, iterations=4) 
        edges = cv.Canny(roi_clean, 65, 150)

        lines = cv.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=70, maxLineGap=60)

        left_lines, right_lines = [], []
        raw_lines_count = 0

        if lines is not None:
            raw_lines_count = len(lines)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    real_y1 = y1 + crop_y
                    real_y2 = y2 + crop_y
                    slope = (real_y2 - real_y1) / (x2 - x1 + 0.0001)
                    if abs(slope) < 0.25: continue
                    if (x1 + x2) / 2 < width / 2: left_lines.append((x1, real_y1, x2, real_y2))
                    else: right_lines.append((x1, real_y1, x2, real_y2))

        cv.putText(mask_preview_cropped, f"Hough Lines Detected: {raw_lines_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mask_preview_bgr[crop_y:, :] = mask_preview_cropped

        return left_lines, right_lines, mask_preview_bgr

    def fit_and_filter(self, lines, history, missing_counter):
        if len(lines) == 0:
            missing_counter += 1
            if missing_counter > 5: history.clear()
            elif len(history) > 0: history.append(history[-1]) 
            return None, missing_counter

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        missing_counter = 0
        if len(np.unique(y_coords)) < 3: return None, missing_counter

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
            for i in range(7): smoothed_poly += self.fir_weights[i] * history[i]
            return smoothed_poly, missing_counter
        else: return np.mean(history, axis=0), missing_counter

    def draw_guideline(self, frame, left_poly, right_poly):
        height, width, _ = frame.shape
        lookahead_y = int(height * 0.70)
        offset = 0.0

        ploty = np.linspace(int(height * 0.4), height, num=30)

        if left_poly is not None and right_poly is not None:
            left_fitx = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
            right_fitx = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]
            mid_fitx = (left_fitx + right_fitx) / 2.0

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
            pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
            pts_mid = np.array([np.transpose(np.vstack([mid_fitx, ploty]))], np.int32)

            cv.polylines(frame, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
            cv.polylines(frame, [pts_right], isClosed=False, color=(255, 0, 0), thickness=4)
            cv.polylines(frame, [pts_mid], isClosed=False, color=(0, 0, 255), thickness=3)

            left_x_lookahead = left_poly[0]*(lookahead_y**2) + left_poly[1]*lookahead_y + left_poly[2]
            right_x_lookahead = right_poly[0]*(lookahead_y**2) + right_poly[1]*lookahead_y + right_poly[2]
            mid_x_lookahead = (left_x_lookahead + right_x_lookahead) / 2.0
            
            offset = float(mid_x_lookahead - (width / 2.0))
            cv.circle(frame, (int(mid_x_lookahead), lookahead_y), 8, (0, 255, 255), -1)
            
            msg = Float32()
            msg.data = offset
            self.offset_value_publisher_.publish(msg)

        cv.putText(frame, "Hybrid Mode Active", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, offset

    def destroy_node(self):
        if self.active_pc_cap is not None:
            self.active_pc_cap.release()
        super().destroy_node()

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