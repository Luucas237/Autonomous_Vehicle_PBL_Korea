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

        # Historia filtracji linii
        self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        self.left_history = deque(maxlen=7)
        self.right_history = deque(maxlen=7)
        self.missing_left = 0
        self.missing_right = 0
        self.last_offset = 0.0

        # Ochrona przed błędnym wykryciem linii poprzecznej / dziwnej na zakręcie
        self.bad_frame_counter = 0
        self.bad_frame_limit = 3

        # Typ linii: 0 = brak / ciągła / niepewna, 1 = przerywana
        self.left_line_type = 0.0
        self.right_line_type = 0.0

        # HSV dla koloru linii
        self.lower_color = np.array([20, 80, 80], dtype="uint8")
        self.upper_color = np.array([40, 255, 255], dtype="uint8")

        # Parametry algorytmu
        self.p_erode = 1
        self.p_dilate = 1
        self.p_canny_min = 30
        self.p_canny_max = 120
        self.p_hough_thr = 15
        self.p_hough_min = 15
        self.p_hough_max = 50
        self.current_curve_threshold = 0.0005

        # Prosty zderzak wizyjny z Twojej wcześniejszej wersji
        self.bumper_state = 'INACTIVE'
        self.bumper_start_time = 0.0
        self.bumper_dir = 0.0
        self.pixel_threshold = 1200

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
        self.algo_params_subscriber = self.create_subscription(
            Float32MultiArray,
            '/mentorpi/vision/algo_params',
            self.algo_params_callback,
            10
        )

        self.telemetry_publisher = self.create_publisher(
            Float32MultiArray,
            '/vision/lane_telemetry',
            10
        )

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info(
            'Vision Node [SLAVE] gotowy: linie ciągłe/przerywane + filtr fałszywych poprzecznych detekcji.'
        )

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
            self.get_logger().info(
                f"==> Change: Erode={self.p_erode}, Dilate={self.p_dilate}, "
                f"Canny={self.p_canny_min}-{self.p_canny_max}, HoughMin={self.p_hough_min}"
            )

    def color_callback(self, msg):
        data = msg.data
        if len(data) == 6:
            self.lower_color = np.array([data[0], data[1], data[2]], dtype="uint8")
            self.upper_color = np.array([data[3], data[4], data[5]], dtype="uint8")
            self.get_logger().info(f"==> Change color: Min={self.lower_color}, Max={self.upper_color}")

    def listener_callback(self, msg):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time + 0.0001)
        self.last_time = current_time

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.perform_detection(frame)
        except Exception as e:
            self.get_logger().warn(f"Vision error: {e}")

    def perform_detection(self, frame):
        left_lines, right_lines, roi_mask, bad_cross_lines = self.detect_lines_core(frame)

        left_poly, self.missing_left = self.fit_and_filter(
            left_lines,
            self.left_history,
            self.missing_left
        )
        right_poly, self.missing_right = self.fit_and_filter(
            right_lines,
            self.right_history,
            self.missing_right
        )

        self.calculate_and_log(
            frame.shape,
            left_poly,
            right_poly,
            roi_mask,
            bad_cross_lines
        )

    def detect_lines_core(self, frame):
        height, width = frame.shape[:2]
        crop_y = int(height * 0.55)
        cropped_frame = frame[crop_y:, :]
        crop_h, crop_w = cropped_frame.shape[:2]

        hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_color, self.upper_color)

        # ROI zostaje podobny jak w Twojej wersji
        roi_mask = np.zeros_like(mask)
        x_top_left = int(crop_w * 0.3)
        x_top_right = int(crop_w * 0.7)
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

        roi_blurred = cv.GaussianBlur(roi, (7, 7), 0)
        kernel = np.ones((5, 5), np.uint8)
        roi_clean = cv.erode(roi_blurred, kernel, iterations=self.p_erode)
        roi_clean = cv.dilate(roi_clean, kernel, iterations=self.p_dilate)

        edges = cv.Canny(roi_clean, self.p_canny_min, self.p_canny_max)
        lines = cv.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.p_hough_thr,
            minLineLength=self.p_hough_min,
            maxLineGap=self.p_hough_max
        )

        left_lines, right_lines = [], []
        bad_cross_lines = 0

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    real_y1 = y1 + crop_y
                    real_y2 = y2 + crop_y

                    dx = x2 - x1
                    dy = real_y2 - real_y1
                    length = np.hypot(dx, dy)

                    if length < 10:
                        continue

                    slope = dy / (dx + 0.0001)
                    x_mid = (x1 + x2) / 2.0
                    y_span = abs(dy)
                    x_span = abs(dx)

                    # ======================================================
                    # FILTR 1: odrzucenie poziomych/poprzecznych śmieci
                    # ======================================================
                    # Linia pasa w obrazie zwykle nie powinna być bardzo pozioma.
                    # Jeżeli Hough złapie długi poziomy segment, to na zakręcie
                    # potrafi rozwalić obliczenie środka pasa.
                    if x_span > 80 and y_span < 25:
                        bad_cross_lines += 1
                        continue

                    if abs(slope) < 0.25:
                        bad_cross_lines += 1
                        continue

                    # ======================================================
                    # FILTR 2: dziwna prawie pionowa kreska blisko środka
                    # ======================================================
                    # Czasami na zakręcie pojawia się fałszywy segment w środku.
                    # Jeżeli jest prawie pionowy i centralny, nie ufamy mu.
                    if abs(dx) < 8 and width * 0.35 < x_mid < width * 0.65:
                        bad_cross_lines += 1
                        continue

                    # ======================================================
                    # Normalny podział na lewą i prawą linię po znaku nachylenia
                    # ======================================================
                    if slope < 0:
                        left_lines.append((x1, real_y1, x2, real_y2))
                    else:
                        right_lines.append((x1, real_y1, x2, real_y2))

        # Klasyfikacja przerywana/ciągła na podstawie segmentów Hougha
        self.left_line_type = self.classify_dashed_line(left_lines)
        self.right_line_type = self.classify_dashed_line(right_lines)

        return left_lines, right_lines, roi_clean, bad_cross_lines

    def classify_dashed_line(self, lines):
        """
        0.0 -> raczej linia ciągła / brak pewności
        1.0 -> raczej linia przerywana

        Założenie praktyczne:
        - linia przerywana daje kilka krótszych segmentów,
        - między segmentami występują widoczne przerwy po osi y.
        """
        if lines is None or len(lines) < 3:
            return 0.0

        segments = []
        for x1, y1, x2, y2 in lines:
            length = np.hypot(x2 - x1, y2 - y1)
            y_mid = (y1 + y2) / 2.0
            segments.append((y_mid, length))

        segments.sort(key=lambda s: s[0])

        lengths = [s[1] for s in segments]
        y_mids = [s[0] for s in segments]

        avg_len = float(np.mean(lengths))
        gaps = []

        for i in range(1, len(y_mids)):
            gaps.append(abs(y_mids[i] - y_mids[i - 1]))

        if len(gaps) == 0:
            return 0.0

        avg_gap = float(np.mean(gaps))
        max_gap = float(np.max(gaps))

        # Progi robocze, które możesz później dostroić.
        # Gdy łapie Ci za dużo linii jako przerywane: zwiększ max_gap albo zmniejsz avg_len.
        # Gdy nie łapie przerywanych: obniż max_gap i avg_gap.
        if len(lines) >= 3 and avg_len < 85 and max_gap > 35 and avg_gap > 18:
            return 1.0

        return 0.0

    def fit_and_filter(self, lines, history, missing_counter):
        if len(lines) == 0:
            missing_counter += 1

            if missing_counter > 15:
                history.clear()
            elif len(history) > 0:
                # Przez krótki czas trzymamy ostatnią sensowną linię.
                history.append(history[-1])

            if len(history) > 0:
                return history[-1], missing_counter
            return None, missing_counter

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        missing_counter = 0

        if len(np.unique(y_coords)) < 2:
            return None, missing_counter

        poly1 = np.polyfit(y_coords, x_coords, 1)
        poly = np.array([0.0, poly1[0], poly1[1]])

        if len(history) == 0:
            history.append(poly)
        else:
            smoothed_poly = 0.8 * poly + 0.2 * history[-1]
            history.append(smoothed_poly)

        return history[-1], missing_counter

    def calculate_and_log(self, frame_shape, left_poly, right_poly, roi_mask, bad_cross_lines):
        current_time = time.time()
        height, width, _ = frame_shape
        target_offset = self.last_offset
        center_x = width / 2.0
        lookahead_y = int(height * 0.70)

        LANE_WIDTH_PX = 470.0
        mid_poly = None

        if left_poly is not None and right_poly is not None:
            mid_poly = (left_poly + right_poly) / 2.0
        elif left_poly is not None:
            mid_poly = np.copy(left_poly)
            mid_poly[2] += (LANE_WIDTH_PX / 2.0)
        elif right_poly is not None:
            mid_poly = np.copy(right_poly)
            mid_poly[2] -= (LANE_WIDTH_PX / 2.0)

        if mid_poly is not None:
            mid_x_look = mid_poly[0] * (lookahead_y ** 2) + mid_poly[1] * lookahead_y + mid_poly[2]
            target_offset = float(mid_x_look - center_x)

        # ==============================================================
        # Tryb ochronny przed fałszywą linią poprzeczną na zakręcie
        # ==============================================================
        if bad_cross_lines >= 2:
            self.bad_frame_counter += 1
        else:
            self.bad_frame_counter = max(0, self.bad_frame_counter - 1)

        bad_frame = self.bad_frame_counter >= self.bad_frame_limit

        if bad_frame:
            # Nie ufamy nowemu offsetowi. Trzymamy ostatni dobry kierunek.
            # To jest bezpieczniejsze niż nagłe skręcenie w fałszywą linię.
            target_offset = self.last_offset

        crop_h, crop_w = roi_mask.shape
        bumper_h, bumper_w = 70, 180

        left_zone = roi_mask[crop_h - bumper_h: crop_h, 0: bumper_w]
        right_zone = roi_mask[crop_h - bumper_h: crop_h, crop_w - bumper_w: crop_w]

        left_pixels = cv.countNonZero(left_zone)
        right_pixels = cv.countNonZero(right_zone)
        bumper_active_flag = False

        def calculate_dynamic_force(pixels):
            return 100.0 + 200.0 * min(1.0, max(0.0, (pixels - self.pixel_threshold) / 3000.0))

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
            if current_time - self.bumper_start_time < 0.6:
                target_offset = self.bumper_dir
            else:
                self.bumper_state = 'ALIGNING'
                self.bumper_start_time = current_time
        elif self.bumper_state == 'ALIGNING':
            bumper_active_flag = True
            if current_time - self.bumper_start_time < 0.4:
                target_offset = -self.bumper_dir * 0.30
            else:
                self.bumper_state = 'INACTIVE'

        self.last_offset = target_offset

        # ==============================================================
        # TELEMETRIA
        # ==============================================================
        # [0]  - lewa linia wykryta
        # [1]  - lewa A
        # [2]  - lewa B
        # [3]  - lewa C
        # [4]  - prawa linia wykryta
        # [5]  - prawa A
        # [6]  - prawa B
        # [7]  - prawa C
        # [8]  - target_offset
        # [9]  - bumper active
        # [10] - left bumper pixels
        # [11] - right bumper pixels
        # [12] - left dashed flag:  1 = przerywana
        # [13] - right dashed flag: 1 = przerywana
        # [14] - bad frame flag:    1 = obraz podejrzany
        telemetry_array = [0.0] * 15

        if left_poly is not None:
            telemetry_array[0] = 1.0
            telemetry_array[1:4] = left_poly.tolist()

        if right_poly is not None:
            telemetry_array[4] = 1.0
            telemetry_array[5:8] = right_poly.tolist()

        telemetry_array[8] = float(target_offset)
        telemetry_array[9] = 1.0 if bumper_active_flag else 0.0
        telemetry_array[10] = float(left_pixels)
        telemetry_array[11] = float(right_pixels)
        telemetry_array[12] = float(self.left_line_type)
        telemetry_array[13] = float(self.right_line_type)
        telemetry_array[14] = 1.0 if bad_frame else 0.0

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
