# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32

# import cv2 as cv
# import numpy as np
# from collections import deque
# import time

# class ProcessFrame(Node):
#     def __init__(self):
#         super().__init__('lane_detector_node')
        
#         self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        
#         self.left_history = deque(maxlen=7)
#         self.right_history = deque(maxlen=7)
        
#         self.missing_left = 0
#         self.missing_right = 0
#         self.last_offset = 0.0

#         self.cap = cv.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.get_logger().error("Nie można otworzyć kamery!")

#         self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)
#         self.timer = self.create_timer(0.033, self.timer_callback)

#         self.last_time = time.time()
#         self.fps = 0.0
#         self.get_logger().info('Wizja gotowa')

#     def timer_callback(self):
#         ret, frame = self.cap.read()
#         if not ret: return

#         current_time = time.time()
#         self.fps = 1.0 / (current_time - self.last_time + 0.0001)
#         self.last_time = current_time

#         self.perform_detection(frame)
#         cv.waitKey(1)

#     def perform_detection(self, frame):
#         left_lines, right_lines, roi_debug = self.detect_white_lines(frame)

#         height, width, _ = frame.shape
#         y1 = int(height * 0.4)
#         y2 = height
#         ploty = np.linspace(y1, y2, num=20)

#         left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
#         right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)

#         left_fitx = self.get_fitx(left_poly, ploty)
#         right_fitx = self.get_fitx(right_poly, ploty)

#         output = frame.copy()
#         output, status = self.draw_guideline(output, ploty, left_fitx, right_fitx, width)

#         cv.imshow("Region of Interest", roi_debug)
#         cv.imshow("Podglad", output)

#     def detect_white_lines(self, frame):
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         lower_white = np.array([0, 0, 180], dtype="uint8")
#         upper_white = np.array([180, 30, 255], dtype="uint8")
#         mask = cv.inRange(hsv, lower_white, upper_white)

#         blur = cv.GaussianBlur(mask, (5, 5), 0)
#         edges = cv.Canny(blur, 50, 150)

#         height, width = edges.shape
#         roi_mask = np.zeros_like(edges)
#         top_width = int(width * 0.4)
#         bottom_width = width
#         top_y = int(height * 0.4)
#         bottom_y = height

#         vertices = np.array([[ 
#             ((width - top_width) // 2, top_y),
#             ((width + top_width) // 2, top_y),
#             (bottom_width, bottom_y),
#             (0, bottom_y)
#         ]], dtype=np.int32)

#         cv.fillPoly(roi_mask, vertices, 255)
#         roi_edges = cv.bitwise_and(edges, roi_mask)

#         roi_debug = cv.cvtColor(roi_edges, cv.COLOR_GRAY2BGR)
#         cv.polylines(roi_debug, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

#         lines = cv.HoughLinesP(roi_edges, 1, np.pi/180, 15, minLineLength=7, maxLineGap=3)

#         left_lines = []
#         right_lines = []

#         if lines is not None:
#             for line in lines:
#                 for x1, y1, x2, y2 in line:
#                     slope = (y2 - y1) / (x2 - x1 + 0.0001)
#                     if abs(slope) < 0.15: continue
#                     if (x1 + x2) / 2 < width / 2: 
#                         left_lines.append((x1, y1, x2, y2))
#                     else: 
#                         right_lines.append((x1, y1, x2, y2))

#         return left_lines, right_lines, roi_debug

#     def fit_and_filter(self, lines, history, missing_counter):
#         if len(lines) == 0:
#             missing_counter += 1
#             if missing_counter > 5:
#                 history.clear()
#             elif len(history) > 0:
#                 history.append(history[-1]) 
#             return None, missing_counter

#         x_coords, y_coords = [], []
#         for x1, y1, x2, y2 in lines:
#             x_coords.extend([x1, x2])
#             y_coords.extend([y1, y2])

#         missing_counter = 0

#         if len(np.unique(y_coords)) < 3:
#             return None, missing_counter

#         poly = np.polyfit(y_coords, x_coords, 2)
#         poly[0] = np.clip(poly[0], -0.002, 0.002)

#         history.append(poly)

#         if len(history) == 7:
#             smoothed_poly = np.zeros(3)
#             for i in range(7):
#                 smoothed_poly += self.fir_weights[i] * history[i]
#             return smoothed_poly, missing_counter
#         else:
#             return np.mean(history, axis=0), missing_counter

#     def get_fitx(self, poly, ploty):
#         if poly is None: return None
#         return poly[0]*ploty**2 + poly[1]*ploty + poly[2]

#     def draw_guideline(self, frame, ploty, left_fitx, right_fitx, width):
#         center_status = "BRAK LINII"
#         offset = self.last_offset

#         if left_fitx is not None and right_fitx is not None:
#             left_pts = np.int32(np.column_stack((left_fitx, ploty))).reshape((-1, 1, 2))
#             right_pts = np.int32(np.column_stack((right_fitx, ploty))).reshape((-1, 1, 2))

#             cv.polylines(frame, [left_pts], isClosed=False, color=(255, 0, 0), thickness=4)
#             cv.polylines(frame, [right_pts], isClosed=False, color=(255, 0, 0), thickness=4)

#             mid_fitx = (left_fitx + right_fitx) / 2
#             mid_pts = np.int32(np.column_stack((mid_fitx, ploty))).reshape((-1, 1, 2))
#             cv.polylines(frame, [mid_pts], isClosed=False, color=(0, 0, 255), thickness=3)

#             lookahead_idx = int(len(ploty) * 0.4) 
#             mid_x = mid_fitx[lookahead_idx]
#             cv.drawMarker(frame, (int(mid_x), int(ploty[lookahead_idx])), (0, 255, 255), cv.MARKER_CROSS, 20, 2)

#             offset = mid_x - (width // 2)
#             self.last_offset = offset

#         if abs(offset) < width * 0.05: center_status = "SRODEK"
#         elif offset < 0: center_status = "SKREC W LEWO"
#         else: center_status = "SKREC W PRAWO"

#         msg = Float32()
#         msg.data = float(offset)
#         self.offset_value_publisher_.publish(msg)

#         cv.putText(frame, f"FPS: {self.fps:.1f} | Kierunek: {center_status}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         return frame, center_status

#     def destroy_node(self):
#         self.cap.release()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = ProcessFrame()
#     try: rclpy.spin(node)
#     except KeyboardInterrupt: pass
#     node.destroy_node()
#     rclpy.shutdown()
#     cv.destroyAllWindows()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import cv2 as cv
import numpy as np
from collections import deque
import time

class ProcessFrame(Node):
    def __init__(self):
        super().__init__('lane_detector_node')
        
        self.fir_weights = np.array([0.075, 0.125, 0.175, 0.250, 0.175, 0.125, 0.075])
        
        self.left_history = deque(maxlen=7)
        self.right_history = deque(maxlen=7)
        
        self.missing_left = 0
        self.missing_right = 0
        self.last_offset = 0.0

        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Nie można otworzyć kamery!")

        # -----------------------------------------------------------------
        # PUBLISHER: Wysyła wartość offsetu (Float32) na topic 'offset_value'
        # W simple_drive.py musisz zasubskrybować ten sam topic: 'offset_value'
        # -----------------------------------------------------------------
        self.offset_value_publisher_ = self.create_publisher(Float32, 'offset_value', 10)
        
        self.timer = self.create_timer(0.033, self.timer_callback)

        self.last_time = time.time()
        self.fps = 0.0
        self.get_logger().info('Wizja gotowa! Wykrywam czarne linie (proste) + publikuje Offset.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return

        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time + 0.0001)
        self.last_time = current_time

        self.perform_detection(frame)
        cv.waitKey(1)

    def perform_detection(self, frame):
        left_lines, right_lines, roi_debug = self.detect_black_lines(frame)

        height, width, _ = frame.shape
        y1 = int(height * 0.4)
        y2 = height
        ploty = np.linspace(y1, y2, num=20)

        left_poly, self.missing_left = self.fit_and_filter(left_lines, self.left_history, self.missing_left)
        right_poly, self.missing_right = self.fit_and_filter(right_lines, self.right_history, self.missing_right)

        left_fitx = self.get_fitx(left_poly, ploty)
        right_fitx = self.get_fitx(right_poly, ploty)

        output = frame.copy()
        output, status = self.draw_guideline(output, ploty, left_fitx, right_fitx, width)

        cv.imshow("Region of Interest", roi_debug)
        cv.imshow("Podglad", output)

    def detect_black_lines(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        lower_black = np.array([0, 0, 0], dtype="uint8")
        upper_black = np.array([180, 255, 80], dtype="uint8") 
        
        mask = cv.inRange(hsv, lower_black, upper_black)

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

        roi_debug = cv.cvtColor(roi_edges, cv.COLOR_GRAY2BGR)
        cv.polylines(roi_debug, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

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

        return left_lines, right_lines, roi_debug

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

        if len(np.unique(y_coords)) < 2:
            return None, missing_counter

        poly = np.polyfit(y_coords, x_coords, 1)

        history.append(poly)

        if len(history) == 7:
            smoothed_poly = np.zeros(2)
            for i in range(7):
                smoothed_poly += self.fir_weights[i] * history[i]
            return smoothed_poly, missing_counter
        else:
            return np.mean(history, axis=0), missing_counter

    def get_fitx(self, poly, ploty):
        if poly is None: return None
        return poly[0] * ploty + poly[1]

    def draw_guideline(self, frame, ploty, left_fitx, right_fitx, width):
        center_status = "BRAK LINII"
        offset = self.last_offset

        if left_fitx is not None and right_fitx is not None:
            left_pts = np.int32(np.column_stack((left_fitx, ploty))).reshape((-1, 1, 2))
            right_pts = np.int32(np.column_stack((right_fitx, ploty))).reshape((-1, 1, 2))

            cv.polylines(frame, [left_pts], isClosed=False, color=(255, 0, 0), thickness=4)
            cv.polylines(frame, [right_pts], isClosed=False, color=(255, 0, 0), thickness=4)

            # --- SRODEK TRASY I WYZNACZANIE OFFSETU ---
            mid_fitx = (left_fitx + right_fitx) / 2
            mid_pts = np.int32(np.column_stack((mid_fitx, ploty))).reshape((-1, 1, 2))
            
            # Rysowanie środkowej, czerwonej linii
            cv.polylines(frame, [mid_pts], isClosed=False, color=(0, 0, 255), thickness=3)

            # Punkt "lookahead" (miejsce, na które patrzymy w przód, by wyznaczyć skręt)
            lookahead_idx = int(len(ploty) * 0.4) 
            mid_x = mid_fitx[lookahead_idx]
            
            # Rysowanie żółtego krzyżyka na celu
            cv.drawMarker(frame, (int(mid_x), int(ploty[lookahead_idx])), (0, 255, 255), cv.MARKER_CROSS, 20, 2)

            # OBLICZANIE OFFSETU: Cel_X minus Srodek_Ekranu
            offset = mid_x - (width / 2.0)
            self.last_offset = offset

        if abs(offset) < width * 0.05: center_status = "SRODEK"
        elif offset < 0: center_status = "SKREC W LEWO"
        else: center_status = "SKREC W PRAWO"

        # -----------------------------------------------------------------
        # WYSYŁANIE OFFSETU PRZEZ ROS2 DO SIMPLE_DRIVE.PY
        # -----------------------------------------------------------------
        msg = Float32()
        msg.data = float(offset)
        self.offset_value_publisher_.publish(msg)

        # -----------------------------------------------------------------
        # WYPISYWANIE OFFSETU W KONSOLI (TERMINALU)
        # -----------------------------------------------------------------
        self.get_logger().info(f"[FPS: {self.fps:4.1f}] Offset: {offset:6.1f} px | Status: {center_status}")

        # Rysowanie napisów na obrazku
        cv.putText(frame, f"Offset: {offset:.1f} px | {center_status}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame, center_status

    def destroy_node(self):
        self.cap.release()
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