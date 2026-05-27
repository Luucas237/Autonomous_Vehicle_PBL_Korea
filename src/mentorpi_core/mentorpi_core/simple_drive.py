#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from gpiozero import PhaseEnableMotor, AngularServo
from enum import Enum, auto

# --- KOREAŃSKA MASZYNA STANÓW ---
class DriveState(Enum):
    LINE_FOLLOW = auto()
    STOP_LINE = auto()
    DYNAMIC_OBS = auto()

class StateMachine:
    def __init__(self):
        self.drive_state = DriveState.LINE_FOLLOW

class SimpleDriveNode(Node):
    def __init__(self):
        super().__init__('cytron_simple_drive_node')

        # --- KONFIGURACJA SPRZĘTU CYTRON ---
        self.get_logger().info("Inicjalizacja pinów Cytron...")
        try:
            self.motor_right = PhaseEnableMotor(phase=26, enable=12)
            self.motor_left = PhaseEnableMotor(phase=24, enable=13)
            self.steering_servo = AngularServo(17, min_angle=-45, max_angle=45, min_pulse_width=0.0005, max_pulse_width=0.0025)
        except Exception as e:
            self.get_logger().error(f'Błąd GPIO: {e}')

        # --- SUBSKRYPCJE Z TWOJEJ ARCHITEKTURY ---
        self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.vision_cb, 10)
        self.create_subscription(Float32, '/vision/offset_raw', self.gui_cb, 10) # Przycisk STOP z Twojego GUI

        # --- ZMIENNE ---
        self.state = StateMachine()
        self.lx = 0.0
        self.rx = 640.0
        self.stopline_detected = 0.0
        
        self.engines_enabled = True
        self.current_angle = 0.0
        self.base_speed = 0.20 # Powolny, bezpieczny przejazd (20% mocy)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Master gotowy: Zmergowano sterowanie Cytron + State Machine')

    def gui_cb(self, msg):
        # Obsługa przycisków z Twojego GUI
        if msg.data == 999.0:
            self.engines_enabled = False
        else:
            self.engines_enabled = True

    def vision_cb(self, msg):
        # Odbiór danych z lande_detector_robot
        if len(msg.data) >= 3:
            self.lx = msg.data[0]
            self.rx = msg.data[1]
            self.stopline_detected = msg.data[2]

    def control_loop(self):
        # 1. Awaryjny STOP z GUI
        if not self.engines_enabled:
            self.motor_left.stop()
            self.motor_right.stop()
            self.steering_servo.angle = 0.0
            return

        # 2. Przełączanie Stanów (State Machine)
        if self.stopline_detected == 1.0:
            self.state.drive_state = DriveState.STOP_LINE
        else:
            self.state.drive_state = DriveState.LINE_FOLLOW

        # 3. Wykonywanie akcji zależnie od stanu
        if self.state.drive_state == DriveState.STOP_LINE:
            self.motor_left.stop()
            self.motor_right.stop()
            self.get_logger().info("Stan: STOP_LINE - Czekam...")
            return
            
        elif self.state.drive_state == DriveState.LINE_FOLLOW:
            # Koreańska matematyka środka:
            lane_center = (self.lx + self.rx) / 2.0
            
            # Odchyłka (środek dla 640px to 320)
            error_px = lane_center - 320.0
            
            # Czułość skrętu mapowana na serwo (-45 do 45)
            target_angle = (error_px / 320.0) * 45.0
            target_angle = max(min(target_angle, 45.0), -45.0)

            # Filtr płynności z Twojego starego simple_drive
            self.current_angle = self.current_angle + 0.4 * (target_angle - self.current_angle)

            # Wysłanie sprzętowe
            self.steering_servo.angle = self.current_angle
            self.motor_left.forward(self.base_speed)
            self.motor_right.forward(self.base_speed)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Twardy STOP. Odcięcie silników.')
        node.motor_left.stop()
        node.motor_right.stop()
        node.steering_servo.angle = 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()