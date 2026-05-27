#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Joy
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
        
        # --- SUBSKRYPCJA PADA ---
        self.create_subscription(Joy, '/joy', self.joy_cb, 10)

        # --- ZMIENNE AUTO (WIZJA) ---
        self.state = StateMachine()
        self.lx = 0.0
        self.rx = 640.0
        self.stopline_detected = 0.0
        self.engines_enabled = True
        self.current_angle = 0.0
        self.base_speed = 0.20 # Prędkość Auto (20%)

        # --- ZMIENNE MANUAL (PAD) ---
        self.mode = 'MANUAL' # Domyślnie robot stoi po włączeniu
        self.last_btn_x = 0
        self.manual_speed = 0.0
        self.manual_steer_target = 0.0
        self.r2_initialized = False

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Master gotowy: Zmergowano MANUAL (Joy) / AUTO (Wizja)')

    def gui_cb(self, msg):
        if msg.data == 999.0:
            self.engines_enabled = False
            self.get_logger().warn("AWARYJNY STOP Z GUI!")
        else:
            self.engines_enabled = True

    def vision_cb(self, msg):
        if len(msg.data) >= 3:
            self.lx = msg.data[0]
            self.rx = msg.data[1]
            self.stopline_detected = msg.data[2]

    def joy_cb(self, msg):
        # 1. PRZEŁĄCZANIE TRYBÓW (Przycisk X)
        # UWAGA: Na padzie Xbox/PC 'X' to zazwyczaj index 2. Na PS4/PS5 'Krzyżyk' to index 0, a 'Kwadrat' to 3.
        # Domyślnie ustawiam index 2 (Standard ROS dla klonów Xboxa)
        btn_x = msg.buttons[2] if len(msg.buttons) > 2 else 0
        
        if btn_x == 1 and self.last_btn_x == 0:
            if self.mode == 'MANUAL':
                self.mode = 'AUTO'
                self.get_logger().info(">>> TRYB ZMIENIONY: AUTO (Maszyna Stanów) <<<")
            else:
                self.mode = 'MANUAL'
                self.get_logger().info(">>> TRYB ZMIENIONY: MANUAL (Pad) <<<")
        self.last_btn_x = btn_x

        # 2. KIEROWNICA - Lewa gałka poziomo (Oś 0)
        # Oś 0 daje od 1.0 (max w lewo) do -1.0 (max w prawo)
        if len(msg.axes) > 0:
            self.manual_steer_target = -msg.axes[0] * 45.0 # Skalowanie na kąty serwa (-45 do 45)

        # 3. GAZ - Spust R2 (Zazwyczaj Oś 5)
        # ROS Joy mapuje spusty od 1.0 (puszczone) do -1.0 (wciśnięte)
        if len(msg.axes) > 5:
            r2_val = msg.axes[5]
            
            # Bezpiecznik sprzętowy dla niezainicjowanych spustów w Linuxie
            if r2_val != 0.0:
                self.r2_initialized = True
                
            if self.r2_initialized:
                gas_percent = (1.0 - r2_val) / 2.0 # Przekształcenie do zakresu 0.0 - 1.0
                self.manual_speed = gas_percent * 0.20 # Limit sprzętowy R2 do max 20% mocy!
            else:
                self.manual_speed = 0.0

    def control_loop(self):
        # 1. Nadrzędny wyłącznik z GUI (działa w obu trybach)
        if not self.engines_enabled:
            self.motor_left.stop()
            self.motor_right.stop()
            self.steering_servo.angle = 0.0
            return

        # ==========================================
        # LOGIKA TRYBU: MANUAL (PAD)
        # ==========================================
        if self.mode == 'MANUAL':
            # Płynność ruchu serwa z pada
            self.current_angle = self.current_angle + 0.4 * (self.manual_steer_target - self.current_angle)
            self.steering_servo.angle = self.current_angle

            # Odpalanie silników tylko gdy R2 wciśnięte
            if self.manual_speed > 0.01:
                self.motor_left.forward(self.manual_speed)
                self.motor_right.forward(self.manual_speed)
            else:
                self.motor_left.stop()
                self.motor_right.stop()
            return

        # ==========================================
        # LOGIKA TRYBU: AUTO (WIZJA + STATE MACHINE)
        # ==========================================
        if self.stopline_detected == 1.0:
            self.state.drive_state = DriveState.STOP_LINE
        else:
            self.state.drive_state = DriveState.LINE_FOLLOW

        # Wykonywanie akcji ze State Machine
        if self.state.drive_state == DriveState.STOP_LINE:
            self.motor_left.stop()
            self.motor_right.stop()
            self.get_logger().info("Auto: STOP LINE - zatrzymano na linii.")
            return
            
        elif self.state.drive_state == DriveState.LINE_FOLLOW:
            lane_center = (self.lx + self.rx) / 2.0
            error_px = lane_center - 320.0
            
            target_angle = (error_px / 320.0) * 45.0
            target_angle = max(min(target_angle, 45.0), -45.0)

            self.current_angle = self.current_angle + 0.4 * (target_angle - self.current_angle)

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