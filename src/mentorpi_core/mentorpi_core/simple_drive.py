#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from gpiozero import PhaseEnableMotor, AngularServo

class DirectCytronDriver(Node):
    def __init__(self):
        super().__init__('cytron_smart_avoider_node')
        self.get_logger().info('Inicjalizacja pinów GPIO...')
        try:
            # Silniki DC
            self.motor_right = PhaseEnableMotor(phase=26, enable=12)
            self.motor_left = PhaseEnableMotor(phase=24, enable=13)
            
            # Serwo kierownicy
            self.steering_servo = AngularServo(
                17, 
                min_angle=-45, 
                max_angle=45, 
                min_pulse_width=0.0005, 
                max_pulse_width=0.0025
            )
        except Exception as e:
            self.get_logger().error(f'Błąd sprzętowy GPIO: {e}')
            raise e

  
        self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.vision_cb, 10)
        self.create_subscription(Float32, '/vision/offset_raw', self.gui_cb, 10)

        # --- ZMIENNE STERUJĄCE ---
        self.lx = 160.0
        self.rx = 480.0
        self.engine_enabled = True
        self.current_angle = 0.0
        
        # Konfiguracja napędu
        self.base_speed = 0.2
        self.steering_sensitivity = 1.0 # Współczynnik czułości skrętu

        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz
        self.get_logger().info('Master gotowy: Pełna sprzętowa kontrola napędu!')

    def vision_cb(self, msg):
        if len(msg.data) >= 2:
            self.lx = msg.data[0]
            self.rx = msg.data[1]

    def gui_cb(self, msg):
 
        if msg.data == 999.0:
            self.engine_enabled = False
        else:
            self.engine_enabled = True

    def control_loop(self):

        if not self.engine_enabled:
            self.motor_left.stop()
            self.motor_right.stop()
            self.steering_servo.angle = 0.0
            return

        # 2. Obliczanie środka pasa
        lane_center = (self.lx + self.rx) / 2.0
        
        # 3. Odchyłka w pikselach (0 to idealny środek dla kamery 640px)
        error_px = lane_center - 320.0 
        

        target_angle = (error_px / 320.0) * 45.0 * self.steering_sensitivity
        
        # target_angle = -target_angle
        
        # Ograniczenie mechaniczne serwa (żeby nie spalić przekładni)
        target_angle = max(min(target_angle, 45.0), -45.0)

        # 5. Filtr dolnoprzepustowy
        self.current_angle = self.current_angle + 0.6 * (target_angle - self.current_angle)

        # 6. Wysłanie sygnałów na piny sprzętowe
        self.steering_servo.angle = self.current_angle
        self.motor_left.forward(self.base_speed)
        self.motor_right.forward(self.base_speed)

def main(args=None):
    rclpy.init(args=args)
    node = DirectCytronDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Otrzymano sygnał STOP. Wyłączanie silników...')
        node.motor_left.stop()
        node.motor_right.stop()
        node.steering_servo.angle = 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()