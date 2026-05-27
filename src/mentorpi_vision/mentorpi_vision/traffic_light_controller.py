#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Do odbierania koloru światła

class TrafficLightController(Node):
    def __init__(self):
        super().__init__('traffic_light_controller')
        
        # Pamięć robota (stan początkowy)
        self.last_important_light = None  # "czerwone" lub "niebieskie"
        
        # Robot subskrybuje (słucha) temat, w którym algorytm AI podaje wykryty kolor
        self.subscription = self.create_subscription(
            String,
            '/detected_traffic_light',  # Tutaj trafia kolor z kamery
            self.light_callback,
            10
        )
        self.get_logger().info("Kontroler sygnalizacji MentorPi uruchomiony! Czekam na światła...")

    def light_callback(self, msg):
        current_light = msg.data.lower().strip()
        self.get_logger().info(f"Wykryto światło: {current_light.upper()}")

        # LOGIKA SYGNALIZACJI:
        
        # 1. ŚWIATŁO CZERWONE -> STOP
        if current_light == "czerwone":
            self.last_important_light = "czerwone"
            self.execute_action("STOP")

        # 2. ŚWIATŁO NIEBIESKIE -> PRZYSPIESZ
        elif current_light == "niebieskie":
            self.last_important_light = "niebieskie"
            self.execute_action("PRZYSPIESZ")

        # 3. ŚWIATŁO ŻÓŁTE -> Zależy od historii
        elif current_light == "żółte":
            if self.last_important_light == "czerwone":
                self.execute_action("START")
            elif self.last_important_light == "niebieskie":
                self.execute_action("ZWOLNIJ")
            else:
                self.execute_action("ZWOLNIJ")  # Profilaktycznie, jeśli żółte jest pierwsze

    def execute_action(self, action):
        """Funkcja, która w przyszłości wyśle komendy bezpośrednio do kół robota"""
        if action == "STOP":
            self.get_logger().warn("AKCJA: Wykryto CZERWONE. Zatrzymuję silniki!")
            # Tutaj trafi kod: wysyłanie prędkości 0 do robota
        elif action == "START":
            self.get_logger().info("AKCJA: Żółte po czerwonym. Ruszam! (START)")
        elif action == "PRZYSPIESZ":
            self.get_logger().info("AKCJA: NIEBIESKIE. Pełna prędkość do przodu!")
        elif action == "ZWOLNIJ":
            self.get_logger().warn("AKCJA: Żółte po niebieskim. Zwalniam przed zakrętem/stopem.")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()