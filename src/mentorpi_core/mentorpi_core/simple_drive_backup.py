#!/usr/bin/env python3
import time
from gpiozero import PhaseEnableMotor, AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device

try:
    Device.pin_factory = LGPIOFactory()
except Exception as e:
    print(f"BŁĄD LGPIO: {e}")
    exit(1)

print("--- Test begin ---")

try:
    print("[1/3] Konfiguracja pinów...")
    # Silnik prawy: DIR (Phase) = GPIO26, PWM (Enable) = GPIO12
    motor_right = PhaseEnableMotor(phase=26, enable=12)
    # Silnik lewy: DIR (Phase) = GPIO24, PWM (Enable) = GPIO13
    motor_left = PhaseEnableMotor(phase=24, enable=13)
    
    # Serwomechanizm na GPIO17 z rozszerzonymi impulsami (0.5ms - 2.5ms)
    steering_servo = AngularServo(
        17, 
        min_angle=-45, 
        max_angle=45, 
        min_pulse_width=0.0005, 
        max_pulse_width=0.0025
    )

    predkosc_testowa = 0.35

    print("[2/3] Test silników DC (Napęd)...")
    
    print(" -> Centrowanie kół...")
    steering_servo.angle = 0.0
    time.sleep(1)

    print(" -> Jazda do PRZODU...")
    motor_left.forward(predkosc_testowa)
    motor_right.forward(predkosc_testowa)
    time.sleep(2.0)

    print(" -> STOP...")
    motor_left.stop()
    motor_right.stop()
    time.sleep(1.0)

    print(" -> Jazda do TYŁU...")
    motor_left.backward(predkosc_testowa)
    motor_right.backward(predkosc_testowa)
    time.sleep(2.0)

    print(" -> STOP...")
    motor_left.stop()
    motor_right.stop()
    time.sleep(1.0)

    print("[3/3] Test Serwomechanizmu (Skręt)...")
    
    print(" -> Skręt: Maksymalnie w LEWO...")
    steering_servo.angle = -45.0
    time.sleep(1.5)

    print(" -> Skręt: Maksymalnie w PRAWO...")
    steering_servo.angle = 45.0
    time.sleep(1.5)

    print(" -> Powrót kół na ŚRODEK...")
    steering_servo.angle = 0.0
    time.sleep(1.0)

    print("--- TEST ZAKOŃCZONY SUKCESEM ---")

except KeyboardInterrupt:
    print("\n[!] Test przerwany przez użytkownika (Ctrl+C)")
finally:
    print("Odcinanie zasilania od silników...")
    try:
        motor_left.stop()
        motor_right.stop()
    except:
        pass