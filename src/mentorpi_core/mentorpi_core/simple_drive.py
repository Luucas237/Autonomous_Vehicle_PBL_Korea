#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import LaserScan
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
import math
import numpy as np
import time

# --- STATE MACHINE ---
class DriveState:
    LINE_FOLLOW = 1
    SWERVE = 2
    PASSING = 3
    RETURN = 4
    REVERSE = 5

class StopState:
    NONE = 1
    STOP_LINE_WAIT = 2
    IGNORE_TEMP = 3

class AutoBrain(Node):
    def __init__(self):
        super().__init__('hybrid_control_node')

        # --- SUBSCRIPTIONS ---
        # Pad temporarily disabled
        # self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.create_subscription(Float32MultiArray, '/vision/lane_telemetry', self.vision_callback, 10)
        self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.create_subscription(Float32, '/vision/offset_raw', self.gui_callback, 10)

        # --- PUBLISHERS ---
        self.servo_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 3)

        # --- SYSTEM STATE ---
        self.drive_state = DriveState.LINE_FOLLOW
        self.stop_state = StopState.NONE
        self.gui_engine_enabled = True
        
        self.timer_start = 0.0
        self.state_duration = 0.0
        self.state_start_time = time.time()
        
        # --- VISION DATA ---
        self.lx, self.rx = 210.0, 540.0
        self.stop_line_flag, self.ped_flag = 0.0, 0.0
        self.cam_left_px, self.cam_right_px = 0.0, 0.0
        self.solid_line_px = 1600.0 # Threshold for solid line
        
        # --- LIDAR DATA (360 degrees) ---
        self.front_dist, self.left_dist, self.right_dist = 999.0, 999.0, 999.0
        self.decide_l, self.decide_r = 999.0, 999.0

        # --- AVOIDANCE PARAMS ---
        self.track_side = None
        self.swerve_sign = 0.0
        self.current_filtered_steer = 0.0

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('AutoBrain started in AUTO mode. Pad disabled. Solid line detection Active.')

    def gui_callback(self, msg):
        self.gui_engine_enabled = (msg.data != 999.0)

    def vision_callback(self, msg):
        # Format: [lx, rx, stop_flag, ped_flag, left_px, right_px]
        if len(msg.data) >= 4:
            self.lx, self.rx = msg.data[0], msg.data[1]
            self.stop_line_flag, self.ped_flag = msg.data[2], msg.data[3]
        if len(msg.data) >= 6:
            self.cam_left_px, self.cam_right_px = msg.data[4], msg.data[5]

    def scan_callback(self, msg):
        raw_f, raw_l, raw_r, decide_l, decide_r = 999.0, 999.0, 999.0, 999.0, 999.0
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or np.isnan(r) or r < 0.05 or r > 1.5: continue
            angle = msg.angle_min + i * msg.angle_increment
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            
            if -0.35 < norm_angle < 0.35: raw_f = min(raw_f, r)
            if 1.0 < norm_angle < 2.2: raw_l = min(raw_l, r)
            if -2.2 < norm_angle < -1.0: raw_r = min(raw_r, r)
            if 0.1 < norm_angle < 0.8: decide_l = min(decide_l, r)
            if -0.8 < norm_angle < -0.1: decide_r = min(decide_r, r)

        self.front_dist, self.left_dist, self.right_dist = raw_f, raw_l, raw_r
        self.decide_l, self.decide_r = decide_l, decide_r

    def steer_to_pwm(self, steering):
        return 1500 + int(steering * 322)

    def publish_cmd(self, speed, steer):
        steer = max(min(steer, 1.0), -1.0)
        
        servo_msg = PWMServoState()
        servo_msg.id = [3]
        servo_msg.position = [self.steer_to_pwm(steer)]
        pwm_data = SetPWMServoState()
        pwm_data.state = [servo_msg]
        pwm_data.duration = 0.02
        self.servo_pub.publish(pwm_data)

        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = 0.0 
        self.cmd_vel_pub.publish(twist)

    def set_timer(self, duration):
        self.timer_start = time.time()
        self.state_duration = duration

    def timer_done(self):
        return (time.time() - self.timer_start) >= self.state_duration

    def change_drive_state(self, new_state):
        self.drive_state = new_state
        self.state_start_time = time.time()

    # --- MAIN LOOP (20Hz) ---
    def timer_callback(self):
        if not self.gui_engine_enabled:
            self.publish_cmd(0.0, 0.0)
            return

        target_speed = 0.15 # Constant 15% auto speed
        target_steer = 0.0

        # 1. PEDESTRIAN
        if self.ped_flag == 1.0:
            self.get_logger().info("PEDESTRIAN! Braking.", throttle_duration_sec=1.0)
            self.publish_cmd(0.0, 0.0)
            return

        # 2. STOP LINE
        if self.stop_state == StopState.NONE and self.stop_line_flag == 1.0:
            self.stop_state = StopState.STOP_LINE_WAIT
            self.set_timer(3.0) 
            
        if self.stop_state == StopState.STOP_LINE_WAIT:
            if not self.timer_done():
                self.publish_cmd(0.0, 0.0)
                return
            self.stop_state = StopState.IGNORE_TEMP
            self.set_timer(2.0)
        elif self.stop_state == StopState.IGNORE_TEMP:
            if self.timer_done():
                self.stop_state = StopState.NONE

        # 3. ADVANCED AVOIDANCE STATE MACHINE
        if self.drive_state in [DriveState.LINE_FOLLOW, DriveState.SWERVE, DriveState.PASSING]:
            if self.front_dist < 0.16:
                self.change_drive_state(DriveState.REVERSE)

        if self.drive_state == DriveState.LINE_FOLLOW:
            target_steer = (self.lx - 210.0) / 220.0

            if self.front_dist < 0.45:
                # Solid line checks (Cannot cross if > 1600px)
                forbidden_l = (self.cam_left_px > self.solid_line_px)
                forbidden_r = (self.cam_right_px > self.solid_line_px)

                can_go_l = (self.decide_l > 0.4) and not forbidden_l
                can_go_r = (self.decide_r > 0.4) and not forbidden_r

                if can_go_l and not can_go_r:
                    self.track_side = 'LEFT'
                    self.swerve_sign = 1.0
                elif can_go_r and not can_go_l:
                    self.track_side = 'RIGHT'
                    self.swerve_sign = -1.0
                elif can_go_l and can_go_r:
                    if self.decide_l > self.decide_r:
                        self.track_side, self.swerve_sign = 'LEFT', 1.0
                    else:
                        self.track_side, self.swerve_sign = 'RIGHT', -1.0
                else:
                    self.get_logger().error("BLOCKED! Reversing.")
                    self.change_drive_state(DriveState.REVERSE)
                    return
                
                self.change_drive_state(DriveState.SWERVE)

        elif self.drive_state == DriveState.SWERVE:
            if self.front_dist > 0.45:
                if self.track_side == 'RIGHT' and self.right_dist < 0.8: self.change_drive_state(DriveState.PASSING)
                elif self.track_side == 'LEFT' and self.left_dist < 0.8: self.change_drive_state(DriveState.PASSING)
                    
            if time.time() - self.state_start_time > 2.5:
                self.change_drive_state(DriveState.RETURN)
            
            target_steer = self.swerve_sign * 0.8
                
        elif self.drive_state == DriveState.PASSING:
            clear_behind = (self.right_dist > 0.4) if self.track_side == 'RIGHT' else (self.left_dist > 0.4)
            is_passed = (self.right_dist > 0.9) if self.track_side == 'RIGHT' else (self.left_dist > 0.9)
            
            if is_passed and clear_behind:
                self.change_drive_state(DriveState.RETURN)

            if self.track_side == 'RIGHT': 
                target_steer = (self.right_dist - 0.35) * 1.5
            else: 
                target_steer = (0.35 - self.left_dist) * 1.5
                
        elif self.drive_state == DriveState.RETURN:
            if time.time() - self.state_start_time > 0.5: 
                self.change_drive_state(DriveState.LINE_FOLLOW)
            target_steer = -self.swerve_sign * 0.4
                
        elif self.drive_state == DriveState.REVERSE:
            if time.time() - self.state_start_time > 1.5: 
                self.change_drive_state(DriveState.LINE_FOLLOW)
            target_speed = -0.15
            target_steer = 0.0

        # Smooth steering (Fast for Line Follow, Smooth for Lidar swerve)
        alpha = 0.7 if self.drive_state == DriveState.LINE_FOLLOW else 0.15
        self.current_filtered_steer = (alpha * target_steer) + ((1.0 - alpha) * self.current_filtered_steer)
        
        self.publish_cmd(target_speed, self.current_filtered_steer)

def main(args=None):
    rclpy.init(args=args)
    node = AutoBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_cmd(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()