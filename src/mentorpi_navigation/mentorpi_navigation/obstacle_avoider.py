#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

try:
    from heliobox_msgs.msg import Objectinfo, Objects
    HAS_HELIOBOX = True
except ModuleNotFoundError:
    HAS_HELIOBOX = False
    print("CRITICAL: heliobox_msgs not found!")

class LidarSmartAvoider(Node):
    def __init__(self):
        super().__init__('lidar_smart_avoider_node')

        # --- SUBSCRIPTIONS ---
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self.lidar_CB, 10)
        
        # --- PUBLISHERS ---
        if HAS_HELIOBOX:
            self.obj_pub = self.create_publisher(Objects, '/scan_objs', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacle_markers', 10)
        
        self.get_logger().info("Dynamic Lidar Sensor started.")

    def publish_rviz_markers(self, obstacles, frame_id):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for idx, (r, angle) in enumerate(obstacles):
            m = Marker()
            m.header.frame_id = frame_id 
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "obstacles"
            m.id = idx + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            
            m.pose.position.x = float(r * math.cos(angle))
            m.pose.position.y = float(r * math.sin(angle))
            m.pose.position.z = 0.0
            
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color.a = 1.0
            
            if r <= 0.70:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0 # Red (Close)
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0 # Yellow (Warning)
                
            marker_array.markers.append(m)
            
        self.marker_pub.publish(marker_array)

    def lidar_CB(self, msg):
        clusters = []
        current_cluster = []
        
        # 1. Physics-based clustering (array size independent)
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or np.isnan(r) or r < 0.05 or r > 1.5: 
                continue
                
            angle = msg.angle_min + i * msg.angle_increment
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            
            # Front ROI: approx +/- 23 deg
            if not (-0.4 < norm_angle < 0.4): 
                continue

            if not current_cluster: 
                current_cluster.append((r, norm_angle))
            else:
                last_r, last_angle = current_cluster[-1]
                if math.sqrt(r**2 + last_r**2 - 2*r*last_r*math.cos(norm_angle - last_angle)) < 0.20:
                    current_cluster.append((r, norm_angle))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(r, norm_angle)]
                    
        if current_cluster: 
            clusters.append(current_cluster)

        # 2. Extract valid obstacles
        valid_obstacles = []
        for c in clusters:
            if len(c) > 0:
                valid_obstacles.append(min(c, key=lambda x: x[0]))

        # 3. RViz output
        self.publish_rviz_markers(valid_obstacles, msg.header.frame_id)

        # 4. Korean state machine output
        if HAS_HELIOBOX and valid_obstacles:
            groups = Objects()
            for r, angle in valid_obstacles:
                obj = Objectinfo()
                obj.start_dist = float(r)
                obj.end_dist = float(r)
                deg = math.degrees(angle)
                obj.start_idx = deg
                obj.end_idx = deg
                groups.objects_info.append(obj)
            
            groups.no = len(valid_obstacles)
            self.obj_pub.publish(groups)
            
        self.get_logger().info(f"Detected {len(valid_obstacles)} obstacles.", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = LidarSmartAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()