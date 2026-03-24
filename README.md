# Autonomous_Vehicle_PBL_Korea

**Documentation:** [MentorPi Getting Ready](https://docs.hiwonder.com/projects/MentorPi/en/latest/docs/1.getting_ready.html)  
**VNC for Linux:** [RealVNC Viewer Download](https://www.realvnc.com/en/connect/download/viewer/linux/)  

---

## Connection Details

**WiFi Network:**
* SSID: `HW-67`
* Password: `hiwonder`

**SSH Login:**
* Command: `ssh pi@192.168.149.1`
* Password: `raspberrypi`

---

## Project Architecture & Team Workflow

The development is divided into three distinct modules. Manufacturer packages (`slam`, `navigation`, `simulations`) are treated as read-only libraries. All custom ROS 2 nodes must be developed within the specific packages listed below.

### 1. LiDAR & Spatial Awareness
* **Objective:** Raw point cloud processing, obstacle avoidance, and parking logic.
* **Target Package:** `mentorpi_navigation`
* **Dedicated Branches:** `feature/lidar-obstacle`, `feature/lidar-parking`

### 2. Vision & Perception
* **Objective:** Camera image processing, lane tracking, and traffic sign detection (YOLO/OpenCV).
* **Target Package:** `mentorpi_vision`
* **Dedicated Branches:** `feature/lanes-tracking`, `feature/sign-detection`

### 3. Core Control & Sensor Fusion
* **Objective:** Master decision node, data arbitration (multiplexer), PID speed control, and Ackermann steering kinematics.
* **Target Package:** `mentorpi_core`
* **Dedicated Branches:** `feature/core-control`

---

## ===== LIDAR =====
> **📌 Note:** Ta sekcja to **wizualizacja z tutoriala producenta**. Służy głównie do testowania poprawności działania sprzętu i mapowania (SLAM) za pomocą gotowych paczek Hiwonder.

**Rasbian => Terminator #1**
~/.stop_ros.sh 

**For 2D:**
ros2 launch slam slam.launch.py        


**For 3D:**
```bash
ros2 launch slam rtabmap_slam.launch.py
```

**Rasbian => Terminator #2**
```bash
ros2 launch peripherals teleop_key_control.launch.py 
```

**Ubuntu ==> #1**
```bash
xhost +local:root
```

**Ubuntu ==> #2**
```bash
cd ~/PBL_Korea/Autonomous_Vehicle_PBL_Korea

docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/PBL_Korea/Autonomous_Vehicle_PBL_Korea:/workspace -w /workspace pbl_korea_ros2 bash

colcon build

source /opt/ros/humble/setup.bash
source install/setup.bash
source /workspace/.typerc
```

**For 2D:** ```bash
ros2 launch slam rviz_slam.launch.py
```

**For 3D:**
```bash
ros2 launch slam rviz_rtabmap.launch.py
```

---

## ===== LANE DETECTION =====
> **Note:** To jest **prywatny projekt i autorska implementacja**. Uruchamiamy tutaj nasz własny skrypt w Pythonie (OpenCV) do detekcji linii i obliczania odchyłki dla systemu sterowania.

**Rasbian => Terminator #1**
```bash
~/.stop_ros.sh
ros2 launch peripherals depth_camera.launch.py
```

**Ubuntu ==> #1**
```bash
xhost +local:root
```

**Ubuntu ==> #2**
```bash
cd ~/PBL_Korea/Autonomous_Vehicle_PBL_Korea

docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/PBL_Korea/Autonomous_Vehicle_PBL_Korea:/workspace -w /workspace pbl_korea_ros2 bash
```

**Inside Docker (Ubuntu #2):**
```bash
colcon build --packages-select mentorpi_vision --symlink-install

source install/setup.bash

ros2 run mentorpi_vision lane_detector
```
