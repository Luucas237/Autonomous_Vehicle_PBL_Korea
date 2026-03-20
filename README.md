# Autonomous_Vehicle_PBL_Korea
**Documentation**
https://docs.hiwonder.com/projects/MentorPi/en/latest/docs/1.getting_ready.html

**VNC for Linux**
https://www.realvnc.com/en/connect/download/viewer/linux/

# Login
ssh pi@192.168.149.1

raspberrypi

# WiFi
HW-67

hiwonder

# ===== LIDAR =====
**Rasbian => Terminator #1**

~/.stop_ros.sh 

**For 2D:**

ros2 launch slam slam.launch.py  

**For 3D:**

ros2 launch slam rtabmap_slam.launch.py


**Rasbian => Terminator #2**

ros2 launch peripherals teleop_key_control.launch.py 


**Ubuntu ==> #1**

xhost +local:root


**Ubuntu ==> #2**

cd Autonomous_Vehicle_PBL_Korea

docker run -it --rm --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/PBL_Korea/Autonomous_Vehicle_PBL_Korea:/workspace -w /workspace osrf/ros:humble-desktop bash

colcon build

source /opt/ros/humble/setup.bash

source install/setup.bash

source /workspace/.typerc


**For 2D:**

ros2 launch slam rviz_slam.launch.py


**For 3D:**

ros2 launch slam rviz_rtabmap.launch.py
