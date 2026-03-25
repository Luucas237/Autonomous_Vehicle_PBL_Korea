FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    python3-opencv \
    libgflags-dev \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    ros-humble-image-transport \
    ros-humble-image-publisher \
    ros-humble-camera-info-manager \
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-nav2-rviz-plugins \
    && rm -rf /var/lib/apt/lists/*
