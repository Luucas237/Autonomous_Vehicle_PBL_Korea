FROM osrf/ros:humble-desktop

# Instalacja pakietów systemowych i ROS 2
RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    python3-opencv \
    python3-pip \
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
    ros-humble-ros-gz \
    ros-humble-xacro \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Instalacja biblioteki PyTorch (wersja CPU/podstawowa do inferencji)
RUN pip3 install torch torchvision torchaudio