cmake_minimum_required(VERSION 3.8)
project(mentorpi_sim)

find_package(ament_cmake REQUIRED)

# To magiczne polecenie skopiuje wszystko dokładnie tak, jak ułożyłeś to w VS Code
install(DIRECTORY
  launch
  meshes
  models
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()