cmake_minimum_required(VERSION 3.5)
project(deeplabv3p_hobot_dnn)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED /opt/tros/humble/)
find_package(ai_msgs REQUIRED /opt/tros/humble/)
find_package(hbm_img_msgs REQUIRED /opt/tros/humble/)
find_package(dnn_node REQUIRED /opt/tros/humble/)
find_package(hobot_cv REQUIRED /opt/tros/humble/)

# x3|rdkultra|x86
set(PREFIX_PATH x3)
set(SYS_ROOT ${CMAKE_SYSROOT})

message("build platform X5")
add_definitions(-DPLATFORM_X5)
set(PREFIX_PATH x5)

message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

# 设置一个默认hobotcv库的安装路径
set(HOBOTCV_LIB_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib)
# 如果find_package成功，使用查找到的路径更新HOBOTCV_LIB_INSTALL_PATH
if(hobot_cv_LIBRARIES)
  message("hobot_cv_LIBRARIES is ${hobot_cv_LIBRARIES}")
  # 查找库路径的最后一个/符号
  string(FIND ${hobot_cv_LIBRARIES} "/" index REVERSE)
  # 获取库路径
  string(SUBSTRING ${hobot_cv_LIBRARIES} 0 ${index} HOBOTCV_LIB_INSTALL_PATH)
  message("update HOBOTCV_LIB_INSTALL_PATH as " ${HOBOTCV_LIB_INSTALL_PATH})
endif()

message("HOBOTCV_LIB_INSTALL_PATH is " ${HOBOTCV_LIB_INSTALL_PATH})

link_directories(
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/hbmedia/
  ${HOBOTCV_LIB_INSTALL_PATH}
)

include_directories(
  include
  ${PROJECT_SOURCE_DIR}
  ${SYS_ROOT}/usr/include
)
include_directories(/usr/include/opencv4)
include_directories(/userdata/deps)
include_directories(/usr/lib)
include_directories(/usr/lib/aarch64-linux-gnu)
include_directories(/userdata/sysroot_docker/usr_x5/lib)
include_directories(/userdata/sysroot_docker/usr_x5/lib/aarch64-linux-gnu)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/hbmedia/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/userdata/deps
  ${SYS_ROOT}/userdata/sysroot_docker/usr_x5/lib
  ${SYS_ROOT}/userdata/sysroot_docker/usr_x5/lib/aarch64-linux-gnu
  ${SYS_ROOT}/userdata/sysroot_docker/usr_x5/lib/hbbpu/
)

add_executable(${PROJECT_NAME}
  src/sample.cpp
  src/parser.cpp
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  dnn_node
  sensor_msgs
  ai_msgs
  hbm_img_msgs
  hobot_cv
)

# Install executables
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/config/
  DESTINATION lib/${PROJECT_NAME}/config/
)

ament_package()