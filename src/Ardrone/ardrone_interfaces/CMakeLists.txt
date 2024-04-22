cmake_minimum_required(VERSION 3.8)
project(ardrone_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rosidl_default_generators REQUIRED)

#### ROS2 Interface Directives ####
set(interface_files
  # Messages
  "msg/NAVDataDemo.msg"  
  "msg/NAVDataEulerAngles.msg"  
  "msg/NAVDataMagneto.msg"  
  "msg/NAVDataPressureRaw.msg"  
  "msg/NAVDataPwm.msg"  
  "msg/NAVDataRawMeasures.msg"  
  "msg/NAVDataTime.msg" 
  # Services
  "srv/Command.srv"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${interface_files}
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

ament_export_dependencies(
  rosidl_default_runtime
)
#### ROS2 Interface Directives [END] ####

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
