import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
  # Set the path to different files and folders.
  pkg_share = FindPackageShare(package='quadrotor_communication').find('quadrotor_communication')
  robot_localization_file_path = '/home/zein/Project/quadrotor-plan-control/src/quadrotor_communication/config/ekf.yaml'


  # Start robot localization using an Extended Kalman filter
  start_robot_localization_cmd = Node(
    package='robot_localization',
    executable='ekf_node',
    name='ekf_filter_node',
    output='screen',
    parameters=[robot_localization_file_path])

  
  # Create the launch description and populate
  ld = LaunchDescription()
  # Declare the launch options
  ld.add_action(start_robot_localization_cmd)
  return ld
