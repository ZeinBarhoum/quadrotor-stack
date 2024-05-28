from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression

from ament_index_python.packages import get_package_share_directory
import yaml
import os


def generate_launch_description():
    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'ardrone.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            parameters = dict()
    dfbc_controller_parameters = parameters['DFBCController']

    dfbc_controller_node = Node(
        package='quadrotor_control',
        executable='quadrotor_dfbc',
        output='screen',
        parameters=[dfbc_controller_parameters]
    )
    return LaunchDescription([dfbc_controller_node])
