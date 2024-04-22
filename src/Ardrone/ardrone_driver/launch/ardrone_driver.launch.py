import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('ardrone_driver'),
        'config',
        'ardrone_driver_params.yaml'
        )
        
    ardrone_driver_node = Node(
        package = "ardrone_driver",
        executable = 'ardrone_driver_node',        
        name = "ardrone_driver_node",
        parameters = [config]
    )
    ld.add_action(ardrone_driver_node)

    return ld