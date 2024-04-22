import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('ardrone_control'),
        'config',
        'ardrone_teleop_params.yaml'
        )
        
    # input_teleop_node = Node(
    #     package = "ardrone_control",
    #     executable = 'input_teleop_node',        
    #     name = "input_teleop_node",
    #     parameters = [config],
    #     prefix='xterm -e'
    # )
    # ld.add_action(input_teleop_node)
    ardrone_teleop_control_node = Node(
        package = "ardrone_control",
        executable = 'ardrone_teleop_control_node',        
        name = "ardrone_teleop_control_node",
        parameters = [config],
        prefix='xterm -e'
    )
    ld.add_action(ardrone_teleop_control_node)

    return ld