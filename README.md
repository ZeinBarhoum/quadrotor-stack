# Quadrotor Trajectory Planning and Control

## Overview

A Ros2-based modular system for to facilitate research and development of quadrotor trajectory planning, optimization and tracking.

There are multiple packages in this repository, each representing a module in the system, this includes simulation, path planning, trajectory generation, trajectory tracking and vision-based mapping.

The modules interact with each other using ROS2 topics in a non-synchronous way, each module in the system can run in a unique rate which simulates the real-life case.

The system modules and their interactions are shown in the following graph:

![Simulation System Diagram](/media/system_simulation.svg)

In real life, the simulation node would be replaced with multiple nodes, namely the quadrotor_command node, quadrotor_motion node and quadrotor_state_estimation node. The first one is responsible for communicating with the quadrotor, giving it the rotor commands and receiving images and IMU data. The quadrotor_motion node communicated with the motion-capture system (if exists). Finally, the quadrotor_state_estimation is responsible for estimating and publishing the current state of the quadrotor. The following diagram present the real-life system.
![Real System Diagram](/media/system_real.svg)

## Implemented Modules and Functionalities

- [x] Simulation: quadrotor_simulation
  - [x] Pybullet: quadrotor_pybullet
- [ ] Tracking: quadrotor_control
  - [x] PID: quadrotor_pid
  - [x] DFBC
- [ ] Path Planning : quadrotor_path_finding
  - [x] RRT : quadrotor_rrt
- [ ] Trajectory Generation : quadrotor_trajectory_generation
  - [x] 3rd order polynomials (no optimization) - quadrotor_poly_optimizer
  - [x] Higher order polynomials (with optimization)

## Installation

This project was created for ROS2 humble in Ubuntu 22.04.

There are multiple packages in this repository, it's preferable to install them in one workspace.

First source the ROS2 humble.

```bash
source /opt/ros/humble/setup.bash
```

Next, navigate to your preferable folder and download the repository

```bash
cd /path/to/work/folder
mkdir quadrotor_ws && cd quadrotor_ws
#git clone https://github.com/ZeinBarhoum/quadrotor-plan-control.git #when it becomes public
git clone git@github.com:ZeinBarhoum/quadrotor-plan-control.git
```

Install dependencies:

```bash
cd quadrotor-plan-control && rosdep install --from-paths src --ignore-src && pip install -r requirements.txt
```

Finally, build the workspace.

```bash
colcon build
```

For development, it's better to use symlink-install

```bash
colcon build --symlink-install
```

Don't forget to source the workspace every time before usage

```bash
source install/setup.bash
```

## Usage

### Bring-up Package `quadrotor_bringup`

This package contains launch files for the entire system.
Type the following in the terminal to start simulation, control, path finding, trajectory generation and dashboard packages at once:

```bash
ros2 launch quadrotor_bringup quadrotor_full.launch.py 
```

From here you can command the system using three different mechanisms:

1. Direct position control (step): publish to `quadrotor_reference` topic
2. Trajectory tracking control: publish a polynomial to the `quadrotor_polynomial_trajectory` topic.
3. Waypoint tracking: publish set of waypoints to the `quadrotor_waypoints` topic.
4. Path planning: publish a map to the `quadrotor_map` topic and then a target position to the `quadrotor_plan_command` to automatically collision-free path.

More detailed usage is discussed below. For information about packages, nodes and parameters, check the wiki!

### Simulation Package `quadrotor_simulation`

#### Nodes

The simulation package (`quadrotor_simulation`) contains 3 types of nodes:

- Physics Node: `quadrotor_pybullet_physics`, responsible for taking the input (rotor speeds) and calculating the external forces/torques applied on the quadrotor (nominal, aerodynamic and residuals) and then integrate over time to get the quadrotor state (pose + twist) at every time.
- Camera/Scene Node: `quadrotor_pybullet_camera`, responsible for taking the pose of the quadrotor at every time and visualize the quadrotor with this pose in the environment (scene environment can be different from physics environment), in addition, this scene is responsible simulating a Camera by calculating the projection of the scene on the image plane.
- IMU Node: `quadrotor_imu`, responsible for simulating an IMU sensor by inducing noises with specific covariance and so on.

To start the physics simulation with default parameters:

```bash
ros2 run quadrotor_simulation quadrotor_pybullet_physics 
```

This will start the simulation physics node. This node publishes to `quadrotor_state` topic and subscribes to `quadrotor_rotor_speeds` topic.

While the `quadrotor_pybullet_physics` node is running, camera and IMU sensors can be started using the commands below (each one in a separate terminal)

```bash
# for camera
ros2 run quadrotor_simulation quadrotor_pybullet_camera 
# for IMU
ros2 run quadrotor_simulation quadrotor_imu
```

The `quadrotor_pybullet_camera` node publishes to the `quadrotor_img` topic while the `quadrotor_imu` node publishes to the `quadrotor_imu` topic

Note: `quadrotor_pybullet_camera` will open a new OpenGL window showing the quadrotor. So if the camera node will be used, it's recommended to set the `physics_server` parameter of the `quadrotor_pybullet_physics` to `DIRECT` as follows:

```bash
ros2 run quadrotor_simulation quadrotor_pybullet_physics --ros-args -p physics_server:='DIRECT'
```

#### Launch

For ease of usage, the following launch command will launch physics, camera and IMU nodes together:

```bash
ros2 launch quadrotor_simulation quadrotor_simulation.launch.py
```

### Control Package `quadrotor_control`

#### Nodes

A controller node subscribes to the `quadrotor_state` and `quadrotor_reference` topics and publishes to `quadrotor_rotor_speeds` topic. Currently, there are two controllers implemented in the `quadrotor_control` package which are: Differential Flatness Based Controller (DFBC) `quadrotor_dfbc` and Cascaded PID controller `quadrotor_pid`

Type the following in a separate terminal to run the DFBC controller node:

```bash
ros2 run quadrotor_control quadrotor_dfbc 
```

Note: initially, this node takes the position `[0,0,1]` as a reference

To command the quadrotor to go to a specific position, we need to publish a position (and optionally velocity and acceleration) to the `quadrotor_reference` topic. Run the following:

```bash
ros2 topic pub /quadrotor_reference quadrotor_interfaces/msg/ReferenceState "{current_state: {pose: {position: {x: 2.0, y: 2.0, z: 3.0}}}}" --once
```

If you have a reference trajectory parameterized as a polynomial, you can start the `quadrotor_reference_publisher` node to take this trajectory and continuously publish reference states, to start it run the following:

```bash
ros2 run quadrotor_control quadrotor_reference_publisher
```

This node subscribes to the `quadrotor_polynomial_trajectory` topic and publishes to the `quadrotor_reference` topic.

For example, to command the quadrotor to follow a circle, the following command can be used (polynomial approximation of the circle)

```bash
ros2 topic pub --once /quadrotor_polynomial_trajectory quadrotor_interfaces/msg/PolynomialTrajectory "{header: {}, n: 1, segments: [{poly_x: [1.51383985371781e-07, -7.14571397410070e-06, 0.000130239969470174, -0.00109778199648815, 0.00371446637221780, -0.000980038295903419, 0.0134900091914051, -0.176415378615748, 0.00200274735664833, 1.00066799320626, -0.000130544080559754], poly_y: [2.21841764793139e-07, -6.96929094200916e-06, 7.42738645341275e-05, -0.000215896450055789, -0.000777698773923802, -0.00112904042577661, 0.0429934777291669,  -0.000934805450807571, -0.499647157157789, -5.69214830909161e-05, 1.00000216767309], poly_z: [2.0], poly_yaw: [0.0], start_time: 0.0, end_time: 6.28}]}"
```

#### Launch

To start the controller and reference publisher nodes together run the following launch command:

```bash
ros2 launch quadrotor_control quadrotor_control.launch.py 
```

### Planning Packages

#### Trajectory Generation Package `quadrotor_trajectory_generation`

To command the quadrotor to follow a sequence of waypoints, the node `quadrotor_poly_optimizer` from the `quadrotor_trajectory_generation` package can be used to publish to the `quadrotor_polynomial_trajectory` after receiving a sequence of waypoints on the topic `quadrotor_waypoints`.
Type the following to start this node

```bash
ros2 run quadrotor_trajectory_generation quadrotor_poly_optimizer 
```

An example of commanding the quadrotor to follow a triangle is:

```bash
ros2 topic pub --once /quadrotor_waypoints quadrotor_interfaces/msg/PathWayPoints "{waypoints: [{x: 1, y: 1, z: 1}, {x: 4, y: 4, z: 1}, {x: 1, y: 4, z: 2}], heading_angles: [0.0, 0.0, 0.0]}"
```

Note: the `quadrotor_poly_optimizer` also subscribes to the `quadrotor_map` topic to ensure collision-free trajectory generation.

#### Path Finding Package `quadrotor_path_finding`

To plan a collision free path from the current position (retrieved from the `quadrotor_state` topic) to a target position published to the `quadrotor_plan_command`, we can use the RRT planner node `quadrotor_rrt` from the `quadrotor_path_finding` package as follows:

First, run the `quadrotor_rrt`:

```bash
ros2 run quadrotor_path_finding quadrotor_rrt 
```

To command the quadrotor to go to a specific position avoiding any collisions, use the following:

```bash
ros2 topic pub /quadrotor_plan_command geometry_msgs/msg/Point "{x : 5.0, y : 5.0, z : 5.0}" --once
```

The planner make use of the Occupancy Map published to the `quadrotor_map` topic, if no map is published, it assumes a 10x10x10 empty map.


## Modularity

Each node in this repository can be run in a stand-alone fashion as long as there is data published to the relevant topics. For example, the quadrotor_dfbc node can be used for other projects as long as someone is publishing to the `quadrotor_state` and `quadrotor_reference` topics.
