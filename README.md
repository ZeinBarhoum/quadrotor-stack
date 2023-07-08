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

- [ ] Simulation: quadrotor_simulation
  - [x] Pybullet: quadrotor_pybullet
  - [ ] ??
- [ ] Tracking: quadrotor_control
  - [x] PID: quadrotor_pid
  - [ ] MPC
  - [ ] DFBC
- [ ] Mapping
- [ ] Path Planning
- [ ] Trajectory Generation

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
mkdir quadrotor_ws
git clone https://github.com/ZeinBarhoum/quadrotor-plan-control.git
```

Finally, build the workspace and source the install-file.

```bash
colcon build
source install/setup.bash
```

## Usage

To start the simulation node run the following

```bash
ros2 run quadrotor_simulation quadrotor_pybullet
```

This node listens to the `/quadrotor_rotor_speeds` topic and publishes to `/quadrotor_state` topic.

To launch a simulation with PID controller and simple reference publisher, use the following command.

```bash
ros2 launch quadrotor_bringup quadrotor_simulation.launch.py controller:=quadrotor_pid
```

To command the quadrotor to follow a circle, the following command publishes the polynomial trajectory approximation of a circle to the `/quadrotor_polynomial_trajectory`. The node `quadrotor_reference_publisher` publishes the reference state to `/quadrotor_reference` topic in fixed rate following the polynomial function.

```bash
ros2 topic pub --once /quadrotor_polynomial_trajectory quadrotor_interfaces/msg/PolynomialTrajectory "{header: {}, poly_x: [1.51383985371781e-07, -7.14571397410070e-06, 0.000130239969470174, -0.00109778199648815, 0.00371446637221780, -0.000980038295903419, 0.0134900091914051, -0.176415378615748, 0.00200274735664833, 1.00066799320626, -0.000130544080559754], poly_y: [2.21841764793139e-07, -6.96929094200916e-06, 7.42738645341275e-05, -0.000215896450055789, -0.000777698773923802, -0.00112904042577661, 0.0429934777291669,  -0.000934805450807571, -0.499647157157789, -5.69214830909161e-05, 1.00000216767309], poly_z: [2.0], t_clip: 6.28}"
```

To command the quadrotor to follow a sequence of waypoints, the node quadrotor_poly_optimizer is responsible to publish to the `/quadrotor_polynomial_trajectory` after receiving a sequence of waypoints on the topic `quadrotor_waypoints`. For now, it's not much an optimization task as it's a trajectory generation task with fixed 1 seconds between each two waypoints,

An example of commanding the quadrotor to follow a triangle is:

```bash
ros2 topic pub --once /quadrotor_waypoints quadrotor_interfaces/msg/PathWayPoints "{waypoints: [{x: 1, y: 1, z: 1}, {x: 2, y: 2, z: 2}, {x: 1, y: 1, z: 2}]}"
```

Note: The waypoints are considered circular, which means after completing the last one, it returns to the first.

Each node in this repository can be run in a stand-alone fashion as long as there is data published to the relevant topics. For example, the quadrotor_pid node can be used for other projects as long as someone is publishing to the `/quadrotor_state` and `/quadrotor_reference` topics.
