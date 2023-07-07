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

To launch a simulation with PID controller, use the following command.

```bash
ros2 launch quadrotor_bringup quadrotor_simulation.launch.py controller:=quadrotor_pid
```

Each node in this repository can be run in a stand-alone fashion as long as there is data published to the relevant topics. For example, the quadrotor_pid node can be used for other projects as long as someone is publishing to the `/quadrotor_state` and `/quadrotor_reference` topics.
