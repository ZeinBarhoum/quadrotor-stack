# Quadrotor Trajectory Planning and Control

## Overview

A Ros2-based modular system for to facilitate research and development of quadrotor trajectory planning, optimization and tracking.

There are multiple packages in this repository, each representing a module in the system, this includes simulation, path planning, trajectory generation, trajectory tracking and vision-based mapping.

The modules interact with each other using ROS2 topics in a non-synchronous way, each module in the system can run in a unique rate which simulates the real-life case.

The system modules and their interactions are shown in the following graph:

![alt text](/media/system.svg)

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
