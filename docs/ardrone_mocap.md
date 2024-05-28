# Tutorial of Controlling ArDrone2.0 with OptiTrack system 

1. Run ardrone_autonomy package:
```
cd ~/Project/experiments/ros_ws/ 
source /opt/ros/noetic/setup.bash 
source devel/setup.bash 
roslaunch ardrone_autonomy ardrone.launch
```

2. Run ros1_bridge to communicate between ros1 and ros2 

```
cd ~/ros1_bridge/
source /opt/ros/noetic/setup.bash
source /opt/ros/humble/setup.bash 
source install/setup.bash 
ros2 run ros1_bridge parameter_bridge
```

3. Run OptiTrack node: 

```
cd ~/Project/quadrotor-plan-control/ 
source /opt/ros/humble/setup.bash 
source install/setup.bash 
ros2 launch mocap4r2_optitrack_driver optitrack2.launch.py
```

In another terminal 
```
ros2 lifecycle set /mocap4r2_optitrack_driver_node activate
```

4. Run mapping between quadrotor_state and ardrone/imu and mocap data 

```
cd ~/Project/quadrotor-plan-control/ 
source /opt/ros/humble/setup.bash 
source install/setup.bash 
ros2 run quadrotor_communication quadrotor_ardrone_mocap
```
