From ros:humble
# # Set the working directory
WORKDIR /quadrotor_ws


RUN rm -r build install log

RUN apt-get update && apt-get install -y \
    python3-pip \
    xauth 
    # nvidia-container-runtime 

# Copy the ROS2 workspace into the Docker image
    COPY .. /quadrotor_ws

RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN pip install -r requirements.txt

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

#
CMD ["bash"]
