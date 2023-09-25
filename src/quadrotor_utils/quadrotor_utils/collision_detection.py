"""Common collision detection functions."""
import numpy as np
import numpy.typing as npt
from quadrotor_interfaces.msg import PolynomialSegment, PolynomialTrajectory, OccupancyGrid3D
from quadrotor_utils.map_transformations import OccupancyGrid3D_to_voxelmap, point_to_voxel
from typing import Union, Tuple, List


def detect_collision_voxelmap(voxel_map: npt.ArrayLike, voxel: npt.ArrayLike, collision_on_out_of_range: bool = True) -> bool:
    """Detects collision between a point and a 3D map

    Args:
        voxel_map (ArrayLike): 3D map of the environment
        voxel (ArrayLike): 3D point to check for collision 

    Raises:
        ValueError: Map must be 3D
        ValueError: Point must be a vector
        ValueError: Point must be 3D

    Returns:
        bool: True if there is a collision, False otherwise
    """
    voxel_map = np.array(voxel_map)
    voxel = np.array(voxel)

    if voxel_map.ndim != 3:
        raise ValueError('Map must be 3D')

    if voxel.ndim != 1:
        raise ValueError('Point must be a vector')

    if voxel.shape[0] != 3:
        raise ValueError('Point must be 3D')

    for x in [np.floor(voxel[0]), np.ceil(voxel[0])]:
        for y in [np.floor(voxel[1]), np.ceil(voxel[1])]:
            for z in [np.floor(voxel[2]), np.ceil(voxel[2])]:
                try:
                    if voxel_map[int(x), int(y), int(z)] == 1:
                        return True
                except IndexError as e:
                    if (collision_on_out_of_range):
                        return True
                    else:
                        raise e

    return False


def detect_collision_OccupancyGrid3D(grid: OccupancyGrid3D, point: npt.ArrayLike) -> bool:
    """Detects collision between a point and a 3D map

    Args:
        grid (OccupancyGrid3D): 3D map of the environment
        point (ArrayLike): 3D point to check for collision 

    Returns:
        bool: True if there is a collision, False otherwise
    """
    voxel = point_to_voxel(grid, point)
    voxel_map = OccupancyGrid3D_to_voxelmap(grid)
    return detect_collision_voxelmap(voxel_map, voxel)


def detect_collision_trajectory_segment(grid: OccupancyGrid3D,
                                        segment: PolynomialSegment,
                                        precision: float = 0.1,
                                        ) -> bool:
    """Detects collision between a trajectory segment and a 3D map

    Args:
        grid (OccupancyGrid3D): 3D map of the environment
        segment (PolynomialSegment): Trajectory segment to check for collision
        precision (float, optional): Precision of the trajectory segment in seconds. Defaults to 0.1.
    Returns:
        bool: True if there is a collision, False otherwise
    """
    poly_x = segment.poly_x
    poly_y = segment.poly_y
    poly_z = segment.poly_z
    t0 = segment.start_time
    tend = segment.end_time

    test_times = np.arange(t0, tend, precision)
    for t in test_times:
        x = np.polyval(poly_x, t)
        y = np.polyval(poly_y, t)
        z = np.polyval(poly_z, t)
        point = np.array([x, y, z])
        if detect_collision_OccupancyGrid3D(grid, point):
            return True
    return False


def detect_collision_trajectory(grid: OccupancyGrid3D,
                                trajectory: PolynomialTrajectory,
                                precision: float = 0.1,
                                ) -> Union[bool, Tuple[bool, List[int]]]:
    """Detects collision between a trajectory and a 3D map

    Args:
        grid (OccupancyGrid3D): 3D map of the environment
        trajectory (PolynomialTrajectory): Trajectory to check for collision
        precision (float, optional): Precision of the trajectory segment in seconds. Defaults to 0.1.
    Returns:
        bool: True if there is a collision, False otherwise
        List[int]: Indices of the segments where the collision happens
    """
    segments = trajectory.segments
    collision_segments_indices = []
    for (i, segment) in enumerate(segments):
        if detect_collision_trajectory_segment(grid, segment, precision):
            collision_segments_indices.append(i)
    if len(collision_segments_indices) > 0:
        return (True, collision_segments_indices)
    return False
