"""Common map transformations functions"""
from quadrotor_interfaces.msg import OccupancyGrid3D
from geometry_msgs.msg import Point
import numpy as np
from typing import Union, Tuple
import numpy.typing as npt


def OccupancyGrid3D_to_voxelmap(grid: OccupancyGrid3D) -> np.ndarray:
    """Converts a 3D occupancy grid map to a 3D NumPy array.

    Args:
        grid (OccupancyGrid3D): The 3D occupancy grid map.

    Returns:
        np.ndarray: The 3D NumPy array.
    """
    n = grid.width
    m = grid.height
    k = grid.depth
    data = grid.data
    return np.array(data).reshape(n, m, k)


def voxelmap_to_OccupancyGrid3D(map: np.ndarray, cell_size: float = 0.05, origin: Union[Tuple[float, float, float], str] = 'middle') -> OccupancyGrid3D:
    """Converts a 3D NumPy array to a 3D occupancy grid map.

    Args:
        map (np.ndarray): The 3D NumPy array.
        cell_size (float, optional): Size of each voxel in the map (in meters). Defaults to 0.05.
        origin (Union[Tuple[float, float, float], str], optional): Position of the (0,0,0) voxel in the world, 'middle' if the map is centered in xy. Defaults to 'middle'.

    Raises:
        ValueError: _description_

    Returns:
        OccupancyGrid3D: _description_
    """
    grid = OccupancyGrid3D()
    grid.width = map.shape[0]
    grid.height = map.shape[1]
    grid.depth = map.shape[2]
    if (isinstance(origin, str)):
        if (origin == 'middle'):
            origin = (-np.fix(map.shape[0]/2)*cell_size, -np.fix(map.shape[1]/2)*cell_size, 0)
        else:
            raise ValueError('Invalid origin string')
    grid.origin = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
    grid.cell_size = cell_size
    grid.data = map.flatten().astype(np.int8).tolist()
    return grid


def point_to_voxel(grid: OccupancyGrid3D, point: npt.ArrayLike, with_round: bool = False) -> np.ndarray:
    """Converts a point in 3D space to a voxel in the given 3D occupancy grid map.

    Args:
        map (OccupancyGrid3D): The 3D occupancy grid map.
        point (npt.ArrayLike): The point in 3D space to convert to a voxel.
        round (bool, optional): Whether to round the voxel coordinates to the nearest integer. Defaults to False.
        raise_out_of_range (bool, optional): Whether to raise a ValueError if the point is outside the map bounds. Defaults to True.

    Returns:
        np.ndarray: The voxel coordinates as a numpy array of shape (3,).

    Raises:
        ValueError: If the point is not a vector.
        ValueError: If the point is not 3D.
        ValueError: If the point is outside the map bounds.
    """
    point = np.array(point)
    if point.ndim != 1:
        raise ValueError('Point must be a vector')
    if point.shape[0] != 3:
        raise ValueError('Point must be 3D')

    origin = grid.origin
    cell_size = grid.cell_size
    n = grid.width
    m = grid.height
    k = grid.depth

    x = (point[0] - origin.x) / cell_size
    y = (point[1] - origin.y) / cell_size
    z = (point[2] - origin.z) / cell_size

    if x < 0 or x >= n:
        raise ValueError('x out of range')
    if y < 0 or y >= m:
        raise ValueError('y out of range')
    if z < 0 or z >= k:
        raise ValueError('z out of range')

    if with_round:
        x = round(x)
        y = round(y)
        z = round(z)

    return np.array([x, y, z])


def voxel_to_point(grid: OccupancyGrid3D, voxel: npt.ArrayLike) -> np.ndarray:
    """Converts a voxel in the given 3D occupancy grid map to a point in 3D space.

    Args:
        map (OccupancyGrid3D): The 3D occupancy grid map.
        voxel (npt.ArrayLike): The voxel in the map to convert to a point.

    Returns:
        np.ndarray: The point in 3D space as a numpy array of shape (3,).

    Raises:
        ValueError: If the voxel is not a vector.
        ValueError: If the voxel is not 3D.
        ValueError: If the voxel is outside the map bounds.
    """
    voxel = np.array(voxel)
    origin = grid.origin
    cell_size = grid.cell_size

    if voxel.ndim != 1:
        raise ValueError('Voxel must be a vector')
    if voxel.shape[0] != 3:
        raise ValueError('Voxel must be 3D')
    if voxel[0] < 0 or voxel[0] >= grid.width:
        raise ValueError('Voxel x out of range')
    if voxel[1] < 0 or voxel[1] >= grid.height:
        raise ValueError('Voxel y out of range')
    if voxel[2] < 0 or voxel[2] >= grid.depth:
        raise ValueError('Voxel z out of range')

    x = voxel[0] * cell_size + origin.x
    y = voxel[1] * cell_size + origin.y
    z = voxel[2] * cell_size + origin.z

    return np.array([x, y, z])
