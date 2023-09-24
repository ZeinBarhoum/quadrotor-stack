import numpy as np
from quadrotor_utils.map_transformations import OccupancyGrid3D_to_voxelmap, voxelmap_to_OccupancyGrid3D, point_to_voxel, voxel_to_point
from quadrotor_interfaces.msg import OccupancyGrid3D
from geometry_msgs.msg import Point


def test_OccupancyGrid3D_to_voxelmap():
    # Create a sample OccupancyGrid3D object
    grid = OccupancyGrid3D()
    grid.width = 2
    grid.height = 2
    grid.depth = 2
    grid.data = [0, 1, 1, 0, 0, 1, 0, 1]

    # Convert the OccupancyGrid3D object to a voxel map
    voxel_map = OccupancyGrid3D_to_voxelmap(grid)

    # Define the expected output
    expected_output = np.array([[[0, 1], [1, 0]], [[0, 1], [0, 1]]])

    # Check that the output matches the expected output
    assert np.array_equal(voxel_map, expected_output)


def test_voxelmap_to_OccupancyGrid3D():
    # Create a sample voxel map
    voxel_map = np.array([[[0, 1], [1, 0]], [[0, 1], [0, 1]]])

    # Convert the voxel map to an OccupancyGrid3D object
    grid = voxelmap_to_OccupancyGrid3D(voxel_map)

    # Define the expected output
    expected_output = OccupancyGrid3D()
    expected_output.width = 2
    expected_output.height = 2
    expected_output.depth = 2
    expected_output.origin = Point(x=-0.05, y=-0.05, z=0.0)
    expected_output.cell_size = 0.05
    expected_output.data = [0, 1, 1, 0, 0, 1, 0, 1]

    # Check that the output matches the expected output
    assert grid.width == expected_output.width
    assert grid.height == expected_output.height
    assert grid.depth == expected_output.depth
    assert grid.origin.x == expected_output.origin.x
    assert grid.origin.y == expected_output.origin.y
    assert grid.origin.z == expected_output.origin.z
    assert grid.cell_size == expected_output.cell_size
    assert grid.data == expected_output.data


def test_point_to_voxel():
    # Create a sample OccupancyGrid3D object
    grid = OccupancyGrid3D()
    grid.width = 2
    grid.height = 2
    grid.depth = 2
    grid.origin = Point(x=0.0, y=0.0, z=0.0)
    grid.cell_size = 1.0
    grid.data = [0, 1, 1, 0, 0, 1, 0, 1]

    # Define a sample point
    point = np.array([1.5, 1.5, 1.5])

    # Convert the point to a voxel
    voxel = point_to_voxel(grid, point)

    # Define the expected output
    expected_output = np.array([1.5, 1.5, 1.5])

    # Check that the output matches the expected output
    assert np.array_equal(voxel, expected_output)


def test_voxel_to_point():
    # Create a sample OccupancyGrid3D object
    grid = OccupancyGrid3D()
    grid.width = 2
    grid.height = 2
    grid.depth = 2
    grid.origin = Point(x=0.0, y=0.0, z=0.0)
    grid.cell_size = 1.0
    grid.data = [0, 1, 1, 0, 0, 1, 0, 1]

    # Define a sample voxel
    voxel = np.array([1, 1, 1])

    # Convert the voxel to a point
    point = voxel_to_point(grid, voxel)

    # Define the expected output
    expected_output = np.array([1.0, 1.0, 1.0])

    # Check that the output matches the expected output
    assert np.array_equal(point, expected_output)
