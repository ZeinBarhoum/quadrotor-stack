from setuptools import setup
import os
from glob import glob

package_name = 'quadrotor_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'world'), glob('world/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zein Alabedeen Barhoum',
    maintainer_email='zein.barhoum799@gmail.com',
    description='Quadrotor simulation that include different simulators',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadrotor_pybullet = quadrotor_simulation.quadrotor_pybullet:main',
            'quadrotor_pybullet_dataset = quadrotor_simulation.quadrotor_pybullet_dataset:main'
        ],
    },
)
