from setuptools import setup
from glob import glob
import os

package_name = 'quadrotor_control'

resource_paths = []
directories = glob('resource/')+glob('resource/*/')+glob('resource/*/*/')
for directory in directories:
    resource_paths.append((os.path.join('share', package_name, directory), glob(f'{directory}/*.*')))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (os.path.join('share', package_name, 'resource'), glob('resource/*', recursive=True)),
    ] + resource_paths,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zein Alabedeen Barhoum',
    maintainer_email='zein.barhoum799@gmail.com',
    description='Trajectory Tracking for Quadrotor',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadrotor_pid = quadrotor_control.quadrotor_pid:main',
            'quadrotor_dfbc = quadrotor_control.quadrotor_dfbc:main',
            'quadrotor_dataset = quadrotor_control.quadrotor_dataset:main',
        ],
    },
)
