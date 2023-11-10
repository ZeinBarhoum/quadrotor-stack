from setuptools import setup
from glob import glob
package_name = 'quadrotor_dashboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name + '/resource', glob("resource/*.ui")),
        ('share/' + package_name, glob("*.xml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zein',
    maintainer_email='zein.barhoum799@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadrotor_path_visualizer = quadrotor_dashboard.quadrotor_path_visualizer:main',
            'quadrotor_image_visualizer = quadrotor_dashboard.quadrotor_image_visualizer:main',
            'quadrotor_model_errors_visualizer = quadrotor_dashboard.quadrotor_model_errors_visualizer:main',
            'rqt_plan_command = quadrotor_dashboard.main:main',
            'rqt_publish_waypoints = quadrotor_dashboard.plugin_publish_waypoints:main',
        ],
    },
)
