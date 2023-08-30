from setuptools import setup

package_name = 'quadrotor_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
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
            'quadrotor_dfbs = quadrotor_control.quadrotor_dfbs:main',
        ],
    },
)
