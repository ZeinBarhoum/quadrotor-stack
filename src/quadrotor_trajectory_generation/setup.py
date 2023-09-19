from setuptools import setup

package_name = 'quadrotor_trajectory_generation'

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
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'quadrotor_reference_publisher = quadrotor_trajectory_generation.quadrotor_reference_publisher:main',
            'quadrotor_poly_optimizer = quadrotor_trajectory_generation.quadrotor_poly_optimizer:main',
        ],
    },
)
