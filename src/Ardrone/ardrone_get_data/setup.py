from setuptools import find_packages, setup

package_name = 'ardrone_get_data'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dog',
    maintainer_email='semenpozizni@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ardrone_get_video = ardrone_get_data.ardrone_get_video:main',
            'ardrone_get_mocapOptiTrack = ardrone_get_data.ardrone_get_mocapOptiTrack:main'
        ],
    },
)
