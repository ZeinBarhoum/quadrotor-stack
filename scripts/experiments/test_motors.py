import time
from pyardrone import ARDrone
from pyardrone import at
drone = ARDrone()
drone.emergency()
def move_drone_pwm(motor1=0, motor2=0, motor3=0, motor4=0):
    if motor1 >= 0 and motor1 <= 511 and\
       motor2 >= 0 and motor2 <= 511 and\
       motor3 >= 0 and motor3 <= 511 and\
       motor4 >= 0 and motor4 <= 511:
        drone.send(at.PWM(motor1, motor2, motor3, motor4))
        return True, ''
    else:
        return False, 'Input pwm value error'
try:
    for i in range(0,512,50):
        print(i)
        move_drone_pwm(0, 0, 0, i)
        if i == 50:
            time.sleep(2)
        time.sleep(1)
    for i in reversed(range(0,512,50)):
        print(i)
        move_drone_pwm(0, 0, 0, i)
        time.sleep(1)

    # for i in reversed(range(0,500,1)):
    #     print(i)
    #     move_drone_pwm(i, i, i, i)
    #     time.sleep(0.1)
    # i = 10
    # while True:
    #     move_drone_pwm(i, 0, 0, 0)
finally:
    move_drone_pwm(0, 0, 0, 0)
    drone.close()
