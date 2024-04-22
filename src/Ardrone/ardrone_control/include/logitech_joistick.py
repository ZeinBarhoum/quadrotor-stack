from inputs import get_gamepad


class LogitechJoistick:

    permit_check = [0, 0, 0, 0]

    throttleState = 0.0
    xState = 0.0
    yState = 0.0
    # hatXState = 0
    # hatYState = 0
    rotateState = 0.0

    cmd_takeoff = 0
    cmd_land = 0
    cmd_liftoff = 0
    cmd_liftdown = 0
    cmd_switchcamera = 0
    cmd_recrash = 0
    cmd_video_reconnect = 0
    val_max_speed = float(0.3)
    val_lin_twistX = float(0)
    val_lin_twistY = float(0)
    val_lin_twistZ = float(0.5)
    val_ang_twistZ = float(0)

    # Returns whether or not x is within n of y

    def withinNOf(self, x, n, y):
        if x in range(n-y, n+y):
            return True
        else:
            return False

    def getStickPosition(self, centreAssist=True, centreAssistSens=10, edgeAssist=True, edgeAssistSens=5):
        xPos = self.xState - 512
        yPos = (self.yState - 512) * -1

        # Snap the position to the centre if close enough
        if centreAssist == True:
            if self.withinNOf(xPos, 0, centreAssistSens):
                xPos = 0
            if self.withinNOf(yPos, 0, centreAssistSens):
                yPos = 0

        # Snap the position to the edge if close enough
        if edgeAssist == True:
            if self.withinNOf(xPos, 512, edgeAssistSens):
                xPos = 512
            if self.withinNOf(yPos, 512, edgeAssistSens):
                yPos = 512
            if self.withinNOf(xPos, -512, edgeAssistSens):
                xPos = -512
            if self.withinNOf(yPos, -512, edgeAssistSens):
                yPos = -512

        return (xPos, yPos)

    # def getHatPosition(self):
    # 	return (self.hatXState, self.hatYState * -1)

    def getRotatePosition(self, centreAssist=True, centreAssistSens=2, edgeAssist=True, edgeAssistSens=1):
        rotatePos = self.rotateState - 128

        # Snap the position to the centre if close enough
        if centreAssist == True:
            if self.withinNOf(rotatePos, 0, centreAssistSens):
                rotatePos = 0

        # Snap the position to the edge if close enough
        if edgeAssist == True:
            if self.withinNOf(rotatePos, 128, edgeAssistSens):
                rotatePos = 128
            if self.withinNOf(rotatePos, -128, edgeAssistSens):
                rotatePos = -128
        return rotatePos

    def get_gamepad_evenst(self):
        events = get_gamepad()
        for event in events:
            if event.ev_type == "Absolute":
                if event.code == "ABS_THROTTLE":
                    self.throttleState = int(event.state)
                    self.permit_check[0] = True
                elif event.code == "ABS_X":
                    self.xState = int(event.state)
                    self.permit_check[1] = True
                elif event.code == "ABS_Y":
                    self.yState = int(event.state)
                    self.permit_check[2] = True
                # elif event.code == "ABS_HAT0X":
                # 	self.hatXState = int(event.state)
                # elif event.code == "ABS_HAT0Y":
                # 	self.hatYState = int(event.state)
                elif event.code == "ABS_RZ":
                    self.rotateState = int(event.state)
                    self.permit_check[3] = True
            elif event.ev_type == "Key":
                if event.code == "BTN_PINKIE":
                    self.cmd_takeoff = int(event.state)
                elif event.code == "BTN_TOP":
                    self.cmd_land = int(event.state)
                elif event.code == "BTN_TOP2":
                    self.cmd_liftoff = int(event.state)
                elif event.code == "BTN_THUMB2":
                    self.cmd_liftdown = int(event.state)
                elif event.code == "BTN_THUMB":
                    self.cmd_switchcamera = int(event.state)
                elif event.code == "BTN_TRIGGER":
                    self.cmd_recrash = int(event.state)

                elif event.code == "BTN_BASE":
                    self.cmd_video_reconnect = int(event.state)

    def update(self):
        self.get_gamepad_evenst()
        self.val_lin_twistY, self.val_lin_twistX = self.getStickPosition()
        self.val_ang_twistZ = self.getRotatePosition()

        self.val_max_speed = 1 - (self.throttleState / 255.0)

        self.val_lin_twistX = 0 if abs(self.val_lin_twistX) < 100 else self.val_lin_twistX / 512.0
        self.val_lin_twistY = 0 if abs(self.val_lin_twistY) < 100 else self.val_lin_twistY / 512.0
        if (self.cmd_liftdown):
            self.val_lin_twistZ = -1.0
        elif (self.cmd_liftoff):
            self.val_lin_twistZ = 1.0
        else:
            self.val_lin_twistZ = 0.0
        self.val_ang_twistZ = -(self.val_ang_twistZ / 128.0)

    def print(self):
        if self.cmd_takeoff:
            print('Take off')
        elif self.cmd_land:
            print('Land')
        elif self.cmd_switchcamera:
            print('Switch camera')
        else:
            print(f'Twist: [{self.val_lin_twistX:.3f}, {self.val_lin_twistY:.3f}, {self.val_lin_twistZ:.3f}]\n       [{0:.3f}, {0:.3f}, {self.val_ang_twistZ:.3f}]')


# import sys
# joy = LogitechJoistick()
# try:
# 	while True:
# 		joy.update()
# 		joy.print()
# except KeyboardInterrupt:
#     sys.exit(0)
