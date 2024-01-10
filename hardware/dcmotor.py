class DCMotor():

    def __init__(self):
        self.a = 27
        self.b = 17

        self.timeOld = 0
        self.timeNow = 1

        self.WPre = 0
        self.wNow = 0
        self.voltage = 0

    def input_voltage(self, voltage):
        self.voltage = voltage
        Ts = (self.timeNow - self.time_old)
        self.wNow = (1 - (self.a * Ts)) * self.WPre + self.b * Ts * self.voltage
        print(self.wNow)
        self.WPre = self.wNow
        self.time_old = self.timeNow