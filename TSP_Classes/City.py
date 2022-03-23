import math

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getDistance(self, city):
        x = abs(self.x - city.x)
        y = abs(self.y - city.y)
        return math.sqrt((x ** 2) + (y ** 2))