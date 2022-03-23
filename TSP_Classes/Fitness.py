class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fit = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                previous = self.route[i]
                next = None
                if i + 1 < len(self.route):
                    next = self.route[i + 1]
                else:
                    next = self.route[0]

                pathDistance += previous.getDistance(next)

            self.distance = pathDistance
        return self.distance

    def getFitness(self):
        if self.fit == 0:
            self.fit = 1 / float(self.routeDistance())
        return self.fit