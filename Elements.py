from Point import Point
import cv2

class UAV(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(UAV, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("uav_blue.png") / 255.0
        self.icon_w = 5
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class CAR(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(CAR, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("car.png") / 255.0
        self.icon_w = 5
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

class Person(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Person, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("person.png") / 255.0
        self.icon_w = 1
        self.icon_h = 1
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class Wall_H(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Wall_H, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("wall_H_red.png") / 255.0
        self.icon_w = 5
        self.icon_h = 20
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class Wall_V(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Wall_V, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("wall_V_red.png") / 255.0
        self.icon_w = 20
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        
class Person(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Person, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("person.png") / 255.0
        self.icon_w = 8
        self.icon_h = 8
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

    
class Fire(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fire, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("fire.png") / 255.0
        self.icon_w = 2
        self.icon_h = 2
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
    
class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Target, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("door_green.png") / 255.0
        self.icon_w = 10
        self.icon_h = 10
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))