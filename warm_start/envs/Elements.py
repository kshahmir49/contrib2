from pathlib import Path
import cv2
from Point import Point

ASSETS = Path(__file__).resolve().parent / "assets"


def load_icon(filename, width, height):
    image = cv2.imread(str(ASSETS / filename))
    if image is None:
        raise FileNotFoundError(f"Missing icon {ASSETS / filename}")
    return cv2.resize(image / 255.0, (width, height))


class UAV(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 5
        self.icon_h = 5
        self.icon = load_icon("uav_blue.png", self.icon_h, self.icon_w)


class CAR(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 5
        self.icon_h = 5
        self.icon = load_icon("car.png", self.icon_h, self.icon_w)


class Person(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super().__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 8
        self.icon_h = 8
        self.icon = load_icon("person.png", self.icon_h, self.icon_w)
