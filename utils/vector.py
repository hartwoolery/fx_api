from __future__ import annotations
import math
import numpy as np

class Vector(tuple):
    @property
    def x(self):
        return self[0]
      
    
    @property
    def y(self):
        return self[1]
        

    def __new__(cls, x, y=None):
        
        
        if y is not None and isinstance(y, (tuple, list)):
            asdf = asde
            
        if isinstance(x, (tuple, list, np.ndarray)):
            # Handle tuple/list/array input
            return super().__new__(cls, (x[0], x[1]))
        # Handle separate x,y coordinates
        return super().__new__(cls, (x, y))
    
    def length(self) -> float:
        return math.sqrt(self[0]**2 + self[1]**2)
    
    def distance(self, other) -> float:
        return math.sqrt((self[0] - other[0])**2 + (self[1] - other[1])**2)
    
    def normalize(self) -> Vector:
        length = self.length()
        if length == 0:
            return Vector(0, 0)
        return Vector(self[0] / length, self[1] / length)
    
    def angle(self) -> float:
        return np.degrees(np.arctan2(self[1], self[0]))
    
    def rotate(self, angle:float) -> Vector:
        dx, dy = self
        # Rotate the point around the center
        angle_rad = np.radians(angle)  # Convert to radians 
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        # Rotation matrix multiplication
        rotated_x = dx * cos_theta - dy * sin_theta 
        rotated_y = dx * sin_theta + dy * cos_theta 

        return Vector(rotated_x, rotated_y)
    
    def clamp(self, min_value:float, max_value:float) -> Vector:
        return Vector(max(min_value, min(self[0], max_value)), max(min_value, min(self[1], max_value)))
    
    def round(self) -> Vector:
        return Vector(round(self[0]), round(self[1]))
    
    def sign(self) -> Vector:
        return Vector(np.sign(self[0]), np.sign(self[1]))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self[0] + other, self[1] + other)
        return Vector(self[0] + other[0], self[1] + other[1])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self[0] - other, self[1] - other)
        return Vector(self[0] - other[0], self[1] - other[1])
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other - self[0], other - self[1])
        return Vector(other[0] - self[0], other[1] - self[1])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self[0] * other, self[1] * other)
        return Vector(self[0] * other[0], self[1] * other[1])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self[0] / other, self[1] / other)
        return Vector(self[0] / other[0], self[1] / other[1])
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other / self[0], other / self[1])
        return Vector(other[0] / self[0], other[1] / self[1])
    
    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self[0] // other, self[1] // other)
        return Vector(self[0] // other[0], self[1] // other[1])
    
    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other // self[0], other // self[1])
        return Vector(other[0] // self[0], other[1] // self[1])
    
    def __neg__(self) -> Vector:
        return Vector(-self[0], -self[1])
    
    def __abs__(self) -> Vector:
        return Vector(abs(self[0]), abs(self[1]))
    
    def __round__(self) -> Vector:
        return Vector(round(self[0]), round(self[1]))
    
    def __floor__(self) -> Vector:
        return Vector(math.floor(self[0]), math.floor(self[1]))
    
    def __ceil__(self) -> Vector:
        return Vector(math.ceil(self[0]), math.ceil(self[1]))
    
    def __trunc__(self) -> Vector:
        return Vector(math.trunc(self[0]), math.trunc(self[1]))
    
    def __mod__(self, value: Vector) -> Vector:
        return Vector(self[0] % value, self[1] % value)
    
    def __rmod__(self, value: Vector) -> Vector:
        return Vector(value % self[0], value % self[1])
    
    def __pow__(self, value: Vector) -> Vector:
        return Vector(self[0] ** value, self[1] ** value)
    
    def __rpow__(self, value: Vector) -> Vector:
        return Vector(value ** self[0], value ** self[1])
    
    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] == other and self[1] == other
        return self[0] == other[0] and self[1] == other[1]
    
    def __ne__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] != other or self[1] != other
        return self[0] != other[0] or self[1] != other[1]
    
    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] < other and self[1] < other
        return self[0] < other[0] and self[1] < other[1]
    
    def __le__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] <= other and self[1] <= other
        return self[0] <= other[0] and self[1] <= other[1]
    
    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] > other and self[1] > other
        return self[0] > other[0] and self[1] > other[1]
    
    def __ge__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self[0] >= other and self[1] >= other
        return self[0] >= other[0] and self[1] >= other[1]
    