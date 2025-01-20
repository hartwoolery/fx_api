import cv2
import numpy as np
from fx_api.utils.image import ImageUtils
from fx_api.utils.vector import Vector
from typing import Callable

class Button:
    def __init__(self, action:Callable, text:str, size:Vector, color:tuple[int, int, int], font:str, font_size:int, font_color:tuple[int, int, int]):
        self.action = action
        self.text = text
        self.size = size
        self.color = color
        self.font = font
        self.font_size = font_size
        self.font_color = font_color
        self.image = None

    def set_image(self, image):
        self.image = image

    def draw(self, frame, sprite_manager):
        cv2.rectangle(frame, self.position, self.position + self.size, self.color, -1)
        cv2.putText(frame, self.text, self.position + self.size // 2, self.font, self.font_size, self.font_color, 2)
        if self.image is not None:
            ImageUtils.blend(frame, self.image, self.position, self.size)

    def on_mouse_move(self, mouse_pos):
        pass

    def on_mouse_down(self, mouse_pos):
        pass

    def on_mouse_up(self, mouse_pos):
        pass
