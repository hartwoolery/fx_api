import cv2
import numpy as np
from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.sprites.base_sprite import BaseSprite

############################################################
# Image Sprite
############################################################

class ImageSprite(BaseSprite):
    def __init__(self, sprite_manager, image_path:str, unique_id:int):
        
        super().__init__(sprite_manager, None, unique_id, "image")
        self.image_path = image_path
        self.change_sprite_path(image_path)

    def get_sprite_image(self, frame_info:FrameInfo):
        if self.image is None:
            return None
        opacity = self.get_opacity()
        rgba = self.image.copy()
        # Apply opacity to alpha channel
        rgba[:,:,:3] = self.recolor_sprite(rgba[:,:,:3])
        if rgba.shape[2] == 4:  # Check if image has alpha channel
            rgba[:, :, 3] = rgba[:, :, 3] * opacity
        return rgba

    def change_sprite_path(self, path:str):
        self.image_path = path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        # Add alpha channel if image only has 3 channels (BGR)
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            #if jpeg, add alpha channel
            # Create full opacity alpha channel
            alpha = np.full((self.image.shape[0], self.image.shape[1], 1), 255, dtype=np.uint8)
            # Stack BGR with alpha channel
            self.image = np.dstack((self.image, alpha))


        #set this for the world sprite
        resolution = Vector(self.image.shape[1], self.image.shape[0])
        if resolution.x > 512:
            new_scale = 512 / resolution.x
            resolution *= new_scale

        self.bbox = (0,0, resolution.x, resolution.y)
        
        self.true_size = resolution
        self.update_bbox()

    def render(self, frame_info:FrameInfo, transform:dict={}):
        super().render(frame_info, transform)
        if self.image is None or not self.get_enabled():
            return
        
        rgba = self.get_sprite_image(frame_info)
        self.blit_sprite(frame_info, rgba, self.image is not None)
