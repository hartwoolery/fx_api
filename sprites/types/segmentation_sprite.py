import cv2
import numpy as np
from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.sprites.base_sprite import BaseSprite

############################################################
# Segmentation Sprite
############################################################

class SegmentationSprite(BaseSprite):               
    def __init__(self, sprite_manager, object_info: ObjectInfo, unique_id:int, type:str="cutout"):
        super().__init__(sprite_manager, object_info, unique_id, type)
        self.is_segmentation_sprite = True
    
    def get_sprite_image(self, frame_info:FrameInfo):
        opacity = self.get_opacity()
        ############################################################
        # Opacity
        ############################################################
        x1, y1, x2, y2 = map(int, self.bbox)
        mask_crop = self.mask[y1:y2, x1:x2].astype(np.uint8) * int(255 * opacity)
        frame_crop = frame_info.frame[y1:y2, x1:x2]
        
        #self.estimate_rotation(mask_crop, frame_crop)

        frame_crop = self.recolor_sprite(frame_crop)
            
        if self.sprite_manager.selected_sprite == self:
            frame_crop = 0.8 * frame_crop.astype(np.float32)
            if self.sprite_manager.dragging_sprite:
                mask_crop = 0.8 * mask_crop.astype(np.float32)


        ############################################################
        # Calculate the corners and anchor of the transformed image
        ############################################################

        corners = self.bbox_corners
        # Calculate bbox size
        bbox_width = max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]) - min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
        bbox_height = max(corners[0][1], corners[1][1], corners[2][1], corners[3][1]) - min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
        bbox_size = (bbox_width, bbox_height)



        # Stack RGB channels from frame_crop with alpha mask
        rgba = np.dstack((frame_crop, mask_crop))

        return rgba

    def render(self, frame_info:FrameInfo, transform:dict={}):
        super().render(frame_info, transform)


        ############################################################
        if self.mask is None or not self.get_enabled():
            return 
        
        rgba = self.get_sprite_image(frame_info)
        self.blit_sprite(frame_info, rgba, is_transformed=self.is_transformed())
        
        