import cv2
import numpy as np

from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.utils.image import ImageUtils
from fx_api.sprites.base_sprite import BaseSprite

############################################################
# Anchor Sprite
############################################################

class AnchorSprite(BaseSprite):
    def __init__(self, sprite_manager, object_info: ObjectInfo, unique_id:int):
        super().__init__(sprite_manager, object_info, unique_id, "anchor")
     
        self.interactable = False
        self.use_keyframes = False
        self.is_segmentation_sprite = True

        #set this for the world sprite
        resolution = self.sprite_manager.fx.api.get_resolution()
        self.bbox = (0,0, resolution.x, resolution.y)
        self.set_position(resolution//2)
        self.true_size = resolution
        self.update_bbox()

    def render(self, frame_info:FrameInfo, transform:dict={}):
        super().render(frame_info, transform)
        if not self.get_enabled():
            return
        
        is_reparenting = self.sprite_manager.reparent_sprite == self and self.sprite_manager.current_modifiers.get("shift", False) == True
        thickness = 3
        if is_reparenting and self.sprite_manager.reparent_sprite == self.sprite_manager.world_sprite:
            thickness = 6
        if self.sprite_manager.selected_sprite \
            and is_reparenting \
            and self.sprite_manager.selected_sprite.parent != self:
            for i in range(4):
                
                start_point = Vector(self.bbox_corners[i].x, self.bbox_corners[i].y).round()
                end_point = Vector(self.bbox_corners[(i+1)%4].x, self.bbox_corners[(i+1)%4].y).round()
                ImageUtils.draw_dashed_line(frame_info.render_buffer, start_point, end_point, (0,255,0), thickness, 6, 8)

        if self.mask is not None and self.sprite_manager.dragging_sprite:
            x1, y1, x2, y2 = map(int, self.bbox)
            size = Vector(x2 - x1, y2 - y1)
            opacity = 0.5
            mask_crop = self.mask[y1:y2, x1:x2].astype(np.uint8) * (255 * opacity)
            frame_crop = np.full((size.y, size.x, 3), self.object_info.color, dtype=np.uint8)
            rgba = np.dstack((frame_crop, mask_crop))
            scale = self.get_scale()
            if scale.x > 0 or scale.y > 0 and size > 0:
                new_size = size * scale
                new_size = new_size.round()
                rgba = cv2.resize(rgba, (new_size.x, new_size.y), interpolation=cv2.INTER_LINEAR)
                x1 -= (new_size.x - size.x) // 2
                y1 -= (new_size.y - size.y) // 2

            ImageUtils.blend(frame_info.render_buffer, rgba, Vector(x1, y1), centered=False, blend_mode=self.blend_mode)

    
