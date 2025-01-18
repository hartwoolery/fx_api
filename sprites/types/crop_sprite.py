import cv2
import numpy as np
from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.sprites.base_sprite import BaseSprite
from fx_api.utils.image import ImageUtils

############################################################
# Crop Sprite
############################################################

class CropSprite(BaseSprite):
    def __init__(self, sprite_manager, object_info:ObjectInfo, unique_id:int):
        super().__init__(sprite_manager, object_info, unique_id, "crop")
        world_size = self.sprite_manager.fx.api.get_resolution()
        self.bbox = [0,0, world_size.x, world_size.y]
        self.true_size = world_size
        self.update_bbox()
        #self.set_position(world_size//2, frame_index=self.start_keyframe.frame_index)

    def modify_crop(self, button, coord:Vector):
        if button.type == "scale_x":
            # 5 and 7
            if button == self.transform_buttons[7]:
                self.bbox[0] = coord.x
            elif button == self.transform_buttons[5]:
                self.bbox[2] = coord.x
        elif button.type == "scale_y":
            if button == self.transform_buttons[4]:
                self.bbox[1] = coord.y
            elif button == self.transform_buttons[6]:
                self.bbox[3] = coord.y
         
        else:
            # 1, 2, 3, 4
            if button == self.transform_buttons[0]:
                self.bbox[0] = coord.x
                self.bbox[1] = coord.y
            elif button == self.transform_buttons[1]:
                self.bbox[2] = coord.x
                self.bbox[1] = coord.y
            elif button == self.transform_buttons[2]:
                self.bbox[2] = coord.x
                self.bbox[3] = coord.y
            elif button == self.transform_buttons[3]:
                self.bbox[0] = coord.x
                self.bbox[3] = coord.y

        self.validate_bbox()
        self.update_bbox()

    def validate_bbox(self):
        # Get frame dimensions from sprite manager
        world_size = self.sprite_manager.fx.api.get_resolution()
        frame_width, frame_height = world_size.x, world_size.y

        # Ensure coordinates are within frame bounds
        self.bbox[0] = max(0, min(self.bbox[0], frame_width))
        self.bbox[1] = max(0, min(self.bbox[1], frame_height))
        self.bbox[2] = max(0, min(self.bbox[2], frame_width))
        self.bbox[3] = max(0, min(self.bbox[3], frame_height))

        # Ensure bottom-right corner is greater than top-left
        if self.bbox[2] < self.bbox[0]:
            self.bbox[0], self.bbox[2] = self.bbox[2], self.bbox[0]
        if self.bbox[3] < self.bbox[1]:
            self.bbox[1], self.bbox[3] = self.bbox[3], self.bbox[1]

        # Enforce minimum size of 100x100
        min_size = 100
        if self.bbox[2] - self.bbox[0] < min_size:
            # If box is too small, expand right side unless at frame edge
            if self.bbox[2] + min_size <= frame_width:
                self.bbox[2] = self.bbox[0] + min_size
            else:
                self.bbox[0] = self.bbox[2] - min_size

        if self.bbox[3] - self.bbox[1] < min_size:
            # If box is too small, expand bottom unless at frame edge
            if self.bbox[3] + min_size <= frame_height:
                self.bbox[3] = self.bbox[1] + min_size
            else:
                self.bbox[1] = self.bbox[3] - min_size
    

        
        



    def render(self, frame_info:FrameInfo, transform:dict={}):
        # Get the frame dimensions
        frame_height, frame_width = frame_info.render_buffer.shape[:2]
        
        # Create a black mask with 50% opacity
        mask = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
        mask[:,:,3] = 127  # 50% opacity
        
        # Get bbox coordinates
        x1, y1, x2, y2 = map(int, self.bbox)
        
        # Clear the mask in the bbox region
        mask[y1:y2, x1:x2] = 0


        ImageUtils.blend(frame_info.render_buffer, mask, Vector(0,0), centered=False, blend_mode="normal")

        
        # Draw "CROP" text for each button
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text = "CROP"
        text_x = None
        text_y = None
        # Get text size once since it's the same for all
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        button = self.sprite_manager.current_button
        if button is self.transform_buttons[4] or button is self.transform_buttons[0] or button is self.transform_buttons[1]:
             # Draw text for button 4 (above)
            pos = self.transform_buttons[4].location
            text_x = int(pos.x - text_width/2)
            text_y = int(pos.y - 20)

        elif button is self.transform_buttons[5]:
            # Draw text for button 5 (right) 
            pos = self.transform_buttons[5].location
            text_x = int(pos.x + 20)
            text_y = int(pos.y + text_height/2)
        
        elif button is self.transform_buttons[6]  or button is self.transform_buttons[2] or button is self.transform_buttons[3]:
             # Draw text for button 6 (below)
            pos = self.transform_buttons[6].location
            text_x = int(pos.x - text_width/2)
            text_y = int(pos.y + 20 + text_height)

        elif button is self.transform_buttons[7]:
             # Draw text for button 7 (left)
            pos = self.transform_buttons[7].location
            text_x = int(pos.x - text_width - 20)
            text_y = int(pos.y + text_height/2)

        if text_x is not None and text_y is not None:
            cv2.putText(frame_info.render_buffer, text, (text_x, text_y), font, font_scale, (255,255,255), thickness)
            
       
       
        
       
        
        
