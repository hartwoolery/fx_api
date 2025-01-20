import cv2
import numpy as np
import supervision as sv

from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.utils.image import ImageUtils
from fx_api.sprites.transforms import Transformable
from PIL import Image, ImageDraw, ImageFont
from fx_api.sprites.sprite_ui import UIButton
from fx_api.sprites.base_sprite import BaseSprite

class TextSprite(BaseSprite):
    def __init__(self, sprite_manager, font_options:dict, unique_id:int):
        
        super().__init__(sprite_manager, None, unique_id, "text")
 
        self.font_options = font_options
        # Import PIL for text rendering
        self.draw_text()

        world_size = self.sprite_manager.fx.api.get_resolution()
        self.set_position(world_size//2, frame_index=self.start_keyframe.frame_index)
    
    def get_sprite_image(self, frame_info:FrameInfo):
        if self.image is None:
            return None
        rgba = self.image
        return rgba
    
    def render(self, frame_info:FrameInfo, transform:dict={}):
        super().render(frame_info, transform)
        if self.image is None or not self.get_enabled():
            return
        
        rgba = self.get_sprite_image(frame_info)
        self.blit_sprite(frame_info, rgba, self.image is not None)
    
    def draw_text(self):
        # Create a temporary image and draw context to measure text
        temp_img = Image.new('RGBA', (1, 1), (0,0,0,0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Load font or use default
        font_path = self.font_options.get("font_path", None)
        font_size = self.font_options.get("font_size", 48)
        text = self.font_options.get("text", "Hello Banana")
    
        text_color = self.font_options.get("text_color", (255,255,255,255))
        text_bg_color = self.font_options.get("text_bg_color", (0,0,0,0))
        font = ImageFont.truetype(font_path, size=font_size) if font_path is not None else ImageFont.load_default(size=font_size)
        alignment = self.font_options.get("alignment", "center").lower()
        
        # Get text size
        text_bbox = temp_draw.textbbox((0,0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0] 
        text_height = (text_bbox[3] - text_bbox[1]) 
        
        # Add padding
        padding = 20
        canvas_width = text_width + padding * 2
        canvas_height = text_height + padding * 2
        b,g,r,a = text_bg_color
        image = Image.new("RGBA", (canvas_width, canvas_height), (r, g, b, a))
        draw = ImageDraw.Draw(image)

        text_x = padding - text_bbox[0]  # Adjust for any negative bbox offset
        text_y = padding - text_bbox[1]  # Adjust for any negative bbox offset
        b,g,r,a = text_color
        draw.text((text_x, text_y), text, fill=(r, g, b, a), font=font, align=alignment)
        self.image = np.array(image)

        #set this for the world sprite
        resolution = Vector(self.image.shape[1], self.image.shape[0])
        # if resolution.x > 512:
        #     new_scale = 512 / resolution.x
        #     resolution *= new_scale

        self.bbox = (0,0, resolution.x, resolution.y)
        
        self.true_size = resolution
        self.update_bbox()
    
    def text_changed(self, text:str):
        
        self.font_options["text"] = text
        self.name = text.replace('\n', ' ')
        self.draw_text()
    
    def text_alignment_changed(self, alignment:str):
        self.font_options["alignment"] = alignment
        self.draw_text()

    def font_changed(self, font_path:str):
        self.font_options["font_path"] = font_path
        self.draw_text()

    def font_size_changed(self, font_size:int):
        self.font_options["font_size"] = font_size
        self.draw_text()

    def text_color_changed(self, text_color:tuple[int, int, int, int]):
        self.font_options["text_color"] = text_color
        self.draw_text()

    def text_bg_color_changed(self, text_bg_color:tuple[int, int, int, int]):
        self.font_options["text_bg_color"] = text_bg_color
        self.draw_text()