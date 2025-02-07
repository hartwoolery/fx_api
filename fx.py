import os
import numpy as np
import cv2
from fx_api.interface import FXAPI, FrameInfo
from fx_api.sprites.sprite_manager import SpriteManager
from fx_api.utils.anchor_manager import AnchorManager
from fx_api.utils.vector import Vector
from fx_api.sprites.sprite_inspector import SpriteInspector

class FX:
    
    def __init__(self, fx_api: FXAPI, fx_name: str="Custom Effect", fx_path: str="", video_path: str=""):
        super().__init__()
        self.api = fx_api
        self.requires_pose = False
        self.requires_mask = False
        self.requires_inpainting = False
        self.requires_sprites = True
        self.fx_name = fx_name
        self.fx_path = fx_path
        self.video_path = video_path
        self.sprite_manager = None
        self.meta_data = {}
        self.default_meta_data = {}
        self.anchor_manager = AnchorManager(self)
        self.is_ready = False
        self.unique_id = self.fx_name + "::" + self.video_path

        self.setup()
        self.sprite_manager = SpriteManager(self)
        self.sprite_inspector = SpriteInspector(self)

        # force requires_mask if inpainting is required
        if self.requires_inpainting:
            self.requires_mask = True

    # initial sprites created
    def on_ready(self):
        self.is_ready = True
        if self.requires_sprites:
            self.sprite_manager.add_sprites_for_objects(self.api.get_objects())
            self.update_sprite_positions(self.api.get_current_frame())

    def get_name(self):
        return self.__class__.__name__

    def setup(self):
        pass

    def get_sprite_inspector(self):
        custom_items = self.get_custom_inspector()
        if len(custom_items) > 0:
            custom_items.append({"type": "divider"})
        default_items = self.sprite_inspector.get_inspector()
        
        first_item = default_items[:1]
        second_item = default_items[1:]
        return first_item + custom_items + second_item
    
    def get_custom_inspector(self):
        return []
    
    def set_meta(self, key:str, value:any):
        self.meta_data["fx." + key] = value
        self.api.should_refresh_frame = True
        
    def get_meta(self, key:str, default:any=None):
        return self.meta_data.get("fx." + key, default)

    def get_resource(self, path:str):
        return os.path.join(self.fx_path, path)

    def get_image_resource(self, path:str):
        path = self.get_resource(path)
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return None
    
    def update_sprite_positions(self, frame_info: FrameInfo):
        # update sprite positions
        if self.requires_sprites:
            pose_anchors = {}
            # if self.requires_pose:
            #     pose_anchors = self.api.get_pose(frame_info)
            detections = self.api.get_masks(frame_info.index)
            self.sprite_manager.update(frame_info, detections, pose_anchors)

    

    def process_frame(self, frame_info: FrameInfo, render_ui:bool=False):

        self.update_sprite_positions(frame_info)
        self.render_background(frame_info)
        self.render_frame(frame_info)
        self.post_render(frame_info, render_ui)
        self.anchor_manager.render(frame_info, render_ui)

    def current_sprite(self):
        return self.sprite_manager.selected_sprite

    def refresh_frame(self):
        self.api.should_refresh_frame = True

    def render_background(self, frame_info: FrameInfo):
        if self.requires_inpainting:
            frame_info.render_buffer = self.api.get_inpainting(frame_info)
        else:
            frame_info.render_buffer = frame_info.frame.copy()
        #self.sprite_manager.render_anchors(frame_info)

    def render_frame(self, frame_info: FrameInfo):
        self.sprite_manager.render_sprites(frame_info)
    
    def post_render(self, frame_info: FrameInfo, render_ui:bool=False):
        self.sprite_manager.post_render(frame_info, render_ui)
        
    def get_widget(self) -> dict:
        return {}
    
    def handle_event(self, func) -> bool:
        should_bubble = True
        if self.requires_sprites:
            should_bubble = func()
            if not should_bubble:
                self.api.should_refresh_frame = True
        return should_bubble
    
    def on_mouse_down(self, coord: tuple[int, int], modifiers: dict) -> bool:
        return self.handle_event(lambda: self.sprite_manager.on_mouse_down(Vector(coord), modifiers))
    
    def on_mouse_up(self, coord: tuple[ int, int], modifiers: dict) -> bool:
        return self.handle_event(lambda: self.sprite_manager.on_mouse_up(Vector(coord), modifiers))
        

    def on_mouse_move(self, coord: tuple[int, int], modifiers: dict) -> bool:
        return self.handle_event(lambda: self.sprite_manager.on_mouse_move(Vector(coord), modifiers))

    def on_key_press(self, key: str, modifiers: dict) -> bool:
        return self.handle_event(lambda: self.sprite_manager.on_key_press(key, modifiers))
    
    def on_key_release(self, key: str, modifiers: dict) -> bool:
        return self.handle_event(lambda: self.sprite_manager.on_key_release(key, modifiers))
