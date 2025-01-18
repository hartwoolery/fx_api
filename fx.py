import os
import numpy as np
from fx_api.interface import FXAPI, FrameInfo
from fx_api.sprites.sprite_manager import SpriteManager
from fx_api.utils.anchor_manager import AnchorManager
from fx_api.utils.vector import Vector
from fx_api.utils.easing import Easing

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
        self.anchor_manager = AnchorManager(self)
        self.is_ready = False
        self.unique_id = self.fx_name + "::" + self.video_path
      
        self.setup()
        self.sprite_manager = SpriteManager(self)

        # force requires_mask if inpainting is required
        if self.requires_inpainting:
            self.requires_mask = True

    def on_ready(self, is_ready:bool=True):
        self.is_ready = is_ready
        if is_ready and self.requires_sprites:
            self.sprite_manager.add_sprites_for_objects(self.api.get_objects())

    def get_name(self):
        return self.__class__.__name__

    def setup(self):
        pass

    def get_fx_panel(self) -> list:
        panel_items = []
        if self.requires_sprites:
            sprite_types = ["Video", "Image", "Text"]
            sprite_types += [f"Object {obj.id}" for obj in self.api.get_objects()]
            panel_items = [
                {
                    "always_showing": True,
                    "type": "dropdown",
                    "options": sprite_types,
                    "label": "Add New",
                    "button_label": "Add",
                    "action": self.sprite_manager.add_sprite_of_type
                },
                {
                    "type": "divider"
                },
                {
                    "show_for": "all",
                    "type": "dropdown",
                    "label": "Blend Mode",
                    "options": ["Normal", "Additive", "Subtractive", "Multiply", "Screen", "Overlay", "Darken", "Lighten", "Color Dodge", "Color Burn", "Hard Light", "Soft Light", "Difference", "Exclusion", "Hue", "Saturation", "Color", "Luminosity"],
                    "change_action": self.sprite_manager.set_sprite_blend_mode
                },
                {
                    "show_for": "all",
                    "type": "number_input",
                    "label": "Render Order",
                    "min": -1000,
                    "max": 1000,
                    "default": 0,
                    "action": self.sprite_manager.set_render_order
                },
                {
                    "show_for": "all",
                    "type": "slider",
                    "label": "Opacity",
                    "min": 0,
                    "max": 100,
                    "default": 100,
                    "action": self.sprite_manager.set_sprite_opacity
                },
                {
                    "show_for": "all",
                    "type": "dropdown",
                    "label": "Easing",
                    "options": Easing.get_easing_functions(),
                    "change_action": self.sprite_manager.set_sprite_easing
                },
                # {
                #     "type": "checkbox",
                #     "label": "Scale",
                #     "button_label": "Match Parent Scale",
                #     "action": self.sprite_manager.follow_scale_changed
                # },
                # {
                #     "type": "checkbox",
                #     "label": "Rotation",
                #     "button_label": "Match Parent Rotation",
                #     "action": self.sprite_manager.follow_rotation_changed
                # },
                {
                    "show_for": "all",
                    "type": "buttons",
                    "label": "Keyframe",
                    "button_label": "Add Keyframe,Clear Keyframe",
                    "action": [self.sprite_manager.add_keyframe, self.sprite_manager.reset_current_keyframe]
                },
                {
                    "show_for": "image,video",
                    "type": "divider"
                },
                {
                    "show_for": "cutout,image,video",
                    "type": "color_picker",
                    "label": "Recolor",
                    "action": self.sprite_manager.sprite_color_changed
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Time Stretch",
                    "min": -400,
                    "max": 400,
                    "default": 0,
                    "action": self.sprite_manager.time_stretch_changed
                },
                {
                    "show_for": "video",
                    "type": "checkbox",
                    "label": "Key",
                    "default": True,
                    "button_label": "Use Chroma Key",
                    "action": self.sprite_manager.use_chroma_key_changed
                },
                {
                    "show_for": "video",
                    "type": "color_picker",
                    "label": "Chroma Key",
                    "alpha": False,
                    "action": self.sprite_manager.chroma_key_changed
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Spill",
                    "min": 0,
                    "max": 100,
                    "default": 50,
                    "action": self.sprite_manager.chroma_key_spill_changed
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Choke",
                    "min": -100,
                    "max": 100,
                    "default": 10,
                    "action": self.sprite_manager.chroma_key_choke_changed
                },
                {
                    "show_for": "image,video",
                    "type": "button",
                    "label": "Media",
                    "button_label": "Change File...",
                    "action": self.sprite_manager.change_sprite_path
                },
                {
                    "show_for": "text",
                    "type": "divider"
                },
                {
                    "show_for": "text",
                    "type": "text_input",
                    "label": "Text",
                    "default": "Hello Banana",
                    "action": self.sprite_manager.text_changed
                },
                {
                    "show_for": "text",
                    "type": "dropdown",
                    "label": "Alignment",
                    "options": ["Left", "Center", "Right"],
                    "change_action": self.sprite_manager.text_alignment_changed

                },
                {
                    "show_for": "text",
                    "type": "font",
                    "label": "Font",
                    "button_label": "Choose A Font",
                    "action": self.sprite_manager.font_changed
                },
                
                {
                    "show_for": "text",
                    "type": "number_input",
                    "label": "Font Size",
                    "min": 2,
                    "max": 100,
                    "default": 24,
                    "action": self.sprite_manager.font_size_changed
                },
                {
                    "show_for": "text",
                    "type": "color_picker",
                    "label": "Text Color",
                    "action": self.sprite_manager.text_color_changed
                },
                {
                    "show_for": "text",
                    "type": "color_picker",
                    "label": "BG Color",
                    "alpha": True,
                    "action": self.sprite_manager.text_bg_color_changed
                },
            ]
            return panel_items

    def get_resource(self, path:str):
        return os.path.join(self.fx_path, path)

    def prepare_render_frame(self, frame_info: FrameInfo, render_ui:bool=False):
        # prepare background
        if self.requires_inpainting:
            frame_info.render_buffer = self.api.get_inpainting(frame_info)
        else:
            frame_info.render_buffer = frame_info.frame.copy()


        # update sprite positions
        if self.requires_sprites:
            pose_anchors = {}
            if self.requires_pose:
                pose_anchors = self.api.get_pose(frame_info)
            detections = self.api.get_masks(frame_info.index)
            self.sprite_manager.update(frame_info, detections, pose_anchors)
            


        self.render_background(frame_info)
        self.render_frame(frame_info)
        self.post_render(frame_info, render_ui)
        self.anchor_manager.render(frame_info, render_ui)

    def render_background(self, frame_info: FrameInfo):
        self.sprite_manager.render_anchors(frame_info)

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
