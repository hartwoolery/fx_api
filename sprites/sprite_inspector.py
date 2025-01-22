import os

from fx_api.utils.easing import Easing

class SpriteInspector:
    def __init__(self, fx):
        self.fx = fx
        self.api = fx.api
        self.sprite_manager = fx.sprite_manager

    def get_font_name(self):
        font_path = self.sprite_manager.selected_sprite.font_options.get("font_path", None)
        font_family = os.path.basename(font_path) if font_path else "Choose Font..."
        return font_family

    def get_inspector(self) -> list:
        panel_items = []
        if self.fx.requires_sprites:
            sprite_types = ["Text", "Video", "Image"]
            sprite_types += [f"Object {obj.id}" for obj in self.api.get_objects()]
            panel_items = [
                {
                    "always_showing": True,
                    "type": "dropdown",
                    "options": sprite_types,
                    "label": "Add New",
                    "text": "Add",
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
                    "default": "Normal",
                    "action": self.sprite_manager.set_sprite_blend_mode,
                    "get_value": lambda: self.sprite_manager.selected_sprite.blend_mode
                },
                {
                    "show_for": "all",
                    "type": "number_input",
                    "label": "Render Order",
                    "min": -1000,
                    "max": 1000,
                    "default": 0,
                    "action": self.sprite_manager.set_render_order,
                    "get_value": lambda: self.sprite_manager.selected_sprite.render_order
                },
                {
                    "show_for": "all",
                    "type": "slider",
                    "label": "Opacity",
                    "min": 0,
                    "max": 100,
                    "default": 100,
                    "action": self.sprite_manager.set_sprite_opacity,
                    "get_value": lambda: int(self.sprite_manager.selected_sprite.get_opacity() * 100)
                },
                {
                    "show_for": "all",
                    "type": "slider",
                    "label": "Smoothing",
                    "min": 0,
                    "max": 100,
                    "default": 0,
                    "action": self.sprite_manager.set_sprite_smoothing,
                    "get_value": lambda: self.sprite_manager.selected_sprite.smoothing
                },
                {
                    "show_for": "all",
                    "type": "dropdown",
                    "label": "Easing",
                    "options": Easing.get_easing_functions(),
                    "action": self.sprite_manager.set_sprite_easing,
                    "get_value": lambda: self.sprite_manager.selected_sprite.easing
                },
                # {
                #     "type": "checkbox",
                #     "label": "Scale",
                #     "text": "Match Parent Scale",
                #     "action": self.sprite_manager.follow_scale_changed
                # },
                # {
                #     "type": "checkbox",
                #     "label": "Rotation",
                #     "text": "Match Parent Rotation",
                #     "action": self.sprite_manager.follow_rotation_changed
                # },
                {
                    "show_for": "all",
                    "type": "buttons",
                    "label": "Keyframe",
                    "text": "Add Keyframe,Clear Keyframe",
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
                    "action": self.sprite_manager.sprite_color_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.recolor
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Time Stretch",
                    "min": -400,
                    "max": 400,
                    "default": 0,
                    "action": self.sprite_manager.time_stretch_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.time_stretch
                },
                {
                    "show_for": "video",
                    "type": "checkbox",
                    "label": "Key",
                    "default": True,
                    "text": "Use Chroma Key",
                    "action": self.sprite_manager.use_chroma_key_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.use_chroma_key
                },
                {
                    "show_for": "video",
                    "type": "color_picker",
                    "label": "Chroma Key",
                    "alpha": False,
                    "action": self.sprite_manager.chroma_key_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.chroma_key_color
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Spill",
                    "min": 0,
                    "max": 100,
                    "default": 50,
                    "action": self.sprite_manager.chroma_key_spill_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.chroma_key_spill
                },
                {
                    "show_for": "video",
                    "type": "slider",
                    "label": "Choke",
                    "min": -100,
                    "max": 100,
                    "default": 10,
                    "action": self.sprite_manager.chroma_key_choke_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.chroma_key_choke
                },
                {
                    "show_for": "image,video",
                    "type": "button",
                    "label": "Media",
                    "text": "Change File...",
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
                    "action": self.sprite_manager.text_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.font_options.get("text", "Hello Banana")
                },
                {
                    "show_for": "text",
                    "type": "dropdown",
                    "label": "Alignment",
                    "options": ["Left", "Center", "Right"],
                    "action": self.sprite_manager.text_alignment_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.font_options.get("alignment", "Left")
                },
                {
                    "show_for": "text",
                    "type": "font",
                    "label": "Font",
                    "text": "Choose A Font",
                    "action": self.sprite_manager.font_changed,
                    "get_value": self.get_font_name
                },
                
                {
                    "show_for": "text",
                    "type": "number_input",
                    "label": "Font Size",
                    "min": 2,
                    "max": 100,
                    "default": 24,
                    "action": self.sprite_manager.font_size_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.font_options.get("size", 24)
                },
                {
                    "show_for": "text",
                    "type": "color_picker",
                    "label": "Text Color",
                    "alpha": True,
                    "action": self.sprite_manager.text_color_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.font_options.get("text_color", (255,255,255,255))
                },
                {
                    "show_for": "text",
                    "type": "color_picker",
                    "label": "BG Color",
                    "alpha": True,
                    "action": self.sprite_manager.text_bg_color_changed,
                    "get_value": lambda: self.sprite_manager.selected_sprite.font_options.get("text_bg_color", (0,0,0,0))
                },
            ]
            return panel_items