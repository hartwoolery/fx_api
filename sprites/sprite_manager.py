import os
import cv2
import copy
import numpy as np
import supervision as sv
import platform

from fx_api.utils.anchor_manager import AnchorManager, ANCHOR_ID
from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.sprites.types.text_sprite import TextSprite
from fx_api.sprites.types.image_sprite import ImageSprite
from fx_api.sprites.types.anchor_sprite import AnchorSprite
from fx_api.sprites.types.video_sprite import VideoSprite
from fx_api.sprites.types.segmentation_sprite import SegmentationSprite
from fx_api.sprites.types.crop_sprite import CropSprite
from fx_api.utils.vector import Vector
from fx_api.utils.history_manager import HistoryManager, HistoryState, HistoryType
from fx_api.sprites.transforms import KeyFrame

class SpriteManager:
    def __init__(self, fx):
        self.fx = fx
        self.anchor_manager = fx.anchor_manager
        self.history_manager = HistoryManager(self)
        self.current_unique_id = 1
        self.current_frame_index = 0
        self.current_button = None
        self.mouse_start_pos = (0,0)
        self.transformed_ids = {}
        self.sprites = []
        self.anchor_sprites = []
        self.reparent_sprite = None
        self.dragging_sprite = False
        self.current_modifiers = {}

        object_info = ObjectInfo(start_frame = 0, end_frame = self.fx.api.get_total_frames() - 1)
        self.world_sprite = AnchorSprite(self, object_info, -1)
        self.crop_sprite = CropSprite(self, object_info, -2)
        self.selected_sprite = self.crop_sprite
        self.world_sprite.name = "scene"
        self.starting_matrix = None
        self.starting_scale = Vector(1,1)
        self.starting_offset = Vector(0,0)
        self.starting_anchor_point = Vector(0,0)
        self.has_new_keyframes = False
        self.added_new_sprite = False
        self.starting_transform = {}
        

    ############################################################
    # Inspector Actions
    ############################################################

    def clone_sprite(self):
        if self.selected_sprite:
            new_sprite = self.add_sprite_of_type(self.selected_sprite.type, self.selected_sprite)
            
    def delete_sprite(self, delete_sprite=None):
        sprite = delete_sprite or self.selected_sprite
        if sprite:
            for child in sprite.children:
                child.set_parent(sprite.parent)
            self.sprites.remove(sprite)
            parent = sprite.parent
            parent.children.remove(sprite)

            deleted_sprite = sprite
            self.history_manager.add_history(HistoryState(state={
                "type": HistoryType.DELETE_SPRITE,
                "sprite": deleted_sprite
            }))

            if delete_sprite is None:
                self.select_sprite(None)
            if len(parent.children) > 0:
                self.select_sprite(parent.children[len(parent.children)-1])
            self.fx.api.should_refresh_timeline = True
            self.fx.api.should_refresh_tree = True

    def set_sprite_enabled(self, enabled: bool):
        self.sprite_enabled = enabled

    def reset_current_keyframe(self):
        if self.selected_sprite and self.selected_sprite.use_keyframes:
            for keyframe in self.selected_sprite.keyframes:
                if keyframe.frame_index == self.current_frame_index:
                    self.remove_keyframe(keyframe, self.selected_sprite)
                    break

    def remove_keyframe(self, keyframe:KeyFrame, sprite):
        if sprite and sprite.use_keyframes:
            if keyframe.is_bookend:
                keyframe.transform = copy.deepcopy(KeyFrame.default_transform()) if keyframe is sprite.start_keyframe else {}
                self.history_manager.add_history(HistoryState(state={
                    "type": HistoryType.MODIFY_TRANSFORM,
                    "sprite": sprite,
                    "keyframe": keyframe
                }))      
            else:
                sprite.keyframes.remove(keyframe)
                self.history_manager.add_history(HistoryState(state={
                    "type": HistoryType.DELETE_KEYFRAME,
                    "sprite": sprite,
                    "keyframe": keyframe
                }))      
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_timeline = True

    def reset_all_keyframes(self):
        if self.fx.api.prompt_user("Are you sure you want to reset all keyframes? This will delete all keyframes for all sprites and cannot be undone."):
            for sprite in self.sprites:
                sprite.keyframes = [sprite.start_keyframe, sprite.end_keyframe]
            
            self.history_manager.clear_history()
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_timeline = True

    def select_sprite(self, sprite):
        self.selected_sprite = sprite if sprite is not None else self.crop_sprite

        self.fx.api.should_refresh_inspector = True

    def select_next_sprite(self, backwards:bool=False):
        if self.selected_sprite is not self.crop_sprite:
            current_index = self.sprites.index(self.selected_sprite)
            if backwards:
                new_index = (current_index - 1) % len(self.sprites)
            else:
                new_index = (current_index + 1) % len(self.sprites)
            self.select_sprite(self.sprites[new_index])
        else:
            self.select_sprite(self.sprites[0])

    def set_sprite_blend_mode(self, blend_mode: str):
        if self.selected_sprite:
            self.selected_sprite.blend_mode = blend_mode.lower()
            self.fx.api.should_refresh_frame = True

    def set_render_order(self, render_order:float):
        if self.selected_sprite:
            self.selected_sprite.set_render_order(render_order)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_timeline = True

    def set_sprite_easing(self, easing: str):
        if self.selected_sprite:
            self.selected_sprite.easing = easing
            self.fx.api.should_refresh_frame = True

    def set_sprite_smoothing(self, smoothing: int, finished:bool=False):
        if self.selected_sprite:
            self.selected_sprite.set_smoothing(smoothing)

    def set_sprite_opacity(self, opacity: int, finished:bool=False):
        if self.selected_sprite:
            new_opacity = opacity/100.0
            self.selected_sprite.temp_opacity = None if finished else new_opacity
            if self.selected_sprite.get_opacity() != new_opacity and finished:
                starting_transform = copy.deepcopy(self.selected_sprite.local_transform)
                is_new_keyframe = self.selected_sprite.keyframe_for_index(self.current_frame_index) is None
                self.selected_sprite.set_opacity(new_opacity)
                
                if is_new_keyframe:
                    self.history_manager.add_history(HistoryState(state={
                        "type": HistoryType.ADD_KEYFRAME,
                        "sprite": self.selected_sprite,
                        "keyframe": self.selected_sprite.keyframe_for_index(self.current_frame_index)
                    }))
                else:
                    self.history_manager.add_history(HistoryState(state={
                        "type": HistoryType.MODIFY_TRANSFORM,
                        "sprite": self.selected_sprite,
                        "start_transform": starting_transform,
                        "end_transform": copy.deepcopy(self.selected_sprite.local_transform),
                        "frame_index": self.current_frame_index
                    }))

                self.fx.api.should_refresh_timeline = True
            self.fx.api.should_refresh_frame = True

    def change_sprite_path(self):
        if self.selected_sprite:
            if self.selected_sprite.type == "image":
                path = self.fx.api.open_file_dialog("Open Image", "Image Files (*.png *.jpg *.jpeg);;All Files (*.*)")
            elif self.selected_sprite.type == "video":
                path = self.fx.api.open_file_dialog("Open Video", "Video Files (*.mp4);;All Files (*.*)")
            if path is None:
                return
            self.selected_sprite.change_sprite_path(path)
            self.selected_sprite.name = os.path.basename(path)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_tree = True
            self.fx.api.should_refresh_timeline = True

    def font_changed(self, font_path: str):
        if self.selected_sprite:
            self.selected_sprite.font_changed(font_path)
            self.fx.api.should_refresh_frame = True

    def font_size_changed(self, font_size: int):
        if self.selected_sprite:
            self.selected_sprite.font_size_changed(font_size)
            self.fx.api.should_refresh_frame = True

    def text_changed(self, text: str):
        
        if self.selected_sprite and self.selected_sprite.type == "text":
            prev_text = self.selected_sprite.font_options.get("text", None)
            if prev_text != text:
                self.selected_sprite.text_changed(text)
                self.fx.api.should_refresh_frame = True
                self.fx.api.should_refresh_tree = True
                self.fx.api.should_refresh_timeline = True
    
    def text_alignment_changed(self, alignment: str):
        if self.selected_sprite:
            self.selected_sprite.text_alignment_changed(alignment)
            self.fx.api.should_refresh_frame = True

    def text_color_changed(self, color: tuple[int,int,int,int]):
        if self.selected_sprite:
            self.selected_sprite.text_color_changed(color)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True
    
    def text_bg_color_changed(self, color: tuple[int,int,int,int]):
        if self.selected_sprite:
            self.selected_sprite.text_bg_color_changed(color)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def time_stretch_changed(self, time_stretch: float, finished: bool):
        if self.selected_sprite:
            self.selected_sprite.time_stretch = time_stretch
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def use_chroma_key_changed(self, use_chroma_key: bool):
        if self.selected_sprite:
            self.selected_sprite.use_chroma_key = use_chroma_key
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def chroma_key_changed(self, color: tuple[int,int,int]):
        if self.selected_sprite:
            self.selected_sprite.chroma_key_changed(color)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def chroma_key_spill_changed(self, spill: float, finished: bool):
        if self.selected_sprite:
            self.selected_sprite.chroma_key_spill = spill
            self.selected_sprite.reset_spill_masks()
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True
    
    def chroma_key_choke_changed(self, choke: float, finished: bool):
        if self.selected_sprite:
            self.selected_sprite.chroma_key_choke = choke
            self.selected_sprite.reset_choke_masks()
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def sprite_color_changed(self, color: tuple[int,int,int]):
        if self.selected_sprite:
            self.selected_sprite.recolor_changed(color)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_inspector = True

    def add_keyframe(self):
        if self.selected_sprite:
            self.selected_sprite.get_or_add_keyframe(self.current_frame_index)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_timeline = True

    def follow_scale_changed(self, follow_scale: bool):
        if self.selected_sprite and follow_scale:
            obj_id = self.selected_sprite.parent.object_info.id
            self.fx.api.estimate_transforms(obj_id, self.on_transform_estimates_ready)
        self.selected_sprite.follow_scale = follow_scale
            

    def follow_rotation_changed(self, follow_rotation: bool):
        if self.selected_sprite and follow_rotation:
            obj_id = self.selected_sprite.parent.object_info.id
            self.fx.api.estimate_transforms(obj_id, self.on_transform_estimates_ready)
        self.selected_sprite.follow_rotation = follow_rotation

    def on_transform_estimates_ready(self, transforms:list):
        if self.selected_sprite:
            self.selected_sprite.transform_estimates = transforms
            self.fx.api.should_refresh_frame = True

    ############################################################
    # Mouse and Keyboard Actions
    ############################################################

    def on_key_press(self, key: str, modifiers: dict) -> bool:
        self.current_modifiers = modifiers
        should_bubble = True
        if self.selected_sprite and self.reparent_sprite is not None:
            should_bubble = False
        command_key = modifiers.get("ctrl", False) == True
        shift_key = modifiers.get("shift", False) == True
        ctrl_key = "Cmd" if platform.system() == "Darwin" else "Ctrl"
        
        if key == "z" and command_key == True:
            if shift_key == True:
                self.history_manager.redo()
                self.fx.api.update_hint_label("Redo " + str(self.history_manager.current_state).lower())
            else:
                self.history_manager.undo()
                self.fx.api.update_hint_label("Undo " + str(self.history_manager.current_state).lower() + " - hold " + ctrl_key +"+Shift+Z to Redo")
             #redraw timeline
            self.fx.api.should_refresh_timeline = True
            should_bubble = False
        elif modifiers.get("delete", False) == True:
            self.delete_sprite()
            should_bubble = False
        
        return should_bubble

    def on_key_release(self, key: int, modifiers: dict) -> bool:
        self.current_modifiers = {}
        if self.selected_sprite and self.reparent_sprite is not None:
            return False
        return True

    def on_mouse_down(self, coord: tuple[int, int], modifiers: dict) -> bool:
        self.current_modifiers = modifiers
        should_bubble = True
        self.mouse_start_pos = coord
        spr = self.selected_sprite
        if spr is not None and not spr.locked:

            self.starting_transform = copy.deepcopy(spr.local_transform)
            # Find closest button to mouse click
            min_dist = float('inf')
            closest_button = None
            
            for button in self.selected_sprite.transform_buttons:
                # Calculate distance from mouse to button center
                delta = coord - button.location
                dist = delta.length()
                
                # Update if this is the closest button so far
                if dist < min_dist and dist < button.radius * 1.5:
                    min_dist = dist
                    closest_button = button
            
            if closest_button:
                self.current_button = closest_button
                if "scale" in self.current_button.type:
                    self.starting_matrix = spr.global_transform_matrix.copy()
                    self.starting_scale = spr.get_scale(local=True)
                    self.starting_anchor_point = spr.local_to_global(spr.anchor_point)
                    
            # else:
            #     #if mousing down on a sprite
            #     closest_sprite = self.get_closest_sprite(coord, exlude_selected_sprite=False, include_anchor_sprites=False)
            #     self.select_sprite(closest_sprite)
                    
            
            should_bubble = False

        if self.current_button is None:
            # did not hit a button or select a sprite
            # Find closest sprite to click position
            closest_sprite = self.get_closest_sprite(coord)
            should_bubble = self.selected_sprite == closest_sprite
            if closest_sprite:
                
                self.select_sprite(closest_sprite)
                spr = self.selected_sprite
                #self.dragging_sprite = True

                self.starting_matrix = spr.global_transform_matrix.copy()
                
                self.starting_offset = (spr.get_position() - coord)
                
            else:
                self.select_sprite(None)    #if spr.parent != self.world_sprite:

        return should_bubble
    
    def get_closest_sprite(self, coord: Vector, exlude_selected_sprite: bool=False, include_anchor_sprites: bool=False):
        min_dist = float('inf')
        closest_sprite = None
        all_sprites = self.sprites if not include_anchor_sprites else self.sprites + self.anchor_sprites 
       
        for sprite in all_sprites:
            if exlude_selected_sprite and sprite == self.selected_sprite:
                continue
            if sprite.locked:
                continue
            if sprite.use_keyframes and (self.current_frame_index < sprite.start_keyframe.frame_index or self.current_frame_index > sprite.end_keyframe.frame_index):
                continue
            # Calculate distance from mouse to sprite center
            if sprite.is_point_in_sprite(coord, buffer=0):
                delta = coord - sprite.bbox_center
                dist = delta.length()
                render_order_greater = (closest_sprite and sprite.render_order > closest_sprite.render_order) or closest_sprite is None
                render_order_greater_equal = (closest_sprite and sprite.render_order >= closest_sprite.render_order) or closest_sprite is None
                if render_order_greater or (dist < min_dist and render_order_greater_equal):
                    # Update if this is the closest sprite so far
                    min_dist = dist
                    closest_sprite = sprite
                    
        if closest_sprite is None and include_anchor_sprites:
            closest_sprite = self.world_sprite
        return closest_sprite



    def on_mouse_move(self, coord: Vector, modifiers: dict) -> bool:
        self.current_modifiers = modifiers
        translation_delta = coord - self.mouse_start_pos
        spr = self.selected_sprite
        self.reparent_sprite = None
        dragging = modifiers.get("dragging", False)
        if dragging:
            
            if self.current_button and self.selected_sprite:
                # special case for crop sprite
                if self.current_button and "scale" in self.current_button.type and self.selected_sprite.type == "crop":
                    self.selected_sprite.modify_crop(self.current_button, coord)
                

                elif self.current_button.type == "anchor":
                    spr.temp_anchor_point = spr.global_to_local(coord)

                        
                elif "scale" in self.current_button.type:
                    # Get current scale and button position in local space
                    current_scale = spr.get_scale(local=True)
                    button_local = spr.global_to_local(self.current_button.location, self.starting_matrix)
                    mouse_local = spr.global_to_local(coord, self.starting_matrix)
                    mouse_start_local = spr.global_to_local(self.mouse_start_pos, self.starting_matrix)

                    #d1 = abs(mouse_local - mouse_start_local)
                    d2 = abs(spr.anchor_point - mouse_start_local) 
                    d3 = abs(spr.anchor_point - mouse_local) 

                    
                    if self.current_button.type == "scale":
                        new_scale = Vector(d3.x / max(d2.x, 0.1) * self.starting_scale.x, 
                                           d3.y / max(d2.y, 0.1) * self.starting_scale.y)
                        
                        if modifiers.get("shift", False) == True:
                            d = max(abs(new_scale.x), abs(new_scale.y))
                            new_scale = Vector(d, d) * new_scale.sign()

                    # Calculate scale factor based on distance from anchor
                    elif self.current_button.type == "scale_x":

                        new_scale = Vector(d3.x / max(d2.x, 0.1) * self.starting_scale.x, 
                                           current_scale.y) 
                        
                    
                    elif self.current_button.type == "scale_y":
                        # For top/bottom buttons, use y distance
                        new_scale = Vector(current_scale.x, 
                                           d3.y / max(d2.y, 0.1) * self.starting_scale.y) 
                    
                    # Clamp scale to reasonable limits while preserving sign
                    # sign_x = 1 if new_scale.x > 0 else -1
                    # sign_y = 1 if new_scale.y > 0 else -1

                    # Check if signs have changed from starting scale
                    starting_sign = self.starting_scale.sign()

                    
                    sign_mouse = (mouse_local - spr.anchor_point).sign()
                    sign_mouse_start = (mouse_start_local - spr.anchor_point).sign()


                    sign_x = 1 if sign_mouse.x == sign_mouse_start.x else -1
                    sign_y = 1 if sign_mouse.y == sign_mouse_start.y else -1

                    #spr.test_negative_scale()
                    
                    

                    # flip_x = -1 * starting_sign_x if sign_x != starting_sign_x else starting_sign_x     
                    # flip_y = -1 * starting_sign_y if sign_y != starting_sign_y  else starting_sign_y

                    if modifiers.get("shift", False) == True:
                        sign_x = 1
                        sign_y = 1

                    flip_x = starting_sign.x * sign_x
                    flip_y = starting_sign.y * sign_y


                    if self.current_button.type == "scale_x":
                        flip_y = starting_sign.y
                    elif self.current_button.type == "scale_y":
                        flip_x = starting_sign.x


                    new_scale = abs(new_scale).clamp(0.1, 10.0)
                    new_scale = Vector(flip_x * new_scale.x, flip_y * new_scale.y)

                    
                    self.selected_sprite.set_scale(new_scale)
                    #self.mouse_start_pos = coord
                
                    
                elif self.current_button.type == "rotation":
                    # Get current rotation
                    current_rotation = self.selected_sprite.get_rotation()
                    
                    
                    # Calculate the angle from the current_button to the anchor point
                    button_coord = self.current_button.location
                    anchor_point = spr.local_to_global(spr.anchor_point)
                    delta = button_coord - anchor_point
                    
                    angle_button_to_anchor = delta.angle()
                    
                    # Calculate the angle from the mouse cursor to the anchor point
                    delta2 = coord - anchor_point
                    angle_mouse_to_anchor = delta2.angle()
                    
                    # Calculate the angle delta
                    angle_delta = angle_mouse_to_anchor - angle_button_to_anchor
                    
                    # Calculate the new rotation
                    new_rotation = current_rotation - angle_delta

                    scale = spr.get_scale()
                    if scale.x * scale.y < 0:  # If exactly one scale component is negative
                        new_rotation = (new_rotation + 180) % 360

                    
                    self.selected_sprite.set_rotation(new_rotation, local=False)
                    self.mouse_start_pos = coord
                
                
            elif self.current_button is None and self.selected_sprite:
                # we are dragging the sprite
                if self.selected_sprite.type == "crop":
                    return True

                closest_sprite = self.get_closest_sprite(coord, exlude_selected_sprite=True, include_anchor_sprites=True)
                if closest_sprite:
                    # Don't allow reparenting to a child sprite
                    current = closest_sprite
                    is_child = False
                    while current is not None:
                        if current in self.selected_sprite.children:
                            is_child = True
                            break
                        current = current.parent
                    
                    if not is_child and closest_sprite != spr.parent:
                        self.reparent_sprite = closest_sprite
                    else:
                        self.reparent_sprite = None
                
                # if spr.parent is not None:
                #     translation_delta = translation_delta.rotate(-spr.get_rotation(local=True))*spr.get_scale(local=True)
                # else:

                # scale = spr.get_scale(local=True)
                # translation_delta = translation_delta.rotate(-spr.get_rotation(local=True))*abs(scale)
                # local_trans = spr.global_to_local(spr.get_position() + translation_delta )  
                # if scale.x < 0:
                #     local_trans *= Vector(-1,1)
                # if scale.y < 0:
                #     local_trans *= Vector(1,-1)

                spr.set_position( coord + self.starting_offset )
                spr.update_transform()

                
                #self.mouse_start_pos = coord
                self.dragging_sprite = True

                
        if dragging and (spr is not None or self.current_button is not None):
            return False
        
        return True
    
    def should_draw_anchors(self):
        return True

    def on_mouse_up(self, coord: tuple[int, int], modifiers: dict) -> bool:
        self.current_modifiers = {}
        should_bubble = True
        spr = self.selected_sprite

       

        if self.dragging_sprite:
            if modifiers.get("shift", False) == True and self.reparent_sprite is not None:
                spr.set_parent(self.reparent_sprite, change_position=True)
                self.fx.api.should_refresh_tree = True
            should_bubble = False
            
           
        self.reparent_sprite = None
        self.dragging_sprite = False
        
        # handle button release
        if self.current_button is not None and spr is not None:


            if "scale" in self.current_button.type:
                scale = spr.get_scale(local=True)
                if scale.x < 0 and scale.y < 0:
                    scale = abs(scale)
                    spr.set_scale(scale, local=True)
                    spr.set_rotation((spr.get_rotation(local=True) + 180) % 360, local=True)
                    
            
            if self.current_button.type == "anchor":

                # Update anchor point
                new_anchor = spr.global_to_local(coord)
                spr.set_anchor_point(new_anchor)
                
            elif self.current_button.type == "clone":
                self.clone_sprite()
            elif self.current_button.type == "delete":
                self.delete_sprite()

            self.current_button = None        
            should_bubble = False
        
         #check to see if we added a keyframe, but skip for new sprites
        if spr is not None:
            keyframe_spr = spr
            # Compare current transform with starting transform to detect changes
            if self.added_new_sprite:
                self.added_new_sprite = False
                self.has_new_keyframes = False
            elif self.has_new_keyframes:
                self.history_manager.add_history(HistoryState(state={
                    "type": HistoryType.ADD_KEYFRAME,
                    "sprite": keyframe_spr,
                    "keyframe": keyframe_spr.keyframe_for_index(self.current_frame_index)
                }))
                self.has_new_keyframes = False
            elif not KeyFrame.transforms_similar(self.starting_transform, spr.local_transform):
              
                # Create keyframe if transform changed
                self.history_manager.add_history(HistoryState(state={
                    "type": HistoryType.MODIFY_TRANSFORM,
                    "sprite": keyframe_spr,
                    "start_transform": copy.deepcopy(self.starting_transform),
                    "end_transform": copy.deepcopy(spr.local_transform),
                    "frame_index": self.current_frame_index
                }))

        #redraw if deselected
        elif self.selected_sprite is not None:
            if not self.selected_sprite.is_point_in_sprite(coord):
                self.select_sprite(None)
            should_bubble = False

        return should_bubble

    
    def add_sprites_for_objects(self, objects: list[ObjectInfo]):
        for object in objects:
            # Skip if sprite already exists for this object ID
            if any(sprite.object_info.id == object.id for sprite in self.sprites):
                continue

            anchor_sprite = self.add_anchor_sprite(object)
            sprite = self.add_segmentation_sprite(object, parent=anchor_sprite)
            
            


        self.history_manager.clear_history()
        self.fx.api.should_refresh_tree = True
        self.fx.api.should_refresh_timeline = True
        

    def add_sprite_of_type(self, type:str, copy_sprite=None, parent=None):

        type = type.lower()
        parent = parent or self.world_sprite
        new_sprite = None
        if type == "cutout":
            if copy_sprite is not None:
                type = "object " + str(copy_sprite.object_info.id)

        if type == "video":
            if copy_sprite is not None:
                parent = copy_sprite.parent
                path = copy_sprite.image_path
            else:
                path = self.fx.api.open_file_dialog("Open Video", "Video Files (*.mp4);;All Files (*.*)")
            
            if path is None:
                return None
            
            video_sprite = VideoSprite(self, path, self.current_unique_id)
            new_sprite = self.add_generic_sprite(video_sprite, parent=parent)
            new_sprite.name = os.path.basename(path)

        if type == "image":
            if copy_sprite is not None:
                parent = copy_sprite.parent
                path = copy_sprite.image_path
            else:
                path = self.fx.api.open_file_dialog("Open Image", "Image Files (*.png *.jpg *.jpeg);;All Files (*.*)")
            
            if path is None:
                return None
            image_sprite = ImageSprite(self, path, self.current_unique_id)
            new_sprite = self.add_generic_sprite(image_sprite, parent=parent)
            new_sprite.name = os.path.basename(path)
            
        elif type == "text":
            font_options = {}
            text = "Hello Banana"
            if copy_sprite is not None:
                parent = copy_sprite.parent
                font_options =  copy.deepcopy(copy_sprite.font_options)
            text_sprite = TextSprite(self, 
                                                  font_options=font_options, 
                                                  unique_id=self.current_unique_id)
            new_sprite = self.add_generic_sprite(text_sprite, parent=parent)

        elif "object" in type:
            try:
                object_id = int(type.split()[1])
                objects = self.fx.api.get_objects()
                object_info = next((obj for obj in objects if obj.id == object_id), None)
                if object_info is None:
                    print("Error adding sprite of type: ", type, "object not found")
                    return
                # Find existing anchor sprite with matching object ID
                if copy_sprite is not None:
                    existing_anchor = copy_sprite.parent
                else:
                    existing_anchor = next((anchor for anchor in self.anchor_sprites if anchor.object_info.id == object_info.id), None)
                
                if existing_anchor is None:
                    print("Error adding sprite of type: ", type, "existing anchor found")
                    return None
                

                new_sprite = self.add_segmentation_sprite(object_info, parent=existing_anchor)
                existing_anchor.update_render_order()
                


            except (IndexError, ValueError):
                print("Error adding sprite of type: ", type, IndexError, ValueError)
        
        if new_sprite is not None:
            self.added_new_sprite = True
            
            if copy_sprite is not None:
                new_sprite.keyframes = copy.deepcopy(copy_sprite.keyframes)
                new_sprite.start_keyframe = copy.deepcopy(copy_sprite.start_keyframe)
                new_sprite.end_keyframe = copy.deepcopy(copy_sprite.end_keyframe)
                if parent == self.world_sprite:
                    world_size = self.fx.api.get_resolution()
                    new_sprite.set_position(world_size//2, frame_index=self.start_keyframe.frame_index)
                # new_sprite.set_position(copy_sprite.get_position() + Vector(20,20), 
                #                         local=True, 
                #                         frame_index=new_sprite.start_keyframe.frame_index)
            new_sprite.parent.update_render_order()
            self.select_sprite(new_sprite)
            self.fx.api.should_refresh_frame = True
            self.fx.api.should_refresh_timeline = True
            self.fx.api.should_refresh_tree = True

            return new_sprite

    def add_segmentation_sprite(self, object:ObjectInfo, parent=None):
        sprite = SegmentationSprite(self, object, self.current_unique_id)
        
        return self.add_generic_sprite(sprite, parent)
    
    def add_generic_sprite(self, sprite, parent=None):
        self.sprites.append(sprite)

        sprite.custom_render_order = len(self.sprites)
            

        sprite.set_parent(parent, change_position=False)
        self.current_unique_id += 1
        added_sprite = sprite
        self.history_manager.add_history(HistoryState(state={
            "type": HistoryType.ADD_SPRITE,
            "sprite": added_sprite
        }))
        

        
        return sprite
    
    def add_anchor_sprite(self, object:ObjectInfo):
        anchor_sprite = AnchorSprite(self, object, self.current_unique_id)
        self.anchor_sprites.append(anchor_sprite)
        #self.current_unique_id += 1
        return anchor_sprite

    def update(self, frame_info: FrameInfo, detections:sv.Detections, pose_anchors:dict):
        # self.transformed_ids = {}
        self.current_frame_index = frame_info.index

        for sprite in self.sprites:
            sprite.update_detections(detections)
            if sprite.use_keyframes:
                sprite.update_keyed_transform(frame_info)

        
        # work our way down the tree
        for sprite in self.anchor_sprites:
            sprite.update_detections(detections)
            sprite.update_transform() # updates all anchors and children

        self.world_sprite.update_transform()
            
        all_sprites = self.anchor_sprites + self.sprites + [self.world_sprite]
        for sprite in all_sprites:
            sprite.update_bbox()

        self.sprites.sort(key=lambda s: s.render_order)
        # update anchor positions
        #self.anchor_manager.update_anchors(pose_anchors, all_sprites)

    def render_anchors(self, frame_info: FrameInfo):
        self.render_sprites(frame_info, self.anchor_sprites + [self.world_sprite])

    def render_sprites(self, frame_info: FrameInfo, sprites=None):
        if sprites is None:
            sprites = self.sprites
        else:
            sprites.sort(key=lambda s: s.render_order)

        
        for sprite in sprites:
            sprite.render(frame_info)

   
    # def print_sprite_tree(self):
    #     def print_children(sprite, level=0, orphaned=False):
    #         indent = "  " * level
    #         print(f"{indent}└─ ID: {sprite.object_info.id}, Type: {sprite.type}, Orphaned: {orphaned}")
    #         for child in sprite.children:
    #             print_children(child, level + 1)

    #     print("\nSprite Tree:")
    #     print("------------")
    #     # Print anchor sprites first
    #     for sprite in self.anchor_sprites:
    #         print_children(sprite)
            
    #     # Print orphaned sprites (those without parents)
    #     orphaned_sprites = [s for s in self.sprites if s.parent is None]
    #     for sprite in orphaned_sprites:
    #         print_children(sprite, orphaned=True)


    def post_render(self, frame_info: FrameInfo, render_ui:bool=False):
        
        if render_ui:
            for sprite in self.sprites:
                if sprite != self.selected_sprite:
                    sprite.render_ui(frame_info)

            # render selected sprite last
            if self.selected_sprite:
                self.selected_sprite.render_ui(frame_info)

            
            

        else:
            pass
            #print("post_render")

        self.crop_sprite.render(frame_info)
        if render_ui:
            self.crop_sprite.render_ui(frame_info)
    
    def load_state(self):
        pass

    def save_state(self):
        pass
