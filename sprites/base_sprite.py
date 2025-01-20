import cv2
import numpy as np
import supervision as sv

from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.utils.image import ImageUtils
from fx_api.sprites.transforms import Transformable
from PIL import Image, ImageDraw, ImageFont
from fx_api.sprites.sprite_ui import UIButton
from fx_api.utils.noise_filter import OneEuroFilter2D

class BaseSprite(Transformable):
    def __init__(self, sprite_manager, object_info: ObjectInfo, unique_id:int, type:str="base"):
        super().__init__(sprite_manager)
        self.type = type
        self.name = self.type + " " + str(unique_id)
        self.object_info = object_info if object_info is not None else ObjectInfo()
        self.meta_data = {}

        self.position_filter = OneEuroFilter2D(freq=60, mincutoff=1.0, beta=0.1, dcutoff=1.0)
        self.smoothing = 0

        self.blend_mode = "normal"

        num_frames = 30
        end_frame = self.sprite_manager.fx.api.get_total_frames() - 1
        cur_frame = self.sprite_manager.current_frame_index
        start_frame = 0           
        end = 0
        if cur_frame < num_frames:
            start = 0
            end = num_frames
        elif cur_frame > end_frame - num_frames:
            start = end_frame - num_frames
            end = end_frame
        else:
            start = cur_frame
            end = cur_frame + num_frames

        last_frame = self.sprite_manager.fx.api.get_total_frames() - 1
        self.start_keyframe.frame_index = self.object_info.start_frame or start
        self.end_keyframe.frame_index = self.object_info.end_frame or end

        colors = self.sprite_manager.fx.api.get_color_palette()

        color_id = object_info.id if object_info is not None else unique_id
        self.object_info.color = colors[color_id % len(colors)]
   


        self.image = None
        self.image_path = None
        self._enabled = True
        # the sprite's normalized anchor point relative to its transform

        
        self.rotation_estimate = None

        self.is_segmentation_sprite = False
        
        self.locked = False
        self.bbox = None
        self.mask = None
        self.unique_id = unique_id
        self.bbox_corners = [] # the global corners of the sprite
        self.bbox_center = []

        self.true_size = Vector(0,0) # the orignal size of the sprite in pixels
        self.interactable = True
        
        self.recolor = None

        self.transform_buttons = []
        for i in range(11):
            transform_type = "scale" 
            if i == 1 and self.type != "crop": transform_type = "rotation"
            elif i == 4 or i == 6: transform_type = "scale_y"
            elif i == 5 or i == 7: transform_type = "scale_x"
            if self.type == "crop" and i > 7:
                continue
            if i == 8: transform_type = "anchor"
            elif i == 9: transform_type = "clone"
            elif i == 10: transform_type = "delete"
            self.transform_buttons.append(UIButton(self, transform_type=transform_type ))
        
    def get_meta(self, key:str, default:any=None):
        return self.meta_data.get(key, default)
    
    def set_meta(self, key:str, value:any):
        self.meta_data[key] = value

    def set_smoothing(self, smoothing:int):
        self.smoothing = smoothing
        self.position_filter.setParameters( freq=60, mincutoff=10/(smoothing*10+10), beta=1/(smoothing*10+10), dcutoff=1.0)
        
    def get_enabled(self):
        return self._enabled and self.start_keyframe.frame_index <= self.sprite_manager.current_frame_index <= self.end_keyframe.frame_index
    
    def set_enabled(self, is_enabled:bool):
        self._enabled = is_enabled
        for child in self.children:
            child.set_enabled(is_enabled)

    def is_point_in_sprite(self, coord:Vector, buffer:int=0,) -> bool:

        corners = self.bbox_corners.copy()
                
        if len(corners) < 4:
            return False
        if buffer > 0:
            expanded_corners = cv2.approxPolyDP(np.array(corners, dtype=np.float32), buffer, True)
            return cv2.pointPolygonTest(expanded_corners, np.array(coord, dtype=np.float32), False) >= 0
        else:
            return cv2.pointPolygonTest(np.array(corners, dtype=np.float32), np.array(coord, dtype=np.float32), False) >= 0
    
  

    def dragging_button(self, type:str) -> bool:
        return self.sprite_manager.current_button is not None and self.sprite_manager.current_button.type == type




    def get_mask(self, detections:sv.Detections):
        if detections is not None:
            masks = detections.mask[detections.tracker_id == self.object_info.id]
            bboxes = detections.xyxy[detections.tracker_id == self.object_info.id]
            if len(masks) > 0 and len(bboxes) > 0:
                return masks[0], bboxes[0]
        return None, None

    def update_detections(self, detections:sv.Detections):
        
        mask, bbox = self.get_mask(detections)
        if mask is not None and bbox is not None:
            self.mask = mask
            self.bbox = bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # Set the true size
            true_width = x2 - x1
            true_height = y2 - y1
            self.true_size = Vector(true_width, true_height)

            if self.type == "anchor":
                center = Vector(x1 + true_width//2, y1 + true_height//2)
                self.local_transform["translation"] = center

    
    def update_bbox(self):
        if self.bbox is not None:

            if self.type == "crop":
                x1, y1, x2, y2 = map(int, self.bbox)
                self.bbox_corners = [Vector(x1,y1), Vector(x2,y1), Vector(x2,y2), Vector(x1,y2)]
                return

            half_size = self.true_size // 2
            

            center_local = Vector(0,0)
            center = self.local_to_global(center_local)


            x1, y1 = center_local - half_size
            x2, y2 = center_local + half_size

            corners = [Vector(x1,y1), Vector(x2,y1), Vector(x2,y2), Vector(x1,y2)]
            corners = [self.local_to_global(corner) for corner in corners]

            
            # corners = [Vector(x1,y1), Vector(x2,y1), Vector(x2,y2), Vector(x1,y2)]
            # corners = [self.local_to_global(corner - self.get_position()) for corner in corners]
            # print("setting bbox corners", corners, self.get_position(local=True))

            self.bbox_center = center
            self.bbox_corners = corners  
    '''       
    def estimate_rotation(self, mask_crop:np.ndarray, background:np.ndarray):
            
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit an ellipse to the contour if it has enough points
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Extract angle from ellipse
                # OpenCV returns angle in range [0,180] counter-clockwise from horizontal
                # Convert to our coordinate system
                angle = -ellipse[2] + 90 #- 90  # Negate to match our clockwise rotation
                if mask_crop.shape[1] > mask_crop.shape[0]:  # If width > height
                    
                    angle -= 90  # Add 90 degrees
                
                # Get center and axes lengths from ellipse
                center = tuple(map(int, ellipse[0]))
                axes = tuple(map(int, ellipse[1]))
                angle_rad = np.deg2rad(ellipse[2]) - np.pi/2

                

                # Calculate end point of major axis
                major_axis_length = max(axes) / 2
                dx = major_axis_length * np.cos(angle_rad) 
                dy = major_axis_length * np.sin(angle_rad)

                if dy > 0:
                    dy *= -1
                    dx *= -1

                # Calculate new angle based on dx and dy
                new_angle = np.rad2deg(np.arctan2(-dy, dx))
                # Normalize angle to [-180, 180]
                while new_angle > 180:
                    new_angle -= 360
                while new_angle < -180:
                    new_angle += 360
                angle = new_angle
                
                # Draw line from center to end of major axis
                end_point = (int(center[0] + dx), int(center[1] + dy))
                cv2.line(background, center, end_point, (0,255,0), 2)
                cv2.circle(background, end_point, 5, (0,255,0), -1)

                # # Normalize angle to [-180, 180]
                # while angle > 180:
                #     angle -= 360
                # while angle < -180:
                #     angle += 360
                # Set the rotation
                self.rotation_estimate = -angle
    
    ''' 
    def recolor_changed(self, color:tuple[int,int,int]):
        self.recolor = (color[0], color[1], color[2], 255)

    def recolor_sprite(self, frame_crop:np.ndarray):
        if self.recolor is None:
            return frame_crop
         # Convert frame crop to HSV for recoloring
        frame_crop_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        # Extract hue from recolor and apply it
        h, s, v = cv2.split(frame_crop_hsv)
        # Convert BGR recolor to HSV and extract hue
        recolor_hsv = cv2.cvtColor(np.uint8([[self.recolor]]), cv2.COLOR_BGR2HSV)
        h.fill(recolor_hsv[0][0][0])  # Fill with extracted hue value
        # Increase saturation by 50%
        #s.fill(self.recolor[3] *)  # Set saturation to maximum value
        s = s.astype(np.uint8)
        # Multiply value channel by recolor value (normalized to 0-1)
        #v = v * (recolor_hsv[0][0][2] / 255.0)
        v = v.astype(np.uint8)
        # Merge channels back and convert to BGR
        frame_crop = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return frame_crop

    def get_sprite_image(self, frame_info:FrameInfo):
        pass

    def blit_sprite(self, frame_info:FrameInfo, rgba:np.ndarray, is_transformed:bool):
        if rgba is None:
            print("blit image is None")
            return
        rotation = self.get_rotation()
        scale = self.get_scale()
        translation = self.get_translation()
        x1, y1, x2, y2 = map(int, self.bbox)
        if is_transformed:
            final_anchor = self.local_to_global(Vector(0,0))
            # Get image dimensions
            p1 = Vector(x1, y1)
            p2 = Vector(x2, y2)
            width, height = p2 - p1
            size = Vector(width, height)
            half_size = size * 0.5
            # Calculate the center of the image
            center = half_size * scale 
            #Vector(width*scale[0] / 2, height*scale[1] / 2)
            unscaled = scale.x == 1 and scale.y == 1 and self.image is None
            rotated = rotation != 0 
            # The position of the anchor point in the crop
            signs = Vector(1,1)

            
            if scale.x < 0:
                rgba = cv2.flip(rgba, 1)  # Flip horizontally
                signs = Vector(-1,1)
            if scale.y < 0:
                rgba = cv2.flip(rgba, 0)  # Flip vertically
                signs = Vector(1,-1)
            if scale.x < 0 and scale.y < 0:
                signs = Vector(-1,-1)


            
            if not unscaled:
                new_size = (self.true_size * max(abs(scale), Vector(0.01, 0.01))).round()
                if new_size > 0:
                    rgba = cv2.resize(rgba, new_size, interpolation=cv2.INTER_LINEAR)
                # Adjust crop_anchor for flipping
            
            crop_anchor = half_size * signs #+ self.anchor_point * signs #half_size + self.anchor_point * signs * half_size
            if signs.x < 0:
                crop_anchor += Vector(size.x, 0)
            if signs.y < 0:
                crop_anchor += Vector(0, size.y)
            crop_anchor = crop_anchor * abs(scale)

                
            if not rotated:
                rgba_large = rgba
            else:
                # Get the longest side of the rgba image
                max_side = max(rgba.shape[0], rgba.shape[1])
                
                # Calculate new size (1.5x the longest side)
                new_size = int(max_side * 1.5)
                
                # Create padded rgba image
                padded_rgba = np.zeros((new_size, new_size, 4), dtype=np.uint8)
                
                # Calculate padding amounts to center the original image
                pad_y = (new_size - rgba.shape[0]) // 2
                pad_x = (new_size - rgba.shape[1]) // 2
                
                # Place original rgba in center of padded image
                padded_rgba[pad_y:pad_y+rgba.shape[0], pad_x:pad_x+rgba.shape[1]] = rgba
                
                # Update rgba to use padded version
                rgba = padded_rgba

                # Calculate the center of the padded image
                center = (new_size / 2, new_size / 2)

                # Get the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                

                # Apply the rotation to the padded image
                rgba_large = cv2.warpAffine(rgba, rotation_matrix, (new_size, new_size))

                # Adjust crop_anchor for padding
                crop_anchor += Vector(pad_x, pad_y)

                # Transform the crop anchor point using the rotation matrix
                crop_anchor_m = np.array([[crop_anchor[0]], [crop_anchor[1]], [1]])
                transformed_crop_anchor = rotation_matrix.dot(crop_anchor_m)
                crop_anchor = Vector(transformed_crop_anchor[0][0], transformed_crop_anchor[1][0])

            
            final_position = (final_anchor - crop_anchor).round()
            
        else:   
            final_position = Vector(x1, y1)
            rgba_large = rgba



        if self.smoothing > 0:
            if self.sprite_manager.current_frame_index == 0:
                self.position_filter.reset()
            
            final_position = self.position_filter(final_position, frame_info.time)

        ImageUtils.blend(frame_info.render_buffer, rgba_large, final_position, centered=False, blend_mode=self.blend_mode)

    def render(self, frame_info:FrameInfo, transform:dict={}):
        self.transform_override = transform
       
        if self.is_segmentation_sprite and self.bbox is not None:
            x1, y1, x2, y2 = map(int, self.bbox)
            size = Vector(x2 - x1, y2 - y1)
            self.set_enabled(size > 0)

            

    def render_ui(self, frame_info:FrameInfo):
        if not self.get_enabled() or len(self.bbox_corners) < 4:
            return
        
        background = frame_info.render_buffer
        render_as_selected = self.sprite_manager.selected_sprite == self
       
        reparenting = self.sprite_manager.selected_sprite and \
            self.sprite_manager.reparent_sprite == self and \
            self.sprite_manager.current_modifiers.get("shift", False) == True and \
            self.sprite_manager.selected_sprite.parent != self
        

        color = (255, 255, 255) if render_as_selected else (255,255,255)
        # Check if there is a keyframe for the current frame index
        # for keyframe in self.keyframes:
        #     if keyframe.frame_index == frame_info.index:
        #         color = (255, 255, 0) if render_as_selected else (170, 170, 0)
        #         break
        # Draw rotated bounding box using the corners
        color = (0,255,0) if reparenting else color
        bbox_thickness = 3 if reparenting else 2 if render_as_selected else 1
        if self.locked:
            color = (0,0,255)

        for i in range(len(self.bbox_corners)):
            start_point = Vector(self.bbox_corners[i].x, self.bbox_corners[i].y).round()
            end_point = Vector(self.bbox_corners[(i+1)%4].x, self.bbox_corners[(i+1)%4].y).round()
            if render_as_selected:
                cv2.line(background, start_point, end_point, color, bbox_thickness, cv2.LINE_AA)
            else:
                ImageUtils.draw_dashed_line(background, start_point, end_point, color, bbox_thickness, 6, 8)

        
        
        if not render_as_selected or self.locked:
            return

        # Draw circles at corners of bounding box
        for i, corner in enumerate(self.bbox_corners):
            self.transform_buttons[i].draw(corner, background)
            

        # Draw circles at midpoints of bounding box lines
        for i in range(len(self.bbox_corners)):
            start_point = self.bbox_corners[i]
            end_point = self.bbox_corners[(i+1)%4]
            
            # Calculate midpoint
            mid_x = int((start_point[0] + end_point[0]) / 2)
            mid_y = int((start_point[1] + end_point[1]) / 2)
            
            # Draw filled circle at midpoint
            self.transform_buttons[i+4].draw((mid_x, mid_y), background)

        if self.type == "crop":
            return

        # Draw line from anchor point to parent anchor point

        anchor_point_frame = self.local_to_global(self.temp_anchor_point)
        if self.parent is not None and \
            self.sprite_manager.selected_sprite.parent != self.sprite_manager.world_sprite:
            parent_anchor = self.parent.local_to_global(self.parent.anchor_point)
            ImageUtils.draw_dashed_line(background, anchor_point_frame, parent_anchor, (255, 255, 255), 2, 6, 8)
            cv2.circle(background, parent_anchor.round(), 5, (255, 255, 255), -1)
        
        # Draw circle at original anchor point
        self.transform_buttons[8].draw(anchor_point_frame, background)

        
        # Get upper corners of bbox
        upper_left = self.bbox_corners[0]
        upper_right = self.bbox_corners[1]

        scale = self.get_scale()
        if scale.x * scale.y < 0:  # If exactly one scale component is negative flip the corners
            upper_left = self.bbox_corners[1]
            upper_right = self.bbox_corners[0]

        # Calculate angle of top edge
        edge_vector = upper_right - upper_left
        angle = np.arctan2(edge_vector.y, edge_vector.x)
            
        
        # Calculate offset perpendicular to the rotated top edge
        perpendicular_angle = angle - np.pi/2  # 90 degrees counter-clockwise
        offset_magnitude = 40
        offset = Vector(
            offset_magnitude * np.cos(perpendicular_angle),
            offset_magnitude * np.sin(perpendicular_angle)
        )
        
        # Calculate center point of top edge
        center = (upper_left + upper_right) * 0.5
        
        # Position buttons above center point
        delete_pos = center + offset - Vector(
            20 * np.cos(angle),  # Offset along edge direction
            20 * np.sin(angle)   # Maintain angle of box
        )
        clone_pos = center + offset + Vector(
            20 * np.cos(angle),  # Offset along edge direction  
            20 * np.sin(angle)   # Maintain angle of box
        )
        
        self.transform_buttons[9].draw(clone_pos, background)
        self.transform_buttons[10].draw(delete_pos, background)

        

        hint_text = None
        if self.sprite_manager.current_modifiers.get("shift", False) == False and \
            self.sprite_manager.reparent_sprite is not None:
            hint_text = "hold SHIFT to re-parent"

        if self.sprite_manager.current_modifiers.get("shift", False) == False and \
            self.sprite_manager.current_button is not None and \
            self.sprite_manager.current_button.type == "scale":
            hint_text = "hold SHIFT to maintain aspect"

        if hint_text is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.66
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(hint_text, font, font_scale, thickness)
            
            # Position text centered above the sprite

            text_x = int(self.bbox_center.x - text_width*0.5)
            # Get the corner with highest y value (lowest on screen)
            max_y = max(corner[1] for corner in self.bbox_corners)
            text_y = int(max_y + 30)  # 30 pixels below the lowest point
           
            # Draw text with black outline for better visibility
            cv2.putText(background, hint_text, (text_x, text_y), font, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
            cv2.putText(background, hint_text, (text_x, text_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)




  
    
    
    

    
    

       
        



        
    