import cv2
import os
import numpy as np
from fx_api.interface import ObjectInfo, FrameInfo
from fx_api.utils.vector import Vector
from fx_api.utils.image import ImageUtils
from fx_api.sprites.base_sprite import BaseSprite

############################################################
# Video Sprite
############################################################

class VideoSprite(BaseSprite):
    def __init__(self, sprite_manager, video_path:str, unique_id:int):
        
        super().__init__(sprite_manager, None, unique_id, "video")
        self.video_path = video_path
        self.temp_folder = None
        self.frame_count = 0
        self.use_chroma_key = True
        self.time_stretch = 0.0
        self.default_time_stretch = 0.0
        self.frames_per_second = 30.0
        self.chroma_key_color = (0,0,0)
        self.chroma_key_choke = 10  # -1.0 to 1.0, negative shrinks mask, positive expands
        self.chroma_key_spill = 50  # 0.0 to 1.0, higher values reduce color bleeding
        
        self.change_sprite_path(video_path)

    def chroma_key_changed(self, color: tuple[int,int,int]):
        self.chroma_key_color = color

    def get_choke_mask(self, hsv:np.ndarray, frame_index:int):
        if self.choke_masks[frame_index] is not None:
            return self.choke_masks[frame_index]
        # Convert RGB chroma key color to HSV
        chroma_rgb = np.uint8([[list(self.chroma_key_color)]])
        big_color = cv2.cvtColor(chroma_rgb, cv2.COLOR_RGB2HSV)[0][0]

        
        # Create distance map instead of binary mask
        color_diff = np.abs(hsv - np.array(big_color, dtype=np.float32))
        # Normalize differences and combine channels with weights
        weights = np.array([2.0, 1.0, 1.0])  # More weight on hue
        normalized_diff = np.sum(color_diff * weights, axis=2) / (np.sum(weights) + 1e-10)
        
        # Convert to mask (0-255)
        mask = np.clip(255 - (normalized_diff * 255 / 10), 0, 255).astype(np.uint8)


        # Apply choke to the mask (positive expands, negative shrinks)
        choke_offset = self.chroma_key_choke * 0.5  # Scale factor for more noticeable effect
        # Apply choke to the mask (positive expands, negative shrinks)
        max_kernel_size = 31
        choke_offset = self.chroma_key_choke / 100
        if choke_offset != 0:
            # Scale from -100,100 to reasonable kernel size and ensure it's odd
            kernel_size = int(abs(choke_offset * max_kernel_size))
            kernel_size = max(1, min(max_kernel_size, kernel_size))
            
            if kernel_size % 2 == 0:
                kernel_size += 1

            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                if choke_offset > 0:
                    mask = cv2.dilate(mask, kernel, iterations=1)
                else:
                    mask = cv2.erode(mask, kernel, iterations=1)
        
        # Smooth the mask
        #mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)

        # Create inverse mask
        #mask_inv = np.clip(255 - mask, 0, 255) #mask_blurred
        # Create inverse mask
        self.choke_masks[frame_index] = mask
        return mask
    
    def get_spill_mask(self, mask:np.ndarray, frame_index:int):
        if self.spill_masks[frame_index] is not None:
            return self.spill_masks[frame_index]
        
        # Calculate edge mask using Sobel operators with larger kernel
        sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=7)  # Increased kernel size
        sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=7)
        edge_mask = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and expand edge area even more aggressively
        max_val = np.max(edge_mask)
        if max_val <= 0:
            max_val = 1

        edge_mask = edge_mask / max_val
        kernel_size = int(self.chroma_key_spill / 2)  # Even less division for larger kernel
        kernel_size = max(7, min(51, kernel_size))  # Increased max kernel size further
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply multiple gaussian blurs to spread the effect further
        edge_mask = cv2.GaussianBlur(edge_mask, (kernel_size, kernel_size), 0)
        edge_mask = cv2.GaussianBlur(edge_mask, (kernel_size, kernel_size), 0)  # Second blur
        
        # Enhance edge mask contrast more dramatically
        edge_mask = np.power(edge_mask, 0.3)  # More aggressive power value
        
        # Apply spill strength with even more intensity
        spill_strength = (self.chroma_key_spill / 100.0) * 4.0  # Quadrupled the effect
        spill_strength = min(1.0, spill_strength)
        spill_mask = edge_mask * spill_strength
        
        # Blend between original and desaturated based on spill mask
        spill_mask = np.expand_dims(spill_mask, axis=2)
        self.spill_masks[frame_index] = spill_mask
        return spill_mask
        

    def get_sprite_image(self, frame_info:FrameInfo):
        start_frame = self.start_keyframe.frame_index
        end_frame = self.end_keyframe.frame_index
        current_frame = frame_info.index

        offset_frame = int((current_frame - start_frame) * self.get_time_stretch())
        
        # Ensure frame index stays within valid range using modulus
        if self.frame_count > 0:
            
            offset_frame = offset_frame % self.frame_count
            rgb = self.get_frame_image(offset_frame).copy()
            opacity = self.get_opacity()


            #chroma key
            if self.chroma_key_color is not None and self.use_chroma_key:
                # change to hsv
                hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV);

                mask = self.get_choke_mask(hsv, offset_frame)

                # Apply spill suppression only to edges
                if self.chroma_key_spill > 0:
                    spill_mask = self.get_spill_mask( mask, offset_frame)

                    # Convert to grayscale for desaturation
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    rgb = rgb * (1 - spill_mask) + np.expand_dims(gray, axis=2) * spill_mask

                # Stack RGB with alpha channel

                mask_inv = cv2.GaussianBlur(255 - mask, (3, 3), 0)
                rgba = np.dstack((rgb, mask_inv))
            else:
                rgba = np.dstack((rgb, np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)))
            
            # Apply opacity to alpha channel
            rgba[:,:,:3] = self.recolor_sprite(rgba[:,:,:3])
            if rgba.shape[2] == 4:  # Check if image has alpha channel
                rgba[:, :, 3] = rgba[:, :, 3] * opacity
            return rgba
        return None
    
    def get_frame_image(self, frame_index:int):
        frame_path = os.path.join(self.temp_folder, f"frame_{frame_index:06d}.jpg")
        return cv2.imread(frame_path)
    
    def reset_spill_masks(self):
        self.spill_masks = [None] * self.frame_count

    def reset_choke_masks(self):
        self.choke_masks = [None] * self.frame_count

    def get_chroma_key_color(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV);
        h,s,v = cv2.split(hsv);

        # get uniques
        unique_colors, counts = np.unique(s, return_counts=True);

        # sort through and grab the most abundant unique color
        big_color = None;
        biggest = -1;
        for a in range(len(unique_colors)):
            if counts[a] > biggest:
                biggest = counts[a];
                # Get full HSV color at location where saturation matches
                sat_mask = (s == unique_colors[a])
                if np.any(sat_mask):
                    # Get first pixel location matching the saturation
                    y, x = np.where(sat_mask)[0][0], np.where(sat_mask)[1][0]
                    big_color = (int(h[y,x]), int(s[y,x]), int(v[y,x]))

        
        # Convert HSV big_color to BGR color

        bgr_color = cv2.cvtColor(np.uint8([[list(big_color)]]), cv2.COLOR_HSV2BGR)[0][0]
        self.chroma_key_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
        

    def get_time_stretch(self):
        ts = 1.0 + abs(self.time_stretch)*0.01
        if self.time_stretch < 0:
            ts = 1.0 / ts
            
        return self.default_time_stretch * ts
        
    def change_sprite_path(self, path:str):
        self.video_path = path
        name = os.path.splitext(os.path.basename(path))[0]
        self.temp_folder = self.sprite_manager.fx.api.get_temp_directory(name)
        
        # Open video file
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video file {path}")
            return

        # Get first frame to set initial image
        ret, self.image = cap.read()
        if not ret:
            print(f"Error reading first frame from {path}")
            return

        # Extract frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as image file
            frame_path = os.path.join(self.temp_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        self.frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        self.default_time_stretch = self.sprite_manager.fx.api.get_frames_per_second() / self.frames_per_second
        
        self.time_stretch = 0

        cap.release()
        self.frame_count = frame_count
        if self.frame_count > 0:
            self.start_keyframe.frame_index = self.sprite_manager.current_frame_index
            total_frames = self.sprite_manager.fx.api.get_total_frames()

            ts = self.get_time_stretch()
            max_end_frame = min(self.start_keyframe.frame_index + int(self.frame_count*ts), total_frames - 1)
            
            self.end_keyframe.frame_index = max_end_frame

            self.reset_choke_masks()
            self.reset_spill_masks()

            image = self.get_frame_image(0)
            self.get_chroma_key_color(image)
            resolution = Vector(image.shape[1], image.shape[0])
            if resolution.x > 512:
                new_scale = 512 / resolution.x
            resolution *= new_scale

            world_size = self.sprite_manager.fx.api.get_resolution()
            self.bbox = (0,0, resolution.x, resolution.y)
           
            self.true_size = resolution
            self.update_bbox()
            self.set_position(world_size//2, frame_index=self.start_keyframe.frame_index)
            
        else:
            self.enabled = False

    def render(self, frame_info:FrameInfo, transform:dict={}):
        super().render(frame_info, transform)
        if self.frame_count == 0 or not self.get_enabled():
            return
        
        
        rgba = self.get_sprite_image(frame_info)
        self.blit_sprite(frame_info, rgba, self.image is not None)