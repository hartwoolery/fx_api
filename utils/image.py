
from fx_api.utils.vector import Vector
import numpy as np
import cv2

class ImageUtils:

    @staticmethod
    def draw_dashed_line(background:np.ndarray, start:Vector, end:Vector, color:tuple[int, int, int], thickness:int=2, dash_length:int=6, gap_length:int=6):
        direction = end - start
        length = direction.length()
        direction = direction.normalize()
        for i in range(0, int(length), dash_length + gap_length):
            dash_start = start + direction * i
            dash_end = start + direction * min(i + dash_length, length)
            cv2.line(background, dash_start.round(), dash_end.round(), color, thickness, cv2.LINE_AA)

    @staticmethod
    def blend(background:np.ndarray, foreground:np.ndarray, position:Vector, centered=False, blend_mode:str="normal"):
        position = position.round()
        if centered:
            paste_x = position[0] - foreground.shape[1] // 2
            paste_y = position[1] - foreground.shape[0] // 2
        else:
            paste_x = position[0]
            paste_y = position[1]

        # Calculate valid source and destination regions accounting for image boundaries
        src_start_y = max(0, -paste_y)
        src_end_y = min(foreground.shape[0], background.shape[0] - paste_y)
        src_start_x = max(0, -paste_x) 
        src_end_x = min(foreground.shape[1], background.shape[1] - paste_x)

        dst_start_y = max(0, paste_y)
        dst_end_y = min(background.shape[0], paste_y + foreground.shape[0])
        dst_start_x = max(0, paste_x)
        dst_end_x = min(background.shape[1], paste_x + foreground.shape[1])


        background_height, background_width = background.shape[:2]

        def blend_bg(background, foreground, alpha, blend_mode):
            # Convert inputs to float32 and normalize to 0-1 range
            bg = background.astype(np.float32) / 255.0
            fg = foreground.astype(np.float32) / 255.0
            
            if blend_mode == "normal":
                result = (1 - alpha) * bg + alpha * fg
            elif blend_mode == "additive":
                result = np.minimum(bg + alpha * fg, 1.0)
            elif blend_mode == "subtractive":
                result = np.maximum(bg - alpha * fg, 0.0)
            elif blend_mode == "multiply":
                result = bg * ((1 - alpha) + alpha * fg)
            elif blend_mode == "screen":
                result = 1 - (1 - bg) * (1 - alpha * fg)
            elif blend_mode == "overlay":
                mask = bg > 0.5
                result = np.where(mask,
                    1 - (1 - 2*(bg-0.5)) * (1 - alpha * fg),
                    2 * bg * (alpha * fg)) * alpha + bg * (1 - alpha)
            elif blend_mode == "darken":
                result = np.minimum(bg, fg) * alpha + bg * (1 - alpha)
            elif blend_mode == "lighten":
                result = np.maximum(bg, fg) * alpha + bg * (1 - alpha)
            elif blend_mode == "color dodge":
                dodge = np.minimum(1, bg / (1 - fg + 1e-6))
                result = dodge * alpha + bg * (1 - alpha)
            elif blend_mode == "color burn":
                burn = 1 - np.minimum(1, (1 - bg) / (fg + 1e-6))
                result = burn * alpha + bg * (1 - alpha)
            elif blend_mode == "hard light":
                mask = fg > 0.5
                result = alpha * np.where(mask,
                    1 - (1 - bg) * (1 - 2*(fg-0.5)),
                    bg * (2*fg)) + (1 - alpha) * bg
            elif blend_mode == "soft light":
                result = alpha * (bg * (fg + 0.5)) + (1 - alpha) * bg
            elif blend_mode == "difference":
                result = np.abs(bg - alpha * fg)
            elif blend_mode == "exclusion":
                result = bg + alpha * fg - 2 * bg * alpha * fg
            elif blend_mode in ["hue", "saturation", "color", "luminosity"]:
                # Convert to HSV, keeping arrays as float32
                bg_hsv = cv2.cvtColor(np.clip(bg, 0, 1), cv2.COLOR_RGB2HSV)
                fg_hsv = cv2.cvtColor(np.clip(fg, 0, 1), cv2.COLOR_RGB2HSV)
                
                if blend_mode == "hue":
                    bg_hsv[...,0] = alpha[...,0] * fg_hsv[...,0] + (1 - alpha[...,0]) * bg_hsv[...,0]
                elif blend_mode == "saturation":
                    bg_hsv[...,1] = alpha[...,0] * fg_hsv[...,1] + (1 - alpha[...,0]) * bg_hsv[...,1]
                elif blend_mode == "color":
                    bg_hsv[...,0] = alpha[...,0] * fg_hsv[...,0] + (1 - alpha[...,0]) * bg_hsv[...,0]
                    bg_hsv[...,1] = alpha[...,0] * fg_hsv[...,1] + (1 - alpha[...,0]) * bg_hsv[...,1]
                else:  # luminosity
                    bg_hsv[...,2] = alpha[...,0] * fg_hsv[...,2] + (1 - alpha[...,0]) * bg_hsv[...,2]
                
                result = cv2.cvtColor(bg_hsv, cv2.COLOR_HSV2RGB)
            
            # Convert back to 0-255 range
            return result * 255.0


        # Ensure dst_start and dst_end are all in range
        if dst_start_y < background_height and dst_start_x < background_width and dst_end_y > 0 and dst_end_x > 0:
            # Blend rgba_large onto background using alpha compositing

            # Extract alpha channel and normalize to 0-1 range
            alpha = foreground[src_start_y:src_end_y, src_start_x:src_end_x, 3:4] / 255.0
            # Check if the background has an alpha channel
            if background.shape[2] == 4:
                # Separate the RGB and alpha channels
                bg_rgb = background[dst_start_y:dst_end_y, dst_start_x:dst_end_x, :3]
                bg_alpha = background[dst_start_y:dst_end_y, dst_start_x:dst_end_x, 3:4] / 255.0

                # Blend the RGB channels
                blended_rgb = blend_bg(bg_rgb, foreground[src_start_y:src_end_y, src_start_x:src_end_x, :3], alpha, blend_mode)

                # Blend the alpha channels
                blended_alpha = np.clip(bg_alpha + alpha * (1 - bg_alpha), 0, 1) * 255.0

                # Combine the blended RGB and alpha channels
                background[dst_start_y:dst_end_y, dst_start_x:dst_end_x, :3] = np.clip(blended_rgb, 0, 255)
                background[dst_start_y:dst_end_y, dst_start_x:dst_end_x, 3:4] = blended_alpha
            else:
                # Blend the RGB channels only
                blended = blend_bg(
                    background[dst_start_y:dst_end_y, dst_start_x:dst_end_x],
                    foreground[src_start_y:src_end_y, src_start_x:src_end_x, :3],
                    alpha,
                    blend_mode
                )
                background[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = np.clip(blended, 0, 255)

            