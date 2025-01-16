
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
    def alpha_blend(background:np.ndarray, foreground:np.ndarray, position:Vector, centered=False):
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

        # Extract alpha channel and normalize to 0-1 range
        alpha = foreground[src_start_y:src_end_y, src_start_x:src_end_x, 3:4] / 255.0
        
        background_height, background_width = background.shape[:2]
        # Ensure dst_start and dst_end are all in range
        if dst_start_y < background_height and dst_start_x < background_width and dst_end_y > 0 and dst_end_x > 0:
            # Blend rgba_large onto background using alpha compositing
            background[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
                (1 - alpha) * background[dst_start_y:dst_end_y, dst_start_x:dst_end_x] + \
                alpha * foreground[src_start_y:src_end_y, src_start_x:src_end_x, :3]