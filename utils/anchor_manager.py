from enum import IntEnum
from fx_api.interface import ObjectInfo, FrameInfo
import supervision as sv
import fx_api.sprites.base_sprite as spr
import cv2
import numpy as np
from fx_api.utils.vector import Vector

# start at 1000 to avoid collision with object ids
class ANCHOR_ID(IntEnum): 
    NOSE = 1000
    LEFT_EYE_INNER = 1001
    LEFT_EYE = 1002
    LEFT_EYE_OUTER = 1003
    RIGHT_EYE_INNER = 1004
    RIGHT_EYE = 1005
    RIGHT_EYE_OUTER = 1006
    LEFT_EAR = 1007
    RIGHT_EAR = 1008
    MOUTH_LEFT = 1009
    MOUTH_RIGHT = 1010
    LEFT_SHOULDER = 1011
    RIGHT_SHOULDER = 1012
    LEFT_ELBOW = 1013
    RIGHT_ELBOW = 1014
    LEFT_WRIST = 1015
    RIGHT_WRIST = 1016
    LEFT_PINKY = 1017
    RIGHT_PINKY = 1018
    LEFT_INDEX = 1019
    RIGHT_INDEX = 1020
    LEFT_THUMB = 1021
    RIGHT_THUMB = 1022
    LEFT_HIP = 1023
    RIGHT_HIP = 1024
    LEFT_KNEE = 1025
    RIGHT_KNEE = 1026
    LEFT_ANKLE = 1027
    RIGHT_ANKLE = 1028
    LEFT_HEEL = 1029
    RIGHT_HEEL = 1030
    LEFT_FOOT_INDEX = 1031
    RIGHT_FOOT_INDEX = 1032
    NECK = 1033

    WORLD = 2000

class Anchor:
    def __init__(self, object_id: int, position: Vector, z: float=0.0, visibility: int=1):
        self.position = Vector(position).round()
        self.object_id = object_id
        self.visibility = visibility
        self.z = z

    def get_position(self, in_sprite = None, in_anchor_space = False)->Vector:
        # if in_sprite is not None:
        #     pos = in_sprite.apply_transform_to_frame_point(self.position)
        #     if in_anchor_space:
        #         return in_sprite.global_to_local(pos)
        # else:
        #     pos = Vector(self.position)
        # return pos.round() #cast to int
        return Vector(self.position).round()

class AnchorManager:
    def __init__(self, fx):
        self.anchors = {}
        self.fx = fx
    
    def update_anchors(self, pose_anchors: dict, sprites: list):
        self.anchors = {}
        # if pose_anchors is not None:
        #     self.anchors = pose_anchors
        #     pos_nums = [ANCHOR_ID.LEFT_SHOULDER, ANCHOR_ID.RIGHT_SHOULDER, ANCHOR_ID.LEFT_EAR, ANCHOR_ID.RIGHT_EAR]
        #     pts = []
        #     for pose_num in pos_nums:
        #         anchor = self.get_anchor(pose_num)
        #         if anchor is not None:
        #             pts.append(anchor.position)
        #     if len(pts) > 0:
        #         neck_pos = np.mean(pts, axis=0)
        #         self.anchors[ANCHOR_ID.NECK] = Anchor(ANCHOR_ID.NECK, Vector(neck_pos))
    
        # for sprite in sprites:
        #     pos = sprite.original_center
        #     id = sprite.object_id
        #     anchor = Anchor(id, pos)
        #     self.anchors[id] = anchor

        #     local_pos = sprite.local_to_global(Vector(0,0))
        #     local_anchor = Anchor(-id, local_pos)
        #     self.anchors[-id] = local_anchor

    def get_anchor(self, id: int):
        return self.anchors.get(id, None)
    
    def draw_line(self, frame_info: FrameInfo, start: int, end: int, color: tuple[int, int, int], in_sprite=None):
        start_anchor = self.get_anchor(start)
        end_anchor = self.get_anchor(end)
        if start_anchor and end_anchor:
            start_pos = start_anchor.get_position(in_sprite)
            end_pos = end_anchor.get_position(in_sprite)
            cv2.line(frame_info.render_buffer, start_pos, end_pos, color, 1)
    
    def render(self, frame_info: FrameInfo, render_ui:bool=False):
        if not render_ui or not self.fx.sprite_manager.should_draw_anchors():
            return
        
        selected_id = None
        local_to_sprite = None


        if self.fx.sprite_manager.selected_sprite is not None:
            selected_id = self.fx.sprite_manager.selected_sprite.object_info.id
            if self.fx.sprite_manager.current_button \
                and self.fx.sprite_manager.current_button.type == "anchor":
                local_to_sprite = self.fx.sprite_manager.selected_sprite

        face_color = (0, 255, 255)
        body_color = (255, 0, 255)
        for anchor in self.anchors.values():
  
           
            color = (0, 255, 0)
            if anchor.object_id >= 1000:
                color = body_color
                if anchor.object_id <= ANCHOR_ID.MOUTH_RIGHT and anchor.object_id != ANCHOR_ID.LEFT_EAR and anchor.object_id != ANCHOR_ID.RIGHT_EAR:
                    color = face_color
            

            pos = anchor.get_position(local_to_sprite if anchor.object_id >= 1000 else None)

            x, y = pos

            radius = 3

            if (local_to_sprite and anchor.object_id < 1000):
                pass
            else:
                pass
                #cv2.circle(frame_info.render_buffer, (x, y), radius, color, -1, cv2.LINE_AA)

            

        
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_EYE_INNER, ANCHOR_ID.RIGHT_EYE_OUTER, face_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_EYE_INNER, ANCHOR_ID.LEFT_EYE_OUTER, face_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.MOUTH_RIGHT, ANCHOR_ID.MOUTH_LEFT, face_color, local_to_sprite)
       
        self.draw_line(frame_info, ANCHOR_ID.LEFT_SHOULDER, ANCHOR_ID.RIGHT_SHOULDER, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_SHOULDER, ANCHOR_ID.NECK, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_SHOULDER, ANCHOR_ID.NECK, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_SHOULDER, ANCHOR_ID.LEFT_ELBOW, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_SHOULDER, ANCHOR_ID.RIGHT_ELBOW, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_ELBOW, ANCHOR_ID.LEFT_WRIST, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_ELBOW, ANCHOR_ID.RIGHT_WRIST, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_WRIST, ANCHOR_ID.LEFT_PINKY, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_WRIST, ANCHOR_ID.RIGHT_PINKY, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_WRIST, ANCHOR_ID.LEFT_THUMB, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_WRIST, ANCHOR_ID.RIGHT_INDEX, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_WRIST, ANCHOR_ID.LEFT_INDEX, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_WRIST, ANCHOR_ID.RIGHT_THUMB, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_SHOULDER, ANCHOR_ID.LEFT_HIP, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_SHOULDER, ANCHOR_ID.RIGHT_HIP, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_HIP, ANCHOR_ID.RIGHT_HIP, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_HIP, ANCHOR_ID.LEFT_KNEE, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_HIP, ANCHOR_ID.RIGHT_KNEE, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_KNEE, ANCHOR_ID.LEFT_ANKLE, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_KNEE, ANCHOR_ID.RIGHT_ANKLE, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_ANKLE, ANCHOR_ID.LEFT_HEEL, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_ANKLE, ANCHOR_ID.RIGHT_HEEL, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.LEFT_HEEL, ANCHOR_ID.LEFT_FOOT_INDEX, body_color, local_to_sprite)
        self.draw_line(frame_info, ANCHOR_ID.RIGHT_HEEL, ANCHOR_ID.RIGHT_FOOT_INDEX, body_color, local_to_sprite)
        

       
        
        
