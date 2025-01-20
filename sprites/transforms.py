
from fx_api.utils.vector import Vector
from fx_api.interface import FrameInfo
from fx_api.utils.easing import Easing

import numpy as np
import copy

class KeyFrame:
    def __init__(self, frame_index, transform={}, is_bookend:bool=False):
        self.frame_index = frame_index
        self.transform = transform.copy() if transform is not None else None
        self.is_bookend = is_bookend

    @staticmethod
    def default_transform():
        return {
            "scale": Vector(1.0, 1.0),
            "translation": Vector(0,0),
            "rotation": 0,
            "opacity": 1.0
        }
    @staticmethod
    def transforms_similar(transform1, transform2):
        default_transform = KeyFrame.default_transform()
        for key in default_transform.keys():
            default_value = default_transform.get(key, None)
            if transform1.get(key, default_value) != transform2.get(key, default_value):
                return False
        return True
        

class Transformable:
    def __init__(self, sprite_manager):
        self.sprite_manager = sprite_manager
        self.render_order = -1.0
        self.custom_render_order = None
        self.use_keyframes = True

        self.transform_estimates = []
        self.follow_scale = False
        self.follow_rotation = False


        transform = copy.deepcopy(KeyFrame.default_transform())
  
        self.start_keyframe = KeyFrame(0, transform=transform, is_bookend=True)
        self.end_keyframe = KeyFrame(1, transform={}, is_bookend=True)
        
        self.keyframes = [self.start_keyframe, self.end_keyframe]

        self.transform_override = {}
        self.local_transform = {}
        self.global_transform = {}
        self.global_transform_matrix = np.eye(3)
        self.local_transform_matrix = np.eye(3)
        self.easing = "Quad Ease In Out"
        self.anchor_point = Vector(0,0) # the sprites local anchor point
        self.temp_anchor_point = Vector(0,0) # used as a temporary anchor point when dragging anchor_point
        self.temp_opacity = None
        


        self.parent = None
        self.children = []
    

    def set_render_order(self, render_order:float):
        self.custom_render_order = render_order
        self.update_render_order()

    def update_render_order(self, offset:float=0):
        if self.custom_render_order is not None:
            self.render_order = self.custom_render_order
        else:    
            parent_render_order = self.parent.render_order if self.parent else -1.0
            self.render_order = parent_render_order + 1.0 + offset
        for idx, child in enumerate(self.children):
            child.update_render_order(idx/100.0)

    def set_parent(self, parent):   
        if parent in parent.children or parent == self:
            return
            
        # Check if parent is a child of this sprite
        current = parent
        while current is not None:
            if current in self.children:
                print("parent is a child of this sprite")
                return
            current = current.parent
        
        prev_parent = self.parent
        if prev_parent is not None and self in prev_parent.children:
            prev_parent.children.remove(self)

        # Store current global position before changing parent
        global_pos = self.get_position()
        
        if parent is None:
            self.parent = None
            self.render_order = -1
            # Maintain global position when becoming root
            self.set_position(global_pos)
        else:
            # Update parent reference before calculating new position
            self.parent = parent
            if self not in parent.children:
                parent.children.append(self)
            
            # Convert global position to new parent's local space
            new_local_pos = parent.global_to_local(global_pos)
            self.set_position(new_local_pos, local=True, frame_index=self.start_keyframe.frame_index)

        self.parent.update_render_order()
        
        # Update transform hierarchy
        self.update_transform()

    
    def set_position(self, position:Vector, local:bool=False, frame_index:int=None):
        if not local and self.parent is not None:
            # First convert current position back to global space
            current_global = self.get_position(local=False)
            if current_global == position:
                return  # Already at desired position
                
            # Convert desired global position to parent's local space
            parent_local_pos = self.parent.global_to_local(position)
            
        
            # Calculate how much the anchor point shifts the sprite in rotated space
            rotation = -self.get_rotation(local=True)
            
            scale = self.get_scale(local=True)
                
            rotated_anchor = (self.anchor_point * (scale)).rotate(rotation) 
            anchor_offset = rotated_anchor - self.anchor_point
            
            # Adjust the position to compensate for the anchor point shift
            parent_local_pos += anchor_offset
         
            position = parent_local_pos
        
        if position != self.get_position(local=True):
            self.set_local_transform("translation", position, frame_index)

    def set_rotation(self, angle:float, local:bool=False, frame_index:int=None):
        if not local and self.parent is not None:
            current_global = self.get_rotation(local=False)
            if current_global == angle:
                return  # Already at desired rotation
            parent_angle = self.parent.get_rotation(local=False)
            angle -= parent_angle


        self.set_local_transform("rotation", angle, frame_index)

    def set_scale(self, scale:Vector, local:bool=False, frame_index:int=None):
        
        if not local and self.parent is not None:
            current_global = self.get_scale(local=False)
            if current_global == scale:
                return  # Already at desired scale
            parent_scale = self.parent.get_scale()
            scale /= parent_scale

        
        self.set_local_transform("scale", scale, frame_index)

    def set_opacity(self, opacity:float, frame_index:int=None):
        self.set_local_transform("opacity", opacity, frame_index)

    # normalized anchor point is between -1 and 1
    def set_anchor_point_normalized(self, new_anchor:Vector):
        half_size = self.true_size / 2
        new_anchor = new_anchor * half_size
        self.set_anchor_point(new_anchor)

    def set_anchor_point(self, new_anchor: Vector):
        vec_zero = Vector(0,0)
        old_pos = self.local_to_global(vec_zero)
        
        self.anchor_point = new_anchor
        self.temp_anchor_point = self.anchor_point

        self.update_transform()

        delta = self.local_to_global(vec_zero) - old_pos
 
        self.set_position(self.get_position() - delta)


    def get_position(self, local:bool=False)->Vector:
        return self.get_translation(local)
    def get_translation(self, local:bool=False)->Vector:
        transform = self.local_transform if local or not self.parent else self.global_transform
        return Vector(self.transform_override.get("translation", transform.get("translation", Vector(0,0)))).round()
    def get_rotation(self, local:bool=False)->float:
        transform = self.local_transform if local or not self.parent else self.global_transform
        rot = self.transform_override.get("rotation", transform.get("rotation", 0)) 
        add_rot = 0
        if self.follow_rotation:
            idx = self.sprite_manager.current_frame_index
            if idx < len(self.transform_estimates):
                add_rot, _ = self.transform_estimates[idx]
            else:
                print("no transform estimates", idx)
        return rot - add_rot
    def get_scale(self, local:bool=False)->Vector:
        transform = self.local_transform if local or not self.parent else self.global_transform
        scale_val = Vector(self.transform_override.get("scale", transform.get("scale", Vector(1.0, 1.0))))
        mul_scale = Vector(1,1)
        if self.follow_scale:
            idx = self.sprite_manager.current_frame_index   
            if idx < len(self.transform_estimates):
                _, mul_scale = self.transform_estimates[idx]
        return scale_val * mul_scale
    def get_opacity(self, local:bool=False)->float:
        opacity_mod = 1.0
        if not local and self.parent is not None:
            parent = self.parent
            while parent is not None:
                opacity_mod *= parent.get_opacity(local=True)
                parent = parent.parent
        opacity = self.transform_override.get("opacity", self.local_transform.get("opacity", 1.0)) 
        if self.temp_opacity != None:
            opacity = self.temp_opacity
        return opacity * opacity_mod
    def is_transformed(self)->bool:
        is_scaled = self.get_scale(local=True) != Vector(1,1)
        is_rotated = self.get_rotation(local=True) != 0
        is_translated = self.get_translation(local=True) != Vector(0,0)
        return is_scaled or is_rotated or is_translated
    
    def local_to_global(self, local_point:Vector, transform_matrix:np.ndarray = None)->Vector:
        """
        Transforms a local point to a global point using the transformation matrix.
        
        :param local_point: Tuple (x, y) representing the local point.
        :param transform_matrix: 3x3 global transformation matrix.
        :return: Tuple (x_global, y_global) representing the global point.
        """
        # Convert local point to homogeneous coordinates
        local_point_homogeneous = np.array([local_point[0], local_point[1], 1])
        
        if transform_matrix is None:
            transform_matrix = self.global_transform_matrix
        # Transform the local point to global space
        global_point_homogeneous = transform_matrix @ local_point_homogeneous
        
        # Extract and return the x and y components
        x_global, y_global = global_point_homogeneous[:2]
        return Vector(x_global, y_global).round()

    def global_to_local(self, global_point:Vector, transform_matrix:np.ndarray = None)->Vector:
        """
        Transforms a global point to a local point using the inverse of the transformation matrix.
        
        :param global_point: Tuple (x, y) representing the global point.
        :param transform_matrix: 3x3 global transformation matrix.
        :return: Tuple (x_local, y_local) representing the local point.
        """
        # Convert global point to homogeneous coordinates
        global_point_homogeneous = np.array([global_point[0], global_point[1], 1])

        if transform_matrix is None:
            transform_matrix = self.global_transform_matrix
        
        # Compute the inverse of the transformation matrix
        inverse_matrix = np.linalg.inv(transform_matrix)
        
        # Transform the global point to the local space
        local_point_homogeneous = inverse_matrix @ global_point_homogeneous
        
        # Extract and return the x and y components
        x_local, y_local = local_point_homogeneous[:2]
        return Vector(x_local, y_local)
    

    def get_or_add_keyframe(self, frame_index:int=None):
        keyframe = self.keyframe_for_index(frame_index)
        if not keyframe:
            transform = self.local_transform.copy()
            default_transform = KeyFrame.default_transform()
            for key in default_transform.keys():
                if key not in transform:
                    transform[key] = default_transform[key]
            
            self.sprite_manager.has_new_keyframes = True
            self.sprite_manager.fx.api.should_refresh_timeline = True
        
            keyframe = KeyFrame(frame_index, transform)
            self.keyframes.append(keyframe)
        
        return keyframe
            
    
    def set_local_transform(self, key:str, value:any, frame_index:int=None):
        self.local_transform[key] = value

        if not self.use_keyframes:
            return
        
        if frame_index is None:
            frame_index = self.sprite_manager.current_frame_index
        
        
        # Find existing keyframe at this index
        keyframe = self.get_or_add_keyframe(frame_index)
        
        
        # Set the transform value
        keyframe.transform[key] = value

        self.sprite_manager.fx.api.should_refresh_frame = True


    def keyframe_for_index(self, frame_index:int):

        for keyframe in self.keyframes:
            if keyframe.frame_index == frame_index:
                return keyframe
        return None
    
    def update_keyed_transform(self, frame_info:FrameInfo):
        
        frame_index = frame_info.index
        total_frames = frame_info.total_frames
        # Sort keyframes by frame index
        sorted_keyframes = sorted([kf for kf in self.keyframes 
                                 if self.start_keyframe.frame_index <= kf.frame_index <= self.end_keyframe.frame_index], 
                                key=lambda kf: kf.frame_index)

        

        # Find surrounding keyframes for each transform property

        default_transform = KeyFrame.default_transform()
        interpolated_transform = copy.deepcopy(default_transform)
        keys = default_transform.keys()
        if len(sorted_keyframes) > 0:
            for key in keys:
                prev_keyframe = None
                next_keyframe = None
                
                for kf in sorted_keyframes:
                    if key not in kf.transform:
                        continue

                    if kf.frame_index <= frame_index:
                        prev_keyframe = kf
                    elif kf.frame_index > frame_index:
                        next_keyframe = kf
                        break

                # Use default transform at frame 0 if no previous keyframe
                if not prev_keyframe:
                    prev_keyframe = KeyFrame(0)
                    prev_keyframe.transform[key] = next_keyframe.transform[key] if next_keyframe else default_transform[key]
              

                # Use same transform as previous keyframe at final frame if no next keyframe 
                if not next_keyframe and prev_keyframe is not None:
                    next_keyframe = KeyFrame(total_frames)
                    next_keyframe.transform[key] = prev_keyframe.transform.get(key, default_transform[key])

                # Calculate interpolation factor
                num_frames = next_keyframe.frame_index - prev_keyframe.frame_index
                current_frame = frame_index - prev_keyframe.frame_index
                t = current_frame / max(1, num_frames)
                t_eased = Easing.ease(self.easing, t)

                # Interpolate values
                start_val = prev_keyframe.transform.get(key)
                end_val = next_keyframe.transform.get(key)
                
                if isinstance(start_val, tuple):
                    # Interpolate tuple values (scale, position etc)
                    interpolated_transform[key] = tuple(
                        start + (end - start) * t_eased
                        for start, end in zip(start_val, end_val)
                    )
                else:
                    # Interpolate single values (rotation, opacity etc)
                    interpolated_transform[key] = start_val + (end_val - start_val) * t_eased

        self.local_transform.update(interpolated_transform) 

        

    def update_transform(self) -> dict:        

        self.local_transform_matrix = self.transform_matrix(
            self.get_scale(local=True), 
            self.get_rotation(local=True), 
            self.get_translation(local=True) , 
            self.anchor_point
        )

        
        if self.parent is not None:
            self.global_transform_matrix = self.parent.global_transform_matrix @ self.local_transform_matrix
            self.global_transform = self.decompose_transform(self.global_transform_matrix)
          
          
        else:
            self.global_transform_matrix = self.local_transform_matrix
            self.global_transform = self.decompose_transform(self.global_transform_matrix)
        
        
            
            # Adjust the global position to account for pivot offset during rotation
            # if self.get_rotation(local=True) != 0:
            #     # Get the rotated pivot offset
                
                
            #     rotated_pivot = self.anchor_point.rotate(self.get_rotation(local=True))
            #     print(self.get_rotation(local=True))
                
            #     # Adjust the global position by the difference between rotated and original pivot
            #     pivot_offset = rotated_pivot - self.anchor_point
            #     current_pos = Vector(self.global_transform["translation"])
            #     anchor_pos = self.local_to_global(self.anchor_point)
            #     self.global_transform["translation"] = tuple(current_pos - anchor_pos)
            #     print(self.global_transform)

        for child in self.children:
            child.update_transform()
            


    def transform_matrix(self, scale, rotation, translation, pivot):
        """
        Creates a transformation matrix with a pivot point.
        
        :param scale: Tuple (sx, sy) representing scale factors.
        :param rotation: Rotation in degrees.
        :param translation: Tuple (tx, ty) for translation.
        :param pivot: Tuple (px, py) for the pivot point.
        :return: 3x3 transformation matrix.
        """
        sx, sy = scale
        theta = np.radians(-rotation)
        tx, ty = translation
        px, py = pivot
        
        # Translate to pivot (move pivot point to the origin)
        T_to_pivot = np.array([[1, 0, -px],
                            [0, 1, -py],
                            [0, 0, 1]])
        
        # Scale
        S = np.array([[sx, 0,  0],
                    [0,  sy, 0],
                    [0,  0,  1]])
        
        # Rotate
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
        
        # Translate back from pivot (restore pivot point to its position)
        T_from_pivot = np.array([[1, 0, px],
                                [0, 1, py],
                                [0, 0, 1]])
        
        # Translate to world position
        T_translation = np.array([[1, 0, tx],
                                [0, 1, ty],
                                [0, 0, 1]])
        
        # Combine transformations
        return T_translation @ (T_from_pivot @ (R @ (S @ T_to_pivot)))
    
    def decompose_transform(self, matrix):
        """
        Decomposes a 2D transformation matrix into translation, rotation, and scale.
        Handles negative scale values correctly.
        
        :param matrix: 3x3 transformation matrix.
        :return: Dict with translation, rotation, and scale components.
        """
        # Extract translation
        tx, ty = matrix[0, 2], matrix[1, 2]
        translation = (tx, ty)
        
        # Extract scale
        sx = np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
        sy = np.sqrt(matrix[0, 1]**2 + matrix[1, 1]**2)
        
        # Determine scale signs based on determinant
        det = np.linalg.det(matrix[:2, :2])
        if det < 0:
            sy = -sy
            
        scale = (sx, sy)
        
        # Extract rotation
        rotation = np.arctan2(matrix[1, 0], matrix[0, 0])
        rotation_degrees = -np.degrees(rotation)
        
        return {"translation": translation, "rotation": rotation_degrees, "scale": scale}
    
    def test_negative_scale(self):
        # Test point
        test_point = Vector(100, 100)
        
        # Test with negative scale
        pivot = Vector(50, 50)
        rotation = 45
        scale = Vector(-1, -1)  # Negative scale
        translation = Vector(200, 200)
        
        matrix = self.transform_matrix(
            translation=translation,
            rotation=rotation,
            scale=scale,
            pivot=pivot
        )
        
        # Decompose and verify
        transform = self.decompose_transform(matrix)
        print(f"Original scale: {scale}")
        print(f"Decomposed scale: {transform['scale']}")
        print(f"Original rotation: {rotation}")
        print(f"Decomposed rotation: {transform['rotation']}")
    
    def test_transform(self):
        # Create a test point
        test_point = Vector(100, 100)
        
        # Set up transform parameters
        pivot = Vector(50, 50)  # Pivot point
        rotation = 45  # 45 degree rotation
        scale = Vector(1, 1)  # No scaling
        translation = Vector(200, 200)  # Translate by 200,200
        
        # Create transform matrix with these parameters
        matrix = self.transform_matrix(
            translation=translation,
            rotation=np.radians(rotation), 
            scale=scale,
            pivot=pivot
        )
        
        # Store matrix temporarily for testing
        old_matrix = self.global_transform_matrix
        self.global_transform_matrix = matrix
        
        # Test transforming point from local to global space
        global_point = self.local_to_global(test_point)
        print(f"Local point {test_point} transformed to global {global_point}")
        
        # Test transforming back to local space
        local_point = self.global_to_local(global_point)
        print(f"Global point {global_point} transformed back to local {local_point}")
        
        # Verify round trip is accurate
        diff = (test_point - local_point).length()
        print(f"Round trip difference: {diff}")
        
        # Restore original matrix
        self.global_transform_matrix = old_matrix