
import numpy as np
import supervision as sv
from fx_api.utils.vector import Vector
class FrameInfo:
    def __init__(self, frame: np.ndarray, index: int, total_frames: int, time: float = 0,  delta_time: float = 0):
        self.frame = frame
        self.render_buffer = None
        self.index = index
        self.time = time
        self.total_frames = total_frames
        self.delta_time = delta_time
        self.detections = None

class ObjectInfo:
    def __init__(self, id: int, color: tuple[int, int, int], start_frame: int=None, end_frame: int=None):
        self.id = id
        self.color = color
        self.start_frame = start_frame
        self.end_frame = end_frame

class FXAPI:
    
    def __init__(self):
        super().__init__()
        
    def get_total_objects(self) -> int:
        """
        Returns the total number of objects in the video.
        """
        return 0 
    
    def get_total_frames(self) -> int:
        """
        Returns the total number of frames in the video.
        """
        return 0

    def get_color_palette(self) -> list[tuple[int, int, int]]:
        """
        Returns the color palette for the video.
        """
        return []
    
    def get_resolution(self) -> Vector:
        """
        Returns the current resolution of the video.
        """
        return Vector(720, 1280) 
    
    def get_pose(self, frame_info: FrameInfo) -> dict:
        """
        Returns the pose for the given frame.
        """
        return None

    def get_masks(self, frame_index: int) -> sv.Detections:
        """
        Returns the mask for the given frame.
        """
        return None
    
    def get_objects(self) -> list[ObjectInfo]:
        """
        Returns the list of objects in the video.
        """
        return [] 
    
    def estimate_transforms(self, object_id:int, finished_callback):
        """
        Estimates the transforms for the given object.
        """
        pass
    
    def get_inpainting(self, frame_info:FrameInfo) -> np.ndarray:
        """
        Returns the inpainting for the given frame.
        """
        return None
    
    def open_file_dialog(self, title:str, filter:str) -> str:
        """
        Opens a file dialog and returns the path of the selected file.
        """
        return None
    
    def prompt_user(self, message:str) -> bool:
        """
        Prompts the user with the given message and returns True if the user confirms.
        """
        return False
    
    def update_inspector(self):
        """
        Updates the inspector.  
        """
        pass
    
    def update_hint_label(self, text:str):
        """
        Updates the hint label in the UI.
        """
        pass

    def update_frame(self, frame_idx:int=None):
        """
        Updates the frame in the UI.
        """
        pass

    def update_timeline(self):
        """
        Renders the timeline in the UI.
        """
        pass

    def update_scene_tree(self):
        """
        Updates the scene tree in the UI.
        """
        pass
