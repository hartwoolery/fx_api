from enum import IntEnum
import copy
from fx_api.sprites.transforms import KeyFrame
from fx_api.utils.vector import Vector

class HistoryType(IntEnum):
    ADD_SPRITE = 1
    DELETE_SPRITE = 2
    MODIFY_TRANSFORM = 3 # transform handle scale, rotation, translation, opacity
    ADD_KEYFRAME = 4
    DELETE_KEYFRAME = 5
    REPARENT = 6
    
class HistoryState(dict):
    def __init__(self, state:dict):
        super().__init__(state)
        self.type = state["type"]

    def __str__(self):
        return self.print_state()
    
    def print_state(self):
        """Print a human readable description of a history state"""
        if self.type == HistoryType.ADD_SPRITE:
            return "Added sprite"
        elif self.type == HistoryType.DELETE_SPRITE:
            return "Deleted sprite" 
        elif self.type == HistoryType.MODIFY_TRANSFORM:
            return "Modified transform"
        elif self.type == HistoryType.ADD_KEYFRAME:
            return "Added keyframe"
        elif self.type == HistoryType.DELETE_KEYFRAME:
            return "Deleted keyframe"
        elif self.type == HistoryType.REPARENT:
            return "Reparented sprite"
        else:
            return "Unknown state"

class HistoryManager:
    def __init__(self, sprite_manager):
        self.sprite_manager = sprite_manager
        self.undo_stack = []
        self.redo_stack = []
        self.current_state = None

    def undo(self):
        if len(self.undo_stack) > 0:
            self.redo_stack.append(self.undo_stack.pop())
            self.current_state = self.redo_stack[-1]
            self.state_changed(True)
        
    def redo(self):
        if len(self.redo_stack) > 0:
            self.undo_stack.append(self.redo_stack.pop())
            self.current_state = self.undo_stack[-1]
            self.state_changed(False)

    def state_is(self, *types: HistoryType):
        return self.current_state is not None and self.current_state.type in types

    def clear_history(self):
        self.undo_stack = []
        self.redo_stack = []
        self.current_state = None


    def add_history(self, state:HistoryState):
        self.undo_stack.append(state)
        self.current_state = state
        #print(f"added history state {self.current_state}")
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def state_changed(self, was_undone:bool):
        if (self.state_is(HistoryType.DELETE_SPRITE) and was_undone) or \
            (self.state_is(HistoryType.ADD_SPRITE) and not was_undone):
            deleted_sprite = self.current_state["sprite"]
            if deleted_sprite not in self.sprite_manager.sprites:
                self.sprite_manager.sprites.append(deleted_sprite)
            deleted_sprite.set_parent(deleted_sprite.parent)
            self.sprite_manager.select_sprite(deleted_sprite)
        elif (self.state_is(HistoryType.ADD_SPRITE) and was_undone) or \
            (self.state_is(HistoryType.DELETE_SPRITE) and not was_undone): 
            added_sprite = self.current_state["sprite"]
            self.sprite_manager.sprites.remove(added_sprite)
            added_sprite.parent.children.remove(added_sprite)
            added_sprite.parent.update_render_order()      
            if self.sprite_manager.selected_sprite == added_sprite:
                self.sprite_manager.selected_sprite = None
        elif self.state_is(HistoryType.MODIFY_TRANSFORM):
            frame_index = self.current_state["frame_index"]
            sprite = self.current_state["sprite"]
            
            transform = self.current_state["start_transform"] if was_undone else self.current_state["end_transform"]
            # Find keyframe at this frame index
            matching_keyframe = sprite.keyframe_for_index(frame_index)
            
            if matching_keyframe is not None:
                matching_keyframe.transform = copy.deepcopy(transform)

        
        elif self.state_is(HistoryType.ADD_KEYFRAME):
            keyframe = self.current_state["keyframe"]
            sprite = self.current_state["sprite"]
            if was_undone:
                sprite.keyframes.remove(keyframe)
            else:
                sprite.keyframes.append(keyframe)
        elif self.state_is(HistoryType.DELETE_KEYFRAME):
            keyframe = self.current_state["keyframe"]
            sprite = self.current_state["sprite"]
            if was_undone:
                sprite.keyframes.append(keyframe)
            else:
                sprite.keyframes.remove(keyframe)
        




