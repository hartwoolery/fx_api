import cv2
import numpy as np
from fx_api.utils.vector import Vector


class UIButton:
    def __init__(self, sprite, transform_type:str):
        
        self.sprite = sprite
        self.type = transform_type
        self.location = Vector(0,0)
        self.radius = 3
        self.thickness = -1
        if self.type == "anchor":
            self.color = sprite.object_info.color
            self.radius = 3
            self.thickness = -1
        elif self.type == "scale":
            self.color = (255, 255, 255)
            self.radius = 3
        elif self.type == "rotation":
            self.color = (255, 255, 255)
            
            self.radius = 6
        elif self.type == "scale_x":
            self.color = (255, 255, 255)
        elif self.type == "scale_y":
            self.color = (255, 255, 255)
        elif self.type == "clone":
            self.color = (160, 160, 160)
            self.thickness = 1
            self.radius = 7
        elif self.type == "delete":
            self.color = (160, 160, 160)
            self.thickness = -1
            self.radius = 10
        else:
            self.color = (255, 0, 255)

        self.radius += 2

        
    def draw(self, new_location:Vector, background:np.ndarray):
        self.location = Vector(new_location)
        radius_factor = 2 if self.sprite.sprite_manager.current_button == self else 1
        center = self.location.round()

        circle_color = self.color#self.sprite.object_info.color if self.type == "rotation" and hasattr(self.sprite, "object_info") else self.color
        

        if not (self.type == "clone"):
            cv2.circle(background, center, self.radius * radius_factor, circle_color, self.thickness, cv2.LINE_AA)
        # Draw crosshair for anchor button
        if self.type == "anchor":
            # Draw horizontal line
            cv2.line(background, 
                    (int(self.location[0] - self.radius), int(self.location[1])),
                    (int(self.location[0] + self.radius), int(self.location[1])),
                    self.color, 1, cv2.LINE_AA)
            # Draw vertical line  
            cv2.line(background,
                    (int(self.location[0]), int(self.location[1] - self.radius)),
                    (int(self.location[0]), int(self.location[1] + self.radius)),
                    self.color, 1, cv2.LINE_AA)
            
            if self.sprite.sprite_manager.current_button == self:
                # Position text above the button
                text_pos = (int(self.location[0]), int(self.location[1] - self.radius - 8))
                text = "ANCHOR" if self.type == "anchor" else ""
                
                # Get text size to center it
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
                
                # Adjust x position to center text
                text_pos = (text_pos[0] - text_width//2, text_pos[1])
                
                # Draw text
                cv2.putText(background, text, text_pos, font, font_scale, self.color, 1, cv2.LINE_AA)
        
        elif self.type == "rotation":
            r = (self.radius * 2 * radius_factor)
            cv2.ellipse(background, center, (r , r), -self.sprite.get_rotation(), 180, 450, self.color, 1)

        elif self.type == "clone":
            # Get rotation angle for the clone symbol
            angle = -self.sprite.get_rotation()
            scale = self.sprite.get_scale()
            if scale.x * scale.y < 0:  # If exactly one scale component is negative flip the rotation
                angle += 180
            r = self.radius * radius_factor * 1.25
            offset = Vector(r*0.7, -r*0.7)
            # Rotate points
            rot_matrix = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            
            offset_rotated = Vector(np.dot(rot_matrix, [offset.x, offset.y]))

            cv2.circle(background, (center-offset_rotated*0.5).round(), self.radius * radius_factor, self.color, -1, cv2.LINE_AA)
            cv2.circle(background, (center+offset_rotated*0.5).round(), self.radius * radius_factor, (255, 255, 255), -1, cv2.LINE_AA)

        
            
        elif self.type == "delete":
            # Get rotation angle for the X symbol
            angle = -self.sprite.get_rotation()
            r = self.radius * radius_factor * 0.5
            
            # Calculate rotated points for the X symbol
            # First diagonal line (top-left to bottom-right)
            x1 = self.location.x - r * np.cos(np.radians(angle - 45))
            y1 = self.location.y - r * np.sin(np.radians(angle - 45))
            x2 = self.location.x + r * np.cos(np.radians(angle - 45))
            y2 = self.location.y + r * np.sin(np.radians(angle - 45))
            
            # Second diagonal line (top-right to bottom-left)
            x3 = self.location.x - r * np.cos(np.radians(angle + 45))
            y3 = self.location.y - r * np.sin(np.radians(angle + 45))
            x4 = self.location.x + r * np.cos(np.radians(angle + 45))
            y4 = self.location.y + r * np.sin(np.radians(angle + 45))
            
            # Draw the rotated X
            cv2.line(background,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255,255,255), 2, cv2.LINE_AA)
            cv2.line(background,
                    (int(x3), int(y3)),
                    (int(x4), int(y4)), 
                    (255,255,255), 2, cv2.LINE_AA)

