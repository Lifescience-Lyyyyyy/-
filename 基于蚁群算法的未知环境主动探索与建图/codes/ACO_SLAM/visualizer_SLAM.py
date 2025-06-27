import pygame
import numpy as np
import os
import math 

try:
    from environment import UNKNOWN, FREE, OBSTACLE, LANDMARK
except ImportError:
    UNKNOWN, FREE, OBSTACLE, LANDMARK = 0, 1, 2, 3


BLACK = (0, 0, 0); WHITE = (255, 255, 255)
LIGHT_GREEN_FREE = (210, 240, 210); # Lighter green
DARK_GRAY_OBSTACLE = (0, 0, 10);   # Darker gray
GRAY_UNKNOWN = (180, 180, 180); # Lighter unknown

ROBOT_COLOR = (255, 0, 0)         # Red for robot's center
ROBOT_UNCERTAINTY_COLOR = (255, 100, 100) # Lighter red for uncertainty ellipse/circle
LANDMARK_COLOR = (0, 215, 0)      # Gold for landmark's center
LANDMARK_UNCERTAINTY_COLOR = (255, 255, 150) # Light yellow for uncertainty
PATH_COLOR = (0, 100, 255)        # Blue for path

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = env_width * cell_size
        self.screen_height = env_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("SLAM with ACO Exploration Demo")
        
        font_size_normal = max(12, int(cell_size * 0.9))
        font_size_small = max(10, int(cell_size * 0.6))
        try:
            self.font = pygame.font.SysFont(None, font_size_normal)
            self.font_small = pygame.font.SysFont(None, font_size_small)
        except pygame.error: 
             self.font = pygame.font.Font(None, font_size_normal)
             self.font_small = pygame.font.Font(None, font_size_small)

    def draw_probabilistic_map(self, prob_map):
        """Draws the map based on occupancy probabilities [0.0, 1.0]."""
        if prob_map is None: return

        map_img = np.zeros((prob_map.shape[0], prob_map.shape[1], 3), dtype=np.uint8)

        free_mask = prob_map < 0.5
        free_factor = (0.5 - prob_map[free_mask]) / 0.5 
        map_img[free_mask] = np.array(WHITE) * (1 - free_factor[:, np.newaxis]) + \
                             np.array(LIGHT_GREEN_FREE) * free_factor[:, np.newaxis]
        
        occ_mask = prob_map > 0.5
        occ_factor = (prob_map[occ_mask] - 0.5) / 0.5 
        map_img[occ_mask] = np.array(GRAY_UNKNOWN) * (1 - occ_factor[:, np.newaxis]) + \
                            np.array(DARK_GRAY_OBSTACLE) * occ_factor[:, np.newaxis]
        
        map_img[prob_map == 0.5] = GRAY_UNKNOWN

        map_surface = pygame.surfarray.make_surface(map_img.transpose(1, 0, 2))
        scaled_surface = pygame.transform.scale(map_surface, (self.screen_width, self.screen_height))
        self.screen.blit(scaled_surface, (0,0))
        
        # Draw grid lines on top
        for r in range(prob_map.shape[0]):
            pygame.draw.line(self.screen, (200,200,200), (0, r*self.cell_size), (self.screen_width, r*self.cell_size))
        for c in range(prob_map.shape[1]):
            pygame.draw.line(self.screen, (200,200,200), (c*self.cell_size, 0), (c*self.cell_size, self.screen_height))

    def draw_pose_with_uncertainty(self, believed_pos, covariance, color, uncertainty_color):
        """
        Draws a pose (robot or landmark) with its uncertainty ellipse.
        believed_pos: (r, c) tuple of the mean position.
        covariance: 2x2 numpy array representing the pose uncertainty.
        """
        center_x_px = believed_pos[1] * self.cell_size + self.cell_size / 2
        center_y_px = believed_pos[0] * self.cell_size + self.cell_size / 2

        
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_deg = -np.degrees(angle_rad) 
        radius_px = np.sqrt(np.max(eigenvalues)) * self.cell_size * 2.0 
        radius_px = max(self.cell_size * 0.5, radius_px) 
        max_radius_int = int(np.ceil(radius_px)) * 2
        if max_radius_int > 0:
            ellipse_surface = pygame.Surface((max_radius_int, max_radius_int), pygame.SRCALPHA)
            pygame.draw.circle(ellipse_surface, uncertainty_color + (80,), 
                               (max_radius_int // 2, max_radius_int // 2), int(radius_px))
            
            self.screen.blit(ellipse_surface, (center_x_px - max_radius_int//2, center_y_px - max_radius_int//2))
        
        pygame.draw.circle(self.screen, color, (center_x_px, center_y_px), int(self.cell_size * 0.25))

    def draw_robot_path(self, path_history, path_color=PATH_COLOR, thickness_factor=0.08):
        if len(path_history) < 2: return
        pixel_path = []
        for r_p, c_p in path_history:
            px = c_p * self.cell_size + self.cell_size // 2
            py = r_p * self.cell_size + self.cell_size // 2
            pixel_path.append((px,py))
        if len(pixel_path) > 1:
            pygame.draw.lines(self.screen, path_color, False, pixel_path, max(1, int(self.cell_size * thickness_factor)))
            
    def draw_text(self, text, position, color=BLACK, small=False):
        font_to_use = self.font_small if small else self.font
        try:
            text_surface = font_to_use.render(text, True, color, (255,255,255,180)) 
            self.screen.blit(text_surface, position)
        except pygame.error as e: print(f"Warning: Pygame font rendering error: {e}")
    
    def handle_events_simple(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return False
        return True
    
    def update_display(self): pygame.display.flip()
    def clear_screen(self): self.screen.fill(WHITE)
    def quit(self): pygame.quit()
    
