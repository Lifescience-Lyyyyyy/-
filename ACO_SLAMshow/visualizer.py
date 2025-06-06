# visualizer.py
import pygame
import numpy as np
import os

try:
    from environment import UNKNOWN, FREE, OBSTACLE
except ImportError:
    UNKNOWN, FREE, OBSTACLE = 0,1,2 

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY_UNKNOWN = (128, 128, 128)
LIGHT_GREEN_FREE = (144, 238, 144) 
DARK_GRAY_OBSTACLE = (60, 60, 60)
RED_ROBOT = (255, 0, 0) 
BLUE_PATH = (0, 0, 255)
ORANGE_GOAL = (255, 165, 0) 
GREEN_START = (0, 200, 0)
PURPLE_ANT_PATH_ITER_BASE = (128, 0, 128) 
LIGHT_BLUE_ANT_PATH_ITER_BASE = (173, 216, 230)
YELLOW_ANT_PATH_ITER_BASE = (255,255,0)
CYAN_ANT_PATH_ITER_BASE = (0,255,255)
PHEROMONE_COLOR_BASE = (30, 30, 180) 

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = env_width * cell_size
        self.screen_height = env_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("ACO Known Map Pathfinding Demo")
        font_size_normal = max(12, int(cell_size * 0.8))
        font_size_small = max(10, int(cell_size * 0.6))
        try:
            self.font = pygame.font.SysFont(None, font_size_normal)
            self.font_small = pygame.font.SysFont(None, font_size_small)
        except pygame.error: 
             self.font = pygame.font.Font(None, font_size_normal)
             self.font_small = pygame.font.Font(None, font_size_small)

    def draw_static_map(self, static_map_grid): 
        if static_map_grid is None: return
        for r in range(static_map_grid.shape[0]):
            for c in range(static_map_grid.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = WHITE 
                cell_state = static_map_grid[r, c]
                if cell_state == FREE: color = LIGHT_GREEN_FREE
                elif cell_state == OBSTACLE: color = DARK_GRAY_OBSTACLE
                elif cell_state == UNKNOWN: color = GRAY_UNKNOWN
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1) 

    def draw_pheromone_map_overlay(self, pheromone_map_data, static_map_grid_ref, pher_max_val_display=None):
        if pheromone_map_data is None: return
        
        if pher_max_val_display is None or pher_max_val_display <= 1e-6:
            non_obstacle_pheromones = pheromone_map_data[static_map_grid_ref != OBSTACLE]
            if non_obstacle_pheromones.size > 0 and np.max(non_obstacle_pheromones) > 1e-6:
                pher_max_val_display = np.max(non_obstacle_pheromones)
            else: 
                pher_max_val_display = 1.0 
        
        min_alpha = 10  
        max_alpha = 160 

        for r in range(pheromone_map_data.shape[0]):
            for c in range(pheromone_map_data.shape[1]):
                if static_map_grid_ref[r,c] == OBSTACLE: continue 

                pher_value = pheromone_map_data[r,c]
                norm_pher = np.clip(pher_value / pher_max_val_display, 0, 1) 
                alpha = int(min_alpha + norm_pher * (max_alpha - min_alpha))
                
                if alpha > min_alpha + 5: 
                    cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    color_intensity = int(norm_pher * 155) + 100 
                    pheromone_color_instance = (PHEROMONE_COLOR_BASE[0], PHEROMONE_COLOR_BASE[1], np.clip(color_intensity,0,255))
                    cell_surface.fill(pheromone_color_instance + (alpha,))
                    self.screen.blit(cell_surface, (c * self.cell_size, r * self.cell_size))

    def draw_positions(self, positions_list, color, radius_factor=0.35, is_path=False, line_thickness_factor=0.1):
        if not positions_list: return
        
        path_color = color[:3] if len(color) == 4 else color
        alpha_val = color[3] if len(color) == 4 and is_path else 255
        thickness = max(1, int(self.cell_size * line_thickness_factor))

        if is_path and len(positions_list) > 1:
            pixel_path = []
            for r_p, c_p in positions_list:
                px = c_p * self.cell_size + self.cell_size // 2
                py = r_p * self.cell_size + self.cell_size // 2
                pixel_path.append((px, py))
            
            if alpha_val < 255 and len(pixel_path) > 1: 
                for i in range(len(pixel_path) - 1):
                    line_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                    pygame.draw.line(line_surf, path_color + (alpha_val,), pixel_path[i], pixel_path[i+1], thickness)
                    self.screen.blit(line_surf, (0,0))
            elif len(pixel_path) > 1: 
                pygame.draw.lines(self.screen, path_color, False, pixel_path, thickness)
        elif not is_path : 
             for r_p, c_p in positions_list:
                center_x = c_p * self.cell_size + self.cell_size // 2
                center_y = r_p * self.cell_size + self.cell_size // 2
                pygame.draw.circle(self.screen, path_color, (center_x, center_y), int(self.cell_size * radius_factor))

    def draw_ant_paths_iteration_known_map(self, static_map_grid, pheromone_map_for_display, 
                                           ant_paths_this_iter, start_pos, goal_pos, 
                                           iteration_num, total_iterations_planner,
                                           pher_display_max=None): # Renamed total_iterations
        self.clear_screen()
        self.draw_static_map(static_map_grid)
        if pheromone_map_for_display is not None:
             self.draw_pheromone_map_overlay(pheromone_map_for_display, static_map_grid, pher_max_val_display=pher_display_max)
        
        self.draw_positions([start_pos], GREEN_START, radius_factor=0.4)
        self.draw_positions([goal_pos], ORANGE_GOAL, radius_factor=0.4)

        path_base_colors = [PURPLE_ANT_PATH_ITER_BASE, LIGHT_BLUE_ANT_PATH_ITER_BASE, 
                            YELLOW_ANT_PATH_ITER_BASE, CYAN_ANT_PATH_ITER_BASE]
        ant_path_alpha = 100 

        for idx, ant_path in enumerate(ant_paths_this_iter): # ant_paths_this_iter is now paths for ONE iteration
            if ant_path and len(ant_path) > 1:
                current_color_rgb = path_base_colors[idx % len(path_base_colors)]
                self.draw_positions(ant_path, current_color_rgb + (ant_path_alpha,), is_path=True, line_thickness_factor=0.05)
        
        self.draw_text(f"ACO Known Map - Ant Iter: {iteration_num + 1}/{total_iterations_planner}", 
                       (10, self.screen_height - 30 if self.screen_height > 60 else 10), color=BLACK)
        self.update_display()

    def draw_final_path(self, static_map_grid, pheromone_map_to_display, final_path, start_pos, goal_pos, pher_display_max=None):
        self.clear_screen()
        self.draw_static_map(static_map_grid)
        if pheromone_map_to_display is not None:
             self.draw_pheromone_map_overlay(pheromone_map_to_display, static_map_grid, pher_max_val_display=pher_display_max)
        
        self.draw_positions([start_pos], GREEN_START, radius_factor=0.45)
        self.draw_positions([goal_pos], ORANGE_GOAL, radius_factor=0.45)
        if final_path:
            self.draw_positions(final_path, BLUE_PATH, is_path=True, line_thickness_factor=0.15) 
        
        self.draw_text("Final Path by ACO", (10,10), color=BLACK)
        if final_path:
             self.draw_text(f"Path Length: {len(final_path)-1}", (10,30 if self.cell_size > 10 else 20), color=BLACK)
        else:
             self.draw_text("No path found to goal.", (10,30 if self.cell_size > 10 else 20), color=BLACK)
        self.update_display()

    def draw_text(self, text, position, color=BLACK, small=False):
        font_to_use = self.font_small if small else self.font
        try:
            text_surface = font_to_use.render(text, True, color)
            self.screen.blit(text_surface, position)
        except pygame.error as e: print(f"Warning: Pygame font rendering error: {e}")
    
    def update_display(self): pygame.display.flip()
    def clear_screen(self): self.screen.fill(WHITE)
    def handle_events_simple(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return False
        return True
    def quit(self): pygame.quit()