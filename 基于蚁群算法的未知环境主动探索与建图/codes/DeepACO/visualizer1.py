import pygame
import numpy as np
import os

try:
    from environment import UNKNOWN, FREE, OBSTACLE
except ImportError:
    UNKNOWN, FREE, OBSTACLE = 0,1,2 

# --- Colors (确保这些颜色定义与您当前使用的一致) ---
BLACK = (0, 0, 0); WHITE = (255, 255, 255); GRAY_UNKNOWN = (128, 128, 128)
LIGHT_GREEN_FREE = (144, 238, 144); DARK_GRAY_OBSTACLE = (60, 60, 60)
ROBOT_COLORS=RED_ROBOT = (255, 0, 0) 
BLUE_PATH = (0, 0, 255)
ORANGE_GOAL = (255, 165, 0) 
GREEN_START = (0, 200, 0)
PURPLE_ANT_PATH_ITER_BASE = (128, 0, 128) 
LIGHT_BLUE_ANT_PATH_ITER_BASE = (173, 216, 230)
YELLOW_ANT_PATH_ITER_BASE = (255,255,0)
CYAN_ANT_PATH_ITER_BASE = (0,255,255)
PHEROMONE_COLOR_BASE = (30, 30, 180) 
PATH_TRAIL_COLOR = (100, 100, 255, 100) # For multi-robot trail

CLUSTER_COLORS = [
    (255, 0, 0, 150),    # Red
    (0, 255, 0, 150),    # Green
    (0, 0, 255, 150),    # Blue
    (255, 255, 0, 150),  # Yellow
    (255, 0, 255, 150),  # Magenta
    (0, 255, 255, 150),  # Cyan
    (128, 0, 0, 150),    # Maroon
    (0, 128, 0, 150),    # Dark Green
    (0, 0, 128, 150),    # Navy
    (128, 128, 0, 150),  # Olive
    (128, 0, 128, 150),  # Purple
    (0, 128, 128, 150),  # Teal
    (255, 165, 0, 150),  # Orange
    (210, 105, 30, 150), # Chocolate
    (106, 90, 205, 150)  # Slate Blue
]

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init() # Ensure Pygame is initialized
        self.cell_size = cell_size
        self.screen_width = env_width * cell_size
        self.screen_height = env_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Exploration / Pathfinding Demo") # Generic caption
        font_size_normal = max(12, int(cell_size * 0.8))
        font_size_small = max(10, int(cell_size * 0.6))
        try:
            self.font = pygame.font.SysFont(None, font_size_normal)
            self.font_small = pygame.font.SysFont(None, font_size_small)
        except pygame.error: 
             self.font = pygame.font.Font(None, font_size_normal)
             self.font_small = pygame.font.Font(None, font_size_small)

    def draw_grid_map(self, map_grid): 
        if map_grid is None: return
        for r in range(map_grid.shape[0]):
            for c in range(map_grid.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = WHITE 
                cell_state = map_grid[r, c]
                if cell_state == FREE: color = LIGHT_GREEN_FREE
                elif cell_state == OBSTACLE: color = DARK_GRAY_OBSTACLE
                elif cell_state == UNKNOWN: color = GRAY_UNKNOWN
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1) 

    def draw_pheromone_map_overlay(self, pheromone_map_data, static_map_grid_ref, pher_max_val_display=None):
        if pheromone_map_data is None: return
        if pher_max_val_display is None or pher_max_val_display <= 1e-6:
            non_obstacle_pheromones = pheromone_map_data[static_map_grid_ref != OBSTACLE] if static_map_grid_ref is not None else pheromone_map_data
            if non_obstacle_pheromones.size > 0 :
                max_val = np.max(non_obstacle_pheromones)
                if max_val > 1e-6: pher_max_val_display = max_val
                else: pher_max_val_display = 1.0
            else: pher_max_val_display = 1.0 
        if pher_max_val_display <= 1e-6 : pher_max_val_display = 1.0
        min_alpha = 10; max_alpha = 130 

        for r in range(pheromone_map_data.shape[0]):
            for c in range(pheromone_map_data.shape[1]):
                if static_map_grid_ref is not None and static_map_grid_ref[r,c] == OBSTACLE: continue 
                pher_value = pheromone_map_data[r,c]
                norm_pher = np.clip(pher_value / pher_max_val_display, 0, 1) 
                alpha = int(min_alpha + norm_pher * (max_alpha - min_alpha))
                if alpha > min_alpha + 3: 
                    cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    intensity = int(norm_pher * 100) + 50 
                    color_inst = (PHEROMONE_COLOR_BASE[0], PHEROMONE_COLOR_BASE[1], np.clip(intensity,0,255))
                    cell_surface.fill(color_inst + (alpha,))
                    self.screen.blit(cell_surface, (c * self.cell_size, r * self.cell_size))

    def draw_robots(self, robots_list): # For multi-robot
        if not robots_list: return
        for i, robot_obj in enumerate(robots_list):
            r, c = robot_obj.get_position() # Assumes robot_obj has get_position()
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            center_x = c * self.cell_size + self.cell_size // 2
            center_y = r * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), int(self.cell_size * 0.4))
            if hasattr(robot_obj, 'id'): # If robot has an ID attribute
                 id_surf = self.font_small.render(str(robot_obj.id), True, BLACK)
            else: # Fallback to index
                 id_surf = self.font_small.render(str(i), True, BLACK)
            self.screen.blit(id_surf, (center_x - id_surf.get_width()//2, center_y - id_surf.get_height()//2))

    def draw_robot_path(self, path_history, path_color=BLUE_PATH, thickness_factor=0.1): # For single robot path
        if len(path_history) < 2: return
        pixel_path = []
        for r_p, c_p in path_history:
            px = c_p * self.cell_size + self.cell_size // 2
            py = r_p * self.cell_size + self.cell_size // 2
            pixel_path.append((px,py))
        if len(pixel_path) > 1:
            pygame.draw.lines(self.screen, path_color, False, pixel_path, max(1, int(self.cell_size * thickness_factor)))


    def draw_robot_trails(self, robots_list): # For multi-robot trails
        if not robots_list: return
        for i, robot_obj in enumerate(robots_list):
            path_history = robot_obj.actual_pos_history
            if len(path_history) < 2: continue
            trail_color_base = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            
            # Create a surface for drawing transparent lines for this robot's trail
            trail_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            
            pixel_path = []
            for r_p, c_p in path_history:
                px = c_p * self.cell_size + self.cell_size // 2
                py = r_p * self.cell_size + self.cell_size // 2
                pixel_path.append((px,py))

            if len(pixel_path) > 1:
                # Draw lines with alpha on the temporary surface
                pygame.draw.lines(trail_surface, trail_color_base + (PATH_TRAIL_COLOR[3],), False, pixel_path, max(1, self.cell_size//12))
            
            self.screen.blit(trail_surface, (0,0)) # Blit the trail surface onto the main screen


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
                                           pher_display_max=None):
        self.clear_screen()
        self.draw_grid_map(static_map_grid)
        if pheromone_map_for_display is not None:
             self.draw_pheromone_map_overlay(pheromone_map_for_display, static_map_grid, pher_max_val_display=pher_display_max)
        
        if start_pos: self.draw_positions([start_pos], GREEN_START, radius_factor=0.4)
        if goal_pos: self.draw_positions([goal_pos], ORANGE_GOAL, radius_factor=0.4)

        path_base_colors = [PURPLE_ANT_PATH_ITER_BASE, LIGHT_BLUE_ANT_PATH_ITER_BASE, 
                            YELLOW_ANT_PATH_ITER_BASE, CYAN_ANT_PATH_ITER_BASE]
        ant_path_alpha = 100 

        if ant_paths_this_iter:
            for idx, ant_path in enumerate(ant_paths_this_iter):
                if ant_path and len(ant_path) > 1:
                    current_color_rgb = path_base_colors[idx % len(path_base_colors)]
                    self.draw_positions(ant_path, current_color_rgb + (ant_path_alpha,), is_path=True, line_thickness_factor=0.05)
        
        self.draw_text(f"ACO Ant Iter: {iteration_num + 1}/{total_iterations_planner}", 
                       (10, self.screen_height - 30 if self.screen_height > 60 else 10), color=BLACK)
        self.update_display()

    def draw_final_path(self, static_map_grid, pheromone_map_to_display, final_path, start_pos, goal_pos, pher_display_max=None):
        self.clear_screen()
        self.draw_static_map(static_map_grid)
        if pheromone_map_to_display is not None:
             self.draw_pheromone_map_overlay(pheromone_map_to_display, static_map_grid, pher_max_val_display=pher_display_max)
        
        if start_pos: self.draw_positions([start_pos], GREEN_START, radius_factor=0.45)
        if goal_pos: self.draw_positions([goal_pos], ORANGE_GOAL, radius_factor=0.45)
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
    
    def draw_frontier_clusters(self, frontier_clusters, representative_points=None):
        """
        Draws frontier clusters on the screen.
        frontier_clusters: A list of lists, where each inner list is a cluster of (r,c) points.
        representative_points: Optional list of representative (r,c) points for each cluster, to be highlighted.
        """
        if not frontier_clusters:
            return

        for i, cluster in enumerate(frontier_clusters):
            cluster_color_base = CLUSTER_COLORS[i % len(CLUSTER_COLORS)][:3] # Get RGB
            cluster_alpha = CLUSTER_COLORS[i % len(CLUSTER_COLORS)][3]      # Get Alpha

            # Draw each frontier point in the cluster
            for r, c in cluster:
                center_x = int(c * self.cell_size + self.cell_size * 0.5)
                center_y = int(r * self.cell_size + self.cell_size * 0.5)
                
                # Draw a small, semi-transparent circle for each point in cluster
                # Or a small rectangle fill
                s = pygame.Surface((self.cell_size,self.cell_size), pygame.SRCALPHA)
                # Slightly smaller rect inside the cell
                pygame.draw.rect(s, cluster_color_base + (cluster_alpha,), 
                                 (self.cell_size*0.2, self.cell_size*0.2, 
                                  self.cell_size*0.6, self.cell_size*0.6), 0) 
                self.screen.blit(s, (c*self.cell_size, r*self.cell_size))
                
                # Optionally, draw cluster index number
                # id_surf = self.font_small.render(str(i), True, BLACK)
                # self.screen.blit(id_surf, (center_x - id_surf.get_width()//2, center_y - id_surf.get_height()//2))


            # Highlight representative point if provided
            if representative_points and i < len(representative_points) and representative_points[i] is not None:
                rep_r, rep_c = representative_points[i]
                rep_center_x = int(rep_c * self.cell_size + self.cell_size * 0.5)
                rep_center_y = int(rep_r * self.cell_size + self.cell_size * 0.5)
                # Draw a more prominent marker for the representative point
                pygame.draw.circle(self.screen, cluster_color_base, (rep_center_x, rep_center_y), int(self.cell_size * 0.4), 2) # Outline
                pygame.draw.circle(self.screen, WHITE, (rep_center_x, rep_center_y), int(self.cell_size * 0.2)) # Inner dot

    # --- Event Handling and Screen Update ---
    def handle_events(self): # For main_simulation.py (exploration)
        """Handles Pygame events, returns False to quit, or 'SKIP_ANTS'."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_SPACE: # For skipping ant viz in original ACO
                    return "SKIP_ANTS" 
                elif event.key == pygame.K_ESCAPE: # Universal quit
                    return False
        return True

    def handle_events_simple(self): # For simpler demos like known map ACO
        """Handles basic Pygame events: QUIT and ESCAPE key. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: 
                    return False
        return True
    
    def update_display(self): 
        pygame.display.flip()

    def clear_screen(self): 
        self.screen.fill(WHITE)

    def save_screenshot(self, directory, filename):
        if not self.screen:
            print("Error: Screen not initialized, cannot save screenshot.")
            return
        if not os.path.exists(directory): # ensure_dir was external, now internal check
            os.makedirs(directory)
        full_path = os.path.join(directory, filename)
        try:
            pygame.image.save(self.screen, full_path)
        except pygame.error as e:
            print(f"Error saving screenshot to {full_path}: {e}")

    def quit(self): 
        pygame.quit()