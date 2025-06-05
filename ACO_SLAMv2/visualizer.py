# visualizer.py
import pygame
from environment import UNKNOWN, FREE, OBSTACLE # Make sure these are imported if used for colors directly
import os # Needed for path joining

# --- Colors (defined at module level) ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY_UNKNOWN = (128, 128, 128)
LIGHT_GREEN = (144, 238, 144)
DARK_GRAY_OBSTACLE = (60, 60, 60)
RED_ROBOT = (255, 0, 0)
BLUE_PATH = (0, 0, 255)
YELLOW_FRONTIER = (255, 255, 0)
ORANGE_TARGET = (255, 165, 0)
PURPLE_ANT_PATH = (128, 0, 128)
LIGHT_BLUE_ANT_PATH = (173, 216, 230)
# --- End Colors ---

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = env_width * cell_size
        self.screen_height = env_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Active SLAM Simulation")
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)

    # ... (draw_environment, draw_robot, etc. methods remain the same) ...
    def draw_environment(self, known_map, true_map=None, show_true_map=False):
        map_to_draw = true_map if show_true_map else known_map
        if map_to_draw is None: return

        for r in range(map_to_draw.shape[0]):
            for c in range(map_to_draw.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = GRAY_UNKNOWN 
                cell_state = map_to_draw[r, c]
                if cell_state == FREE:
                    color = LIGHT_GREEN
                elif cell_state == OBSTACLE:
                    color = DARK_GRAY_OBSTACLE
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1) 

    def draw_robot(self, robot_pos):
        r, c = robot_pos
        center_x = c * self.cell_size + self.cell_size // 2
        center_y = r * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, RED_ROBOT, (center_x, center_y), self.cell_size // 3)

    def draw_robot_path(self, path_taken_visual): 
        if len(path_taken_visual) < 2:
            return
        
        for i in range(len(path_taken_visual) - 1):
            start_r, start_c = path_taken_visual[i]
            end_r, end_c = path_taken_visual[i+1]

            if abs(start_r - end_r) <=1 and abs(start_c - end_c) <=1 : 
                 start_px = (start_c * self.cell_size + self.cell_size // 2, 
                            start_r * self.cell_size + self.cell_size // 2)
                 end_px = (end_c * self.cell_size + self.cell_size // 2, 
                          end_r * self.cell_size + self.cell_size // 2)
                 pygame.draw.line(self.screen, BLUE_PATH, start_px, end_px, 2)
            
    def draw_frontiers(self, frontiers):
        if not frontiers: return
        for r, c in frontiers:
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, YELLOW_FRONTIER, rect, 2) 

    def draw_target(self, target_pos):
        if not target_pos: return
        r, c = target_pos
        center_x = c * self.cell_size + self.cell_size // 2
        center_y = r * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, ORANGE_TARGET, (center_x, center_y), self.cell_size // 2, 3)

    def draw_ant_paths_iteration(self, robot_pos, known_map, ant_paths_this_iter, environment, iter_num, total_iters):
        self.clear_screen()
        self.draw_environment(known_map) 
        self.draw_robot(robot_pos)
        
        path_colors = [PURPLE_ANT_PATH, LIGHT_BLUE_ANT_PATH, (150,150,0), (0,150,150)]
        color_idx = 0
        for ant_path in ant_paths_this_iter:
            if ant_path and len(ant_path) > 1:
                current_color = path_colors[color_idx % len(path_colors)]
                color_idx +=1
                for i in range(len(ant_path) - 1):
                    start_r, start_c = ant_path[i]
                    end_r, end_c = ant_path[i+1]
                    start_px = (start_c * self.cell_size + self.cell_size // 2,
                                start_r * self.cell_size + self.cell_size // 2)
                    end_px = (end_c * self.cell_size + self.cell_size // 2,
                              end_r * self.cell_size + self.cell_size // 2)
                    pygame.draw.line(self.screen, current_color, start_px, end_px, 1) 
        
        self.draw_text(f"ACO Ant Iteration: {iter_num + 1}/{total_iters}", (10, self.screen_height - 30), color=BLACK)
        self.update_display()
        pygame.time.wait(150) 

    def draw_text(self, text, position, color=BLACK, small=False):
        font_to_use = self.font_small if small else self.font
        text_surface = font_to_use.render(text, True, color)
        self.screen.blit(text_surface, position)
    
    def update_display(self):
        pygame.display.flip()

    def clear_screen(self):
        self.screen.fill(WHITE)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_SPACE:
                    return "SKIP_ANTS" 
        return True

    def quit(self):
        pygame.quit()

    def save_screenshot(self, directory, filename):
        """Saves the current screen content to a file."""
        if not self.screen:
            print("Error: Screen not initialized, cannot save screenshot.")
            return
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        full_path = os.path.join(directory, filename)
        try:
            pygame.image.save(self.screen, full_path)
            # print(f"Screenshot saved to {full_path}")
        except pygame.error as e:
            print(f"Error saving screenshot to {full_path}: {e}")