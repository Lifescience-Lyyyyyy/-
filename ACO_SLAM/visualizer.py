# visualizer.py
import pygame
from environment import UNKNOWN, FREE, OBSTACLE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY_UNKNOWN = (128, 128, 128) # Unknown
LIGHT_GREEN = (144, 238, 144) # Free
DARK_GRAY_OBSTACLE = (60, 60, 60)  # Obstacle
RED_ROBOT = (255, 0, 0)
BLUE_PATH = (0, 0, 255)
YELLOW_FRONTIER = (255, 255, 0)
ORANGE_TARGET = (255, 165, 0)

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = env_width * cell_size
        self.screen_height = env_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Active SLAM Simulation")
        self.font = pygame.font.SysFont(None, 24)

    def draw_environment(self, known_map, true_map=None, show_true_map=False):
        map_to_draw = true_map if show_true_map else known_map
        if map_to_draw is None: return

        for r in range(map_to_draw.shape[0]):
            for c in range(map_to_draw.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = GRAY_UNKNOWN
                if map_to_draw[r, c] == FREE:
                    color = LIGHT_GREEN
                elif map_to_draw[r, c] == OBSTACLE:
                    color = DARK_GRAY_OBSTACLE
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1) # Grid lines

    def draw_robot(self, robot_pos):
        r, c = robot_pos
        center_x = c * self.cell_size + self.cell_size // 2
        center_y = r * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, RED_ROBOT, (center_x, center_y), self.cell_size // 3)

    def draw_path(self, path):
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            start_r, start_c = path[i]
            end_r, end_c = path[i+1]
            start_px = (start_c * self.cell_size + self.cell_size // 2, 
                        start_r * self.cell_size + self.cell_size // 2)
            end_px = (end_c * self.cell_size + self.cell_size // 2, 
                      end_r * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, BLUE_PATH, start_px, end_px, 2)
            
    def draw_frontiers(self, frontiers):
        if not frontiers: return
        for r, c in frontiers:
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, YELLOW_FRONTIER, rect, 2) # Draw border

    def draw_target(self, target_pos):
        if not target_pos: return
        r, c = target_pos
        center_x = c * self.cell_size + self.cell_size // 2
        center_y = r * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, ORANGE_TARGET, (center_x, center_y), self.cell_size // 2, 3)


    def draw_text(self, text, position, color=BLACK):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def update_display(self):
        pygame.display.flip()

    def clear_screen(self):
        self.screen.fill(WHITE)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def quit(self):
        pygame.quit()