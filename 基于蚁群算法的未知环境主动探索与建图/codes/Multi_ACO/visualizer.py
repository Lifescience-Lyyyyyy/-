# visualizer_optimized.py
import pygame
import numpy as np
from collections import deque

try:
    from environment import UNKNOWN, FREE, OBSTACLE
except ImportError:
    UNKNOWN, FREE, OBSTACLE = 0, 1, 2

BLACK = (0, 0, 0); WHITE = (255, 255, 255); GRAY_UNKNOWN = (128, 128, 128)
LIGHT_GREEN_FREE = (144, 238, 144); DARK_GRAY_OBSTACLE = (60, 60, 60)
PHEROMONE_COLOR_BASE = (30, 30, 180)

class Visualizer:
    def __init__(self, env_width, env_height, cell_size=20):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = env_width * self.cell_size
        self.screen_height = env_height * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-Agent Collaborative Exploration (Optimized)")
        self.font = pygame.font.Font(None, max(12, int(cell_size * 0.8)))

        # (优化) 缓存表面
        self.background_cache = pygame.Surface(self.screen.get_size()).convert()
        self.pheromone_cache = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA).convert_alpha()
        
        # (优化) 脏标志，用于决定是否需要重绘缓存
        self.is_background_dirty = True
        self.is_pheromones_dirty = True

    # (优化) 新方法：更新背景缓存
    def _update_background_cache(self, static_map_grid):
        self.background_cache.fill(WHITE)
        for r in range(static_map_grid.shape[0]):
            for c in range(static_map_grid.shape[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                cell_state = static_map_grid[r, c]
                color = WHITE
                if cell_state == FREE: color = LIGHT_GREEN_FREE
                elif cell_state == OBSTACLE: color = DARK_GRAY_OBSTACLE
                elif cell_state == UNKNOWN: color = GRAY_UNKNOWN
                pygame.draw.rect(self.background_cache, color, rect)
                pygame.draw.rect(self.background_cache, BLACK, rect, 1)
        self.is_background_dirty = False

    # (优化) 新方法：更新信息素缓存
    def _update_pheromone_cache(self, pheromone_map, static_map_ref):
        if pheromone_map is None: return
        pher_max_val = np.max(pheromone_map[static_map_ref != OBSTACLE]) if np.any(static_map_ref != OBSTACLE) else 1.0
        if pher_max_val <= 1e-6: pher_max_val = 1.0
        
        self.pheromone_cache.fill((0,0,0,0)) # 清空透明表面
        min_alpha, max_alpha = 10, 160
        for r in range(pheromone_map.shape[0]):
            for c in range(pheromone_map.shape[1]):
                if static_map_ref[r, c] == OBSTACLE: continue
                norm_pher = np.clip(pheromone_map[r, c] / pher_max_val, 0, 1)
                alpha = int(min_alpha + norm_pher * (max_alpha - min_alpha))
                if alpha > min_alpha + 5:
                    s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    intensity = int(norm_pher * 155) + 100
                    color = (PHEROMONE_COLOR_BASE[0], PHEROMONE_COLOR_BASE[1], np.clip(intensity,0,255))
                    s.fill(color + (alpha,))
                    self.pheromone_cache.blit(s, (c * self.cell_size, r * self.cell_size))
        self.is_pheromones_dirty = False
    
    # (优化) 核心绘制方法现在非常快
    def draw_simulation_state(self, known_map, pheromone_map, robots, paths, targets, texts):
        # 检查是否需要更新缓存
        if self.is_background_dirty:
            self._update_background_cache(known_map)
        if self.is_pheromones_dirty:
            self._update_pheromone_cache(pheromone_map, known_map)
            
        # 1. 快速绘制缓存的背景和信息素
        self.screen.blit(self.background_cache, (0, 0))
        self.screen.blit(self.pheromone_cache, (0, 0))

        # 2. 绘制动态元素（机器人、路径等），这些数量少，绘制快
        self.draw_all_paths(robots, paths)
        self.draw_robots(robots)
        self.draw_reserved_targets(targets, robots)
        
        # 3. 绘制文本
        for text_info in texts:
            self.draw_text(text_info['text'], text_info['pos'])

        self.update_display()
        
    def draw_robots(self, robots_list):
        for robot in robots_list:
            r, c = robot.get_position()
            center = (int(c * self.cell_size + self.cell_size / 2), int(r * self.cell_size + self.cell_size / 2))
            pygame.draw.circle(self.screen, robot.color, center, int(self.cell_size * 0.4))
            pygame.draw.circle(self.screen, BLACK, center, int(self.cell_size * 0.4), 1)

    def draw_all_paths(self, robots, paths_deques):
        for i, robot in enumerate(robots):
            path_to_draw = [robot.get_position()] + list(paths_deques[i])
            if len(path_to_draw) > 1:
                pixel_path = [(p[1] * self.cell_size + self.cell_size // 2, 
                               p[0] * self.cell_size + self.cell_size // 2) for p in path_to_draw]
                pygame.draw.lines(self.screen, robot.color, False, pixel_path, 2)

    def draw_reserved_targets(self, reserved_targets, robots):
        for robot_id, target_pos in reserved_targets.items():
            r, c = target_pos
            color = robots[robot_id].color
            center_x = int(c * self.cell_size + self.cell_size / 2)
            center_y = int(r * self.cell_size + self.cell_size / 2)
            pygame.draw.line(self.screen, color, (center_x - 4, center_y - 4), (center_x + 4, center_y + 4), 3)
            pygame.draw.line(self.screen, color, (center_x + 4, center_y - 4), (center_x - 4, center_y + 4), 3)

    def draw_text(self, text, position, color=BLACK):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)
        
    def update_display(self): pygame.display.flip()
    def handle_events_simple(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
        return True
    def quit(self): pygame.quit()