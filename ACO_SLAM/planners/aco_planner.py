# planners/aco_planner.py
from .base_planner import BasePlanner
import numpy as np
import random
from environment import UNKNOWN, FREE

class ACOPlanner(BasePlanner):
    def __init__(self, environment, n_ants=10, n_iterations=20, alpha=1.0, beta=2.0, evaporation_rate=0.5, q0=0.1):
        super().__init__(environment)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic influence
        self.evaporation_rate = evaporation_rate
        self.q0 = q0 # Exploration-exploitation trade-off (not used in this simplified version directly in selection)
        
        # Pheromone: Store pheromone on (from_node, to_node) pairs.
        # Here, from_node is effectively the current robot position, and to_node is a frontier.
        # For simplicity, we re-initialize pheromones for frontiers at each planning step,
        # or we can have a global pheromone map on grid cells that could be frontiers.
        # Let's try simpler: pheromone on frontiers themselves, updated based on selection.
        self.pheromones = {} # {(fr, fc): value}

    def _get_heuristic_value(self, robot_pos, frontier_pos, known_map):
        """
        启发式信息：
        1. 距离的倒数 (越近越好)
        2. 预期信息增益 (从该边界点能看到的未知单元数量)
        """
        dist = np.sqrt((frontier_pos[0] - robot_pos[0])**2 + (frontier_pos[1] - robot_pos[1])**2)
        if dist == 0: dist = 1e-5 # Avoid division by zero

        # Estimate information gain: count unknown cells around frontier within sensor range
        # This is a simplification. A better IG would simulate sensing from the frontier.
        ig = 0
        r_f, c_f = frontier_pos
        # Check a small area around the frontier for UNKNOWN cells
        # This is a proxy for how many new cells might be revealed
        # A more accurate IG would use the robot's sensor_range from the frontier_pos
        # Let's use a fixed small radius for IG estimation around the frontier itself
        ig_sensor_radius = 2 # Smaller than robot's actual sensor, just for quick heuristic
        for dr in range(-ig_sensor_radius, ig_sensor_radius + 1):
            for dc in range(-ig_sensor_radius, ig_sensor_radius + 1):
                nr, nc = r_f + dr, c_f + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        
        # Heuristic: higher IG is better, lower distance is better
        # Combine them: e.g., (IG + 1) / distance. Adding 1 to IG to avoid zero IG issues.
        heuristic = (ig + 1.0) / dist 
        return heuristic

    def plan_next_action(self, robot_pos, known_map):
        frontiers = self.find_frontiers(known_map)
        if not frontiers:
            return None

        # Initialize/update pheromones for current frontiers
        # For frontiers not seen before, give a small initial pheromone
        # For existing ones, they already have values (or apply evaporation globally if storing on grid)
        initial_pheromone = 1.0 
        for f_pos in frontiers:
            if f_pos not in self.pheromones:
                self.pheromones[f_pos] = initial_pheromone
        
        # Temporary dict for current iteration's choices for pheromone update
        # Key: frontier_pos, Value: list of ants that chose it (or sum of their quality)
        delta_pheromones = {f_pos: 0.0 for f_pos in frontiers}


        best_frontier_overall = None
        best_heuristic_overall = -float('inf') # Track best frontier based on heuristic initially

        for iteration in range(self.n_iterations):
            ants_choices = [] # Stores (chosen_frontier, heuristic_value_of_choice) for this iteration's ants

            for ant_k in range(self.n_ants):
                probabilities = []
                prob_sum = 0.0

                # Calculate probabilities for each frontier
                for f_idx, f_pos in enumerate(frontiers):
                    tau = self.pheromones.get(f_pos, initial_pheromone)**self.alpha
                    eta = self._get_heuristic_value(robot_pos, f_pos, known_map)**self.beta
                    
                    # Track the best frontier seen by any ant based on raw heuristic (for initial choice if all probs are zero)
                    if eta > best_heuristic_overall:
                        best_heuristic_overall = eta
                        best_frontier_overall = f_pos
                        
                    probabilities.append({'frontier': f_pos, 'prob_val': tau * eta})
                    prob_sum += probabilities[-1]['prob_val']

                if prob_sum == 0: # All options have zero probability (e.g. zero pheromone and zero heuristic)
                    # Fallback: choose randomly or based on best heuristic found so far
                    if best_frontier_overall:
                        chosen_frontier = best_frontier_overall
                    else: # Should not happen if frontiers exist
                        chosen_frontier = random.choice(frontiers) if frontiers else None
                else:
                    # Normalize probabilities
                    for p_info in probabilities:
                        p_info['prob_val'] /= prob_sum
                    
                    # Roulette wheel selection
                    r = random.random()
                    cumulative_prob = 0.0
                    chosen_frontier = None
                    for p_info in probabilities:
                        cumulative_prob += p_info['prob_val']
                        if r <= cumulative_prob:
                            chosen_frontier = p_info['frontier']
                            break
                    if chosen_frontier is None and probabilities: # Should not happen if prob_sum > 0
                         chosen_frontier = probabilities[-1]['frontier']


                if chosen_frontier:
                    # Quality of choice (heuristic value) - for pheromone update
                    # This is a simplification. Real ACO might build a path and evaluate path quality.
                    # Here, "path" is just one step to a frontier.
                    quality = self._get_heuristic_value(robot_pos, chosen_frontier, known_map)
                    delta_pheromones[chosen_frontier] += quality 
                    ants_choices.append(chosen_frontier)


            # Pheromone evaporation
            for f_pos in list(self.pheromones.keys()): # Iterate over copy of keys for safe modification
                if f_pos in frontiers: # Only evaporate for currently considered frontiers
                    self.pheromones[f_pos] *= (1.0 - self.evaporation_rate)
                else: # Remove pheromone for frontiers that no longer exist
                    del self.pheromones[f_pos]

            # Pheromone deposit
            for f_pos, total_quality_deposit in delta_pheromones.items():
                if f_pos in self.pheromones: # Ensure frontier still exists
                     self.pheromones[f_pos] += total_quality_deposit # Add aggregated quality
                # Reset for next iteration (or handle accumulation differently if desired)
                delta_pheromones[f_pos] = 0.0


        # After all iterations, select the best frontier based on final pheromone levels + heuristic
        best_score = -float('inf')
        final_choice = None
        if not frontiers: return None

        for f_pos in frontiers:
            score = self.pheromones.get(f_pos, initial_pheromone) * \
                    self._get_heuristic_value(robot_pos, f_pos, known_map) # Could also use beta here
            if score > best_score:
                if self._is_reachable(robot_pos, f_pos, known_map): # Simplified reachability
                    best_score = score
                    final_choice = f_pos
        
        if final_choice is None and best_frontier_overall: # Fallback if all scores are bad
            final_choice = best_frontier_overall
        elif final_choice is None and frontiers: # Ultimate fallback
            final_choice = random.choice(frontiers)

        return final_choice

if __name__ == '__main__':
    from environment import Environment
    env = Environment(20, 15, obstacle_percentage=0.2)
    robot_start_pos = (env.height // 2, env.width // 2)
    
    # Simulate some initial known area
    for r_off in range(-2, 3):
        for c_off in range(-2, 3):
            r, c = robot_start_pos[0] + r_off, robot_start_pos[1] + c_off
            if env.is_within_bounds(r,c):
                env.grid_known[r,c] = env.grid_true[r,c]

    aco_planner = ACOPlanner(env, n_ants=5, n_iterations=10)
    
    print("Known map for ACO test:")
    print(env.grid_known)
    
    next_target = aco_planner.plan_next_action(robot_start_pos, env.grid_known)
    print("ACO Next Target:", next_target)
    
    if next_target:
        # Simulate robot moving and sensing to see pheromone update effect (conceptual)
        # In main loop, this would happen naturally
        print(f"Pheromones before potential update for {next_target}: {aco_planner.pheromones.get(next_target)}")
        # ... robot moves, senses ...
        # ... next call to plan_next_action would use updated pheromones ...