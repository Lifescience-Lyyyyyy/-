from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE 
from .geometry_utils import check_line_of_sight 
from collections import deque 

class IGEPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range, 
                 frontier_clustering_distance=3): 
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.frontier_clustering_distance = frontier_clustering_distance 

    def _calculate_information_gain_with_los(self, 
                                             prospective_robot_pos,
                                             current_known_map,
                                             sensor_range_for_ig):
        ig = 0
        r_prospective, c_prospective = prospective_robot_pos


        for dr in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
            for dc in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
                r_target_cell, c_target_cell = r_prospective + dr, c_prospective + dc
                if self.environment.is_within_bounds(r_target_cell, c_target_cell):
                    if current_known_map[r_target_cell, c_target_cell] == UNKNOWN:
                        if check_line_of_sight(r_prospective, c_prospective, 
                                               r_target_cell, c_target_cell, 
                                               current_known_map):
                            ig += 1
        return ig

    def _cluster_frontiers(self, frontiers, known_map):
        """
        Clusters adjacent/nearby frontiers together.
        Uses a simple BFS-like approach based on Manhattan distance.
        Returns a list of clusters, where each cluster is a list of frontier points.
        """
        if not frontiers:
            return []

        clusters = []
        visited_frontiers = set() 

        for frontier_seed in frontiers:
            if frontier_seed in visited_frontiers:
                continue

            current_cluster = []
            queue = deque([frontier_seed])
            visited_frontiers.add(frontier_seed)
            
            while queue:
                current_fr = queue.popleft()
                current_cluster.append(current_fr)

                for other_fr in frontiers:
                    if other_fr not in visited_frontiers:
                        dist = abs(current_fr[0] - other_fr[0]) + abs(current_fr[1] - other_fr[1])
                        if dist <= self.frontier_clustering_distance:
                            visited_frontiers.add(other_fr)
                            queue.append(other_fr)
            
            if current_cluster:
                clusters.append(current_cluster)
        
        return clusters

    def _calculate_cluster_aggregated_ig(self, cluster_frontiers, known_map_snapshot, sensor_range):
        """
        Calculates the aggregated information gain for a cluster of frontiers.
        This attempts to count unique unknown cells visible from ANY point in the cluster.
        A simpler approach is to sum IGs of individual frontiers and apply a penalty for size,
        or pick a representative point.
        This version aims for unique unknown cell count.
        """

        uniquely_seen_unknown_cells = set()

        if not cluster_frontiers: return 0
        
        if not cluster_frontiers: return 0.0, None

        representative_frontier = cluster_frontiers[0] 
        
        ig_from_representative = self._calculate_information_gain_with_los(
            representative_frontier, 
            known_map_snapshot, 
            sensor_range
        )
        
        aggregated_ig = ig_from_representative * len(cluster_frontiers) 
        
        return aggregated_ig, representative_frontier 

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers:
            return None, None

        frontier_clusters = self._cluster_frontiers(all_frontiers, known_map)
        if not frontier_clusters:
            return self._final_fallback_plan(robot_pos, known_map)

        best_utility = -float('inf')
        best_cluster_representative_target = None 
        best_path_to_target = None
        
        known_map_snapshot = np.copy(known_map)

        for cluster in frontier_clusters:
            if not cluster: continue

            aggregated_ig, representative_point_for_pathing = self._calculate_cluster_aggregated_ig(
                cluster, known_map_snapshot, self.robot_sensor_range
            )

            if representative_point_for_pathing is None or aggregated_ig <=0 : 
                continue

            path = self._is_reachable_and_get_path(robot_pos, representative_point_for_pathing, known_map_snapshot)
            
            if path:
                path_cost = len(path) - 1
                if path_cost <= 0: path_cost = 1e-5

                utility = aggregated_ig / path_cost
             
                if utility > best_utility:
                    best_utility = utility
                    best_cluster_representative_target = representative_point_for_pathing
                    best_path_to_target = path
        
        if best_cluster_representative_target is not None:
          
            return best_cluster_representative_target, best_path_to_target
        else:
            return self._final_fallback_plan(robot_pos, known_map)