from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE # OBSTACLE is needed for pathfinding
from .geometry_utils import check_line_of_sight 
from collections import deque # For BFS used in clustering

class IGEPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range, 
                 frontier_clustering_distance=3): # Max distance for frontiers to be in same cluster
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.frontier_clustering_distance = frontier_clustering_distance # Max Manhattan distance for clustering

    def _calculate_information_gain_with_los(self, 
                                             prospective_robot_pos,
                                             current_known_map,
                                             sensor_range_for_ig):
        ig = 0
        r_prospective, c_prospective = prospective_robot_pos
        # Keep track of unknown cells already counted for this IG calculation to avoid double counting
        # if multiple LoS rays from prospective_robot_pos see the same UNKNOWN cell.
        # However, for IG from a single pose, each UNKNOWN cell in LoS is counted once.
        # The main double counting concern is when aggregating IG from *multiple* frontier points in a cluster.
        
        # For IG calculation from a *single* prospective_robot_pos, we don't need to worry about
        # double counting the same UNKNOWN cell *within this single calculation*.
        # The issue arises when summing IGs from different points in a cluster.

        for dr in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
            for dc in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
                # Optional: circular sensor shape
                # if dr**2 + dc**2 > sensor_range_for_ig**2: continue

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
        visited_frontiers = set() # To keep track of frontiers already added to a cluster

        for frontier_seed in frontiers:
            if frontier_seed in visited_frontiers:
                continue

            current_cluster = []
            queue = deque([frontier_seed])
            visited_frontiers.add(frontier_seed)
            
            while queue:
                current_fr = queue.popleft()
                current_cluster.append(current_fr)

                # Check other frontiers for proximity to current_fr
                for other_fr in frontiers:
                    if other_fr not in visited_frontiers:
                        # Using Manhattan distance for clustering
                        dist = abs(current_fr[0] - other_fr[0]) + abs(current_fr[1] - other_fr[1])
                        if dist <= self.frontier_clustering_distance:
                            # And ensure there's a somewhat clear path (e.g. not separated by thick wall)
                            # This simple clustering doesn't do complex path check, just proximity.
                            # More advanced: check A* path length between them on known_map (costly).
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
        # To calculate true aggregated IG, we need to find all unique UNKNOWN cells
        # that would be revealed if the robot could sense from *all* points in the cluster,
        # or from a representative viewpoint for the cluster.
        # This is complex. A common simplification:
        # 1. Pick a representative point for the cluster (e.g., centroid, or point closest to robot).
        # 2. Calculate IG from that representative point.
        # 3. Optionally, multiply by a factor related to cluster size (e.g., sqrt(len(cluster_frontiers))).

        # Simpler Aggregated IG: Sum of individual IGs in the cluster,
        # then penalize/normalize by cluster size to avoid overly favoring huge, sparse clusters.
        # Or, for a more accurate (but costlier) unique count:
        
        uniquely_seen_unknown_cells = set()
        # For each frontier point in the cluster, simulate sensing and collect unique unknown cells
        # This still has some overlap if sensing areas of different frontiers overlap heavily.
        # The most accurate way would be to define a "sensing pose" for the cluster.

        # Let's try: IG of the cluster is the IG from its centroid (if centroid is FREE),
        # scaled by the number of points in the cluster (e.g. sqrt(N) or log(N)).
        if not cluster_frontiers: return 0
        
        # Calculate centroid of the cluster
        # Centroid might not be on a FREE cell or even on the map.
        # A better representative: the frontier in the cluster closest to the robot,
        # OR the frontier in the cluster with the highest individual IG.

        # For now, let's use a simpler approach:
        # Calculate IG from the frontier point in the cluster that is closest to the robot (if path exists).
        # This focuses on the "entry point" IG.
        # The "聚合增益点越多聚合增益越大" can be modeled by scaling this IG.
        
        # Fallback: sum individual IGs and apply a sublinear scaling for size.
        # This doesn't correctly handle overlapping views.
        
        # Correct (but potentially slow) way:
        # Find all UNKNOWN cells. For each UNKNOWN cell, check if it's visible (LoS)
        # from AT LEAST ONE of the frontier_pos in the cluster when robot is at that frontier_pos.
        
        # --- Simplified approach for this example: ---
        # 1. Choose a representative point for the cluster (e.g., the first one for simplicity,
        #    or one that's easily reachable / central).
        # 2. Calculate IG from that representative point.
        # 3. Scale this IG by a factor музею to the cluster size.
        #    Factor = k * len(cluster_frontiers) (linear scaling, as requested by "越多聚合增益越大")
        #    or k * sqrt(len(cluster_frontiers)) (sub-linear to avoid over-favoring huge clusters)
        
        if not cluster_frontiers: return 0.0, None

        # For this version, let's pick the first frontier in the cluster as representative.
        # A better choice would be the one closest to the robot, or with highest individual IG.
        representative_frontier = cluster_frontiers[0] # Simplification
        
        # Calculate IG from this representative point
        ig_from_representative = self._calculate_information_gain_with_los(
            representative_frontier, 
            known_map_snapshot, 
            sensor_range
        )
        
        # Scale by cluster size (number of points in it)
        # "一定要满足聚合点的点越多聚合增益越大"
        # Linear scaling:
        aggregated_ig = ig_from_representative * len(cluster_frontiers) 
        # Sub-linear scaling (often preferred to avoid over-emphasis on size):
        # aggregated_ig = ig_from_representative * np.sqrt(len(cluster_frontiers))
        
        return aggregated_ig, representative_frontier # Return IG and which point was used

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers:
            return None, None

        # Step 1: Cluster the frontiers
        frontier_clusters = self._cluster_frontiers(all_frontiers, known_map)
        if not frontier_clusters:
            # print("IGE (Clustered): No frontier clusters found. Falling back.")
            return self._final_fallback_plan(robot_pos, known_map)

        best_utility = -float('inf')
        best_cluster_representative_target = None 
        best_path_to_target = None
        
        known_map_snapshot = np.copy(known_map)

        # print(f"IGE (Clustered): Found {len(frontier_clusters)} clusters.")

        # Step 2: Evaluate each cluster
        for cluster in frontier_clusters:
            if not cluster: continue

            # Calculate aggregated IG and get a representative point for this cluster
            # The representative point is what the robot will navigate towards.
            # The _calculate_cluster_aggregated_ig now returns the IG and the point used for calculation
            aggregated_ig, representative_point_for_pathing = self._calculate_cluster_aggregated_ig(
                cluster, known_map_snapshot, self.robot_sensor_range
            )

            if representative_point_for_pathing is None or aggregated_ig <=0 : # No valid representative or no gain
                continue

            # Path to the representative point of the cluster
            path = self._is_reachable_and_get_path(robot_pos, representative_point_for_pathing, known_map_snapshot)
            
            if path:
                path_cost = len(path) - 1
                if path_cost <= 0: path_cost = 1e-5

                # Utility = Aggregated_IG / Cost_to_Representative_Point
                utility = aggregated_ig / path_cost
                
                # print(f"  Cluster (size {len(cluster)}), Rep: {representative_point_for_pathing}, AggIG: {aggregated_ig:.2f}, Cost: {path_cost:.2f}, Util: {utility:.2f}")

                if utility > best_utility:
                    best_utility = utility
                    best_cluster_representative_target = representative_point_for_pathing
                    best_path_to_target = path
        
        if best_cluster_representative_target is not None:
            # print(f"IGE (Clustered) Selected Rep: {best_cluster_representative_target} with utility {best_utility:.2f}")
            return best_cluster_representative_target, best_path_to_target
        else:
            # print(f"IGE (Clustered): No suitable cluster found. Falling back.")
            return self._final_fallback_plan(robot_pos, known_map)