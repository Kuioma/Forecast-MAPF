from typing import Literal
from pogema_toolbox.algorithm_config import AlgoBase
import numpy as np
from .CBS import CBS

class CBSInferenceConfig(AlgoBase):
    name: Literal["CBS"] = "CBS"

class CBSInference:
    def __init__(self, cfg: CBSInferenceConfig):
        self.cfg = cfg
        self.planned_paths = {} 
        self.step_idx = 0

    def act(self, observations):
        # Plan if not already done for this episode
        if not self.planned_paths:
            self.plan(observations)
        
        actions = []
        # Mapping from coordinate difference to action index
        # [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]
        #   0        1       2        3       4
        # (dx, dy) -> action
        move_map = {
            (0, 0): 0,
            (-1, 0): 1,
            (1, 0): 2,
            (0, -1): 3,
            (0, 1): 4
        }
        
        for i, obs in enumerate(observations):
            action = 0 # Default wait
            if i in self.planned_paths:
                path = self.planned_paths[i]
                current_step = self.step_idx
                next_step = self.step_idx + 1
                
                if next_step < len(path):
                    # path contains coordinates [x, y]
                    # path[current_step] should ideally match obs['global_xy'] if no execution error
                    # But we trust the plan.
                    
                    curr_pos = path[current_step]
                    next_pos = path[next_step]
                    
                    # Ensure they are tuples/iterables we can subtract
                    dx = int(next_pos[0] - curr_pos[0])
                    dy = int(next_pos[1] - curr_pos[1])
                    
                    action = move_map.get((dx, dy), 0)
            
            actions.append(action)
            
        self.step_idx += 1
        return actions

    def plan(self, observations):
        if not observations:
            return

        # Extract grid
        # CBS expects 0=obstacle, 1=free.
        # Pogema gives 1=obstacle, 0=free.
        pogema_grid = observations[0]['global_obstacles']
        cbs_grid = (pogema_grid == 0).astype(int)
        
        # Extract tasks
        positions = {}
        for i, obs in enumerate(observations):
            start = tuple(obs['global_xy'])
            goal = tuple(obs['global_target_xy'])
            positions[i] = [start, goal]
            
        planner = CBS(cbs_grid, need_convert=False)
        # Using a reasonable iteration limit for benchmark
        result = planner.plan(positions, max_iterations=5000)
        
        if result and result.sol:
            self.planned_paths = result.sol
        else:
            # If failed, empty dict results in valid 'wait' actions
            self.planned_paths = {}

    def reset_states(self):
        self.planned_paths = {}
        self.step_idx = 0
