import sys
import os
import numpy as np
import time

# Add the current directory to sys.path to import cbs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add pogema to sys.path if not installed
# Assuming pogema repo is at ./pogema
pogema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pogema')
if pogema_path not in sys.path:
    sys.path.append(pogema_path)

try:
    from pogema import pogema_v0, GridConfig
except ImportError:
    print("Could not import pogema. Please make sure it is installed or in PYTHONPATH.")
    sys.exit(1)

from cbs.CBS import CBS

def run_cbs_on_pogema():
    # 1. Define Pogema GridConfig
    # Using a small map for testing
    grid_config = GridConfig(
        num_agents=16,
        size=8,
        density=0.2,
        seed=42, # Fixed seed for reproducibility
        max_episode_steps=64,
        observation_type='MAPF' # Helper for full observability if needed
    )

    # 2. Create Environment
    env = pogema_v0(grid_config=grid_config)
    env.reset()
    
    # 3. Extract Grid and Tasks
    # env.grid is the Grid object in recent pogema versions or we access it via accessors
    # In the reviewed code, env is likely the gym wrapper.
    # We can try to access the underlying grid.
    
    # Try to access internal grid
    if hasattr(env, 'grid'):
        grid_obj = env.grid
    elif hasattr(env, 'unwrapped'):
        if hasattr(env.unwrapped, 'grid'):
            grid_obj = env.unwrapped.grid
        else:
            print("Could not find grid object in env")
            return
    else:
        print("Could not find grid object")
        return

    # Extract obstacles
    # Pogema: 1 is obstacle, 0 is free
    # CBS: 0 is obstacle, non-zero is free (based on stastart.py line 42: if grid[i][j] == 0: continue)
    
    pogema_obstacles = grid_obj.get_obstacles(ignore_borders=False) # Get full grid including borders if any
    
    # cbs.py expects a numpy array.
    # In cbs/stastart.py: if grid[i][j] == 0 then it is an obstacle.
    # We need to map Pogema Obstacle (1) -> CBS Obstacle (0)
    # Pogema Free (0) -> CBS Free (1)
    
    cbs_grid = (pogema_obstacles == 0).astype(int)
    
    print(f"Grid Shape: {cbs_grid.shape}")
    
    # Extract Agent Positions
    # list of (x,y)
    starts = grid_obj.get_agents_xy(ignore_borders=False)
    goals = grid_obj.get_targets_xy(ignore_borders=False)
    
    # Prepare positions dict for CBS: {agent_id: (start, goal)}
    positions = {}
    for i in range(len(starts)):
        # Ensure tuples
        positions[i] = (tuple(starts[i]), tuple(goals[i]))
        
    print(f"Agents: {len(positions)}")
    for i, (s, g) in positions.items():
        print(f"Agent {i}: Start={s}, Goal={g}")
        
    # 4. Run CBS
    print("Initializing CBS...")
    # CBS init(world_state)
    planner = CBS(cbs_grid, need_convert=False)
    
    print("Planning...")
    start_time = time.time()
    result = planner.plan(positions, max_iterations=1000)
    end_time = time.time()
    
    if result:
        print(f"Solution found in {end_time - start_time:.4f} seconds!")
        print(f"Cost: {result.cost}")
        for agent_id, path in result.sol.items():
            print(f"Agent {agent_id} Path: {path}")
            
        # Optional: Validate path validity against static obstacles?
        # (CBS guarantees this but good to sanity check)
    else:
        print("No solution found.")

if __name__ == "__main__":
    run_cbs_on_pogema()
