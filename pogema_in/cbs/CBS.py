import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import stastart as stastart
import heapq
from collections import defaultdict
import numpy as np
import json

class constraint:
    def __init__(self,agent_id,time,position) -> None:
        """
        Initialize a constraint for an agent.

        Parameters:
        agent_id (int): The ID of the agent.
        time (int): The time step at which the constraint applies.
        position (list): The position (x, y) where the constraint applies.
        """
        self.agent_id = agent_id
        self.position = position 
        self.time = time
        self.con_dict = defaultdict(lambda:defaultdict(list))
        self.con_dict[self.agent_id] = {self.time:[tuple(self.position)]}
    def __add__(self, other):
        if other == {}:
            return self
        if self.time not in other.con_dict[self.agent_id]:
            other.con_dict[self.agent_id][self.time] = [tuple(self.position)]
        elif tuple(self.position) not in other.con_dict[self.agent_id][self.time]:
            other.con_dict[self.agent_id][self.time].append(tuple(self.position))
        return other
    def __str__(self) -> str:
        return "agent"+str(self.agent_id)+"position"+str(self.position)


class NTnode:
    def __init__(self, con, sol, terminate_obstacle={}, cost=0, num_conflicts=0) -> None:
        self.con = con  ### 约束--- [constraint]
        self.sol = sol  ### 解 {id: sol}
        self.cost = cost
        self.num_conflicts = num_conflicts
        self.terminate_obstacle = terminate_obstacle
        if cost == 0:
            for value in sol.values():
                self.cost += len(value)
        
    def __lt__(self, other):
        # Tie-breaking: Use number of conflicts as secondary priority
        if self.cost != other.cost:
            return self.cost < other.cost
        return self.num_conflicts < other.num_conflicts

    def __str__(self) -> str:
        return f"constraint: {len(self.con)} items, cost: {self.cost}, conflicts: {self.num_conflicts}"

    def sol_add(self, sol, agent_id):
        sol_dict = self.sol.copy()
        sol_dict[agent_id] = sol
        return sol_dict

class CBS:
    def __init__(self, world_state, need_convert=False, converter=None, world_dir=None) -> None:
        self.planner = stastart.Planner(world_state, world_dir=world_dir)
        self.time = 0
        self.need_convert = need_convert
        if need_convert:
            self.converter = converter
    
    def plan(self,position, max_iterations=100): 
        # position = {agentID:(agentpos,goalpos)}
        """
         Plan the paths for all agents.
 
         Parameters:
         position (dict): A dictionary where keys are agent IDs and values are tuples containing the agent's start position and goal position.
         """
        self.OpenList = []
        sol = {}
        _time = 0
        new_terminate_obstacle = {}

        # 未分配到任务的agent不动视为障碍
        # new_terminate_obstacle = {time:pos}
        if self.need_convert:
            for key, value in position.items():
                position[key] = self.converter.world_to_grid(value)
        
        for id, pos in position.items():
            if pos[1] is None:
                new_terminate_obstacle[id] = (0, pos[0])
        
        for agentID, pos in position.items():
            if pos[1] is None: continue
            result = self.planner.plan(pos[0], pos[1], {}, dict_to(new_terminate_obstacle, None))
            if len(result) == 0: continue
            sol[agentID] = result
            new_terminate_obstacle[agentID] = (len(result), pos[1])
            
        initial_conflicts = detect_conflicts(NTnode({}, sol, new_terminate_obstacle))
        heapq.heappush(self.OpenList, NTnode({}, sol, new_terminate_obstacle, num_conflicts=len(initial_conflicts)))
        
        while self.OpenList:
            _time += 1
            if _time % 50 == 0: print("High-level Iteration:", _time)
            if _time > max_iterations: return None
            
            current = heapq.heappop(self.OpenList)
            conflicts = detect_conflicts(current)
            
            if not conflicts:
                if all(len(i) > 0 for i in current.sol.values()):
                    if self.need_convert:
                        for key, value in current.sol.items():
                            current.sol[key] = self.converter.grid_to_world(value)
                    return current
            
            # Standard CBS branches on ONE conflict to keep the tree manageable
            collision_loc, agents = conflicts[0]
            for agent in agents:
                pos_x, pos_y, time = collision_loc
                new_con_obj = constraint(agent, time, [pos_x, pos_y]) + current.con
                
                new_path = self.planner.plan(position[agent][0], position[agent][1], 
                                            new_con_obj.con_dict[agent], 
                                            dict_to(current.terminate_obstacle, agent))
                
                if len(new_path) > 0:
                    new_sol = current.sol_add(new_path, agent)
                    new_term = current.terminate_obstacle.copy()
                    new_term[agent] = (len(new_path), position[agent][1])
                    
                    # Calculate conflicts for tie-breaking
                    temp_node = NTnode(new_con_obj, new_sol, new_term)
                    num_c = len(detect_conflicts(temp_node))
                    heapq.heappush(self.OpenList, NTnode(new_con_obj, new_sol, new_term, num_conflicts=num_c))

def dict_to(d, agent):
    new_dict = defaultdict(set)
    for key, value in d.items():
        if key != agent:
            new_dict[value[0]].add(tuple(value[1]))
    return new_dict


### return ((loc_x,loc_y,time),[agent1....])
def detect_conflicts(node):
    if not node.sol: return []
    max_len = max(len(p) for p in node.sol.values())
    
    # Pre-process terminate obstacles
    term_pos_map = defaultdict(list) # coord -> [(time_start, agent_id)]
    for a_id, (t_start, pos) in node.terminate_obstacle.items():
        term_pos_map[tuple(pos)].append((t_start, a_id))

    past_pos = {} # agent_id -> pos_tuple
    for t in range(max_len):
        current_pos = {}
        pos_to_agents = defaultdict(list)
        
        for i, path in node.sol.items():
            if t < len(path):
                p = tuple(path[t])
                current_pos[i] = p
                pos_to_agents[p].append(i)
                
                # 1. Vertex-Terminate Conflict
                if p in term_pos_map:
                    for t_start, other_ids in term_pos_map[p]:
                        if t >= t_start and i != other_ids:
                            return [((p[0], p[1], t), [i, other_ids])]
                
                # 2. Swap Conflict (O(N) lookup)
                if i in past_pos:
                    prev_i = past_pos[i]
                    # If someone else is now at my previous, and I am now at their previous
                    for other_i, now_p in current_pos.items():
                        if other_i != i and now_p == prev_i:
                            if past_pos.get(other_i) == p:
                                return [((p[0], p[1], t), [i, other_i])]
            else:
                current_pos[i] = None
        
        # 3. Vertex-Vertex Conflict
        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                return [((pos[0], pos[1], t), agents)]
        
        past_pos = current_pos
    return []


class Converter:
    def __init__(self,cell_size,min_x,min_y) -> None:
        self.cell_size = cell_size
        self.min_x = min_x
        self.min_y = min_y
    
    def grid_to_world(self,list):
        # 网格中第一个参数其实是 y 第二个参数是 x
        result = []
        for pos in list:
            world_x = pos[0]*self.cell_size+self.min_x+0.5*self.cell_size
            world_y = pos[1]*self.cell_size+self.min_y+0.5*self.cell_size
            result.append(tuple([world_x,world_y]))
        return result
    def world_to_grid(self,list):
        result = []
        for pos in list:
            grid_x = int((pos[0]-self.min_x)/self.cell_size)
            grid_y = int((pos[1]-self.min_y)/self.cell_size)
            result.append(tuple([grid_x,grid_y]))
        return result
if __name__ == '__main__':
    path = "D:\\isaac\\isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.windows-x86_64.release\\kuka\\"
    json_path = os.path.join(path,"json_file","grid.json")
    with open(json_path) as json_file:
        grid_json = json.load(json_file)
    grid = np.array(grid_json)
    #A dictionary where keys are agent IDs and values are tuples containing the agent's start position and goal position
    position = {0:[tuple([17.754,1.4]),tuple([44.0,1.408])],1:[tuple([25.2,21.8]),tuple([21.368,28.6])],2:[tuple([17.754,59.15]),tuple([32.8,49.15])]}
    print(position)
    conveter = Converter(0.5,12.59,-2.6)
    cbs = CBS(grid,need_convert=True,converter=conveter)
    result = cbs.plan(position, max_iterations=1000)
    print("result:",result.sol[2])