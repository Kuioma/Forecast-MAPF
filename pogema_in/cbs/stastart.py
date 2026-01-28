import numpy as np
from typing import Tuple, List, Dict, Set
from heapq import heappush, heappop
import json

class State:

    def __init__(self, pos: np.ndarray, time: int, g_score: int, h_score: int):
        self.pos = pos
        self.time = time
        self.g_score = g_score
        self.f_score = g_score + h_score

    def __hash__(self) -> int:   ###本质就是把非数值对象给转换成一个数
        return hash((int(self.pos[0]), int(self.pos[1]), self.time))

    def pos_equal_to(self, pos: np.ndarray) -> bool:
        return np.array_equal(self.pos, pos)   ###判断数组元素是否完全相等

    def __lt__(self, other: 'State') -> bool:
        return self.f_score < other.f_score

    def __eq__(self, other: 'State') -> bool:
        return self.__hash__() == other.__hash__()

    def __str__(self):
        return 'State(pos=[' + str(self.pos[0]) + ', ' + str(self.pos[1]) + '], ' \
               + 'time=' + str(self.time) + ', fscore=' + str(self.f_score) + ')'

    def __repr__(self):
        return self.__str__()
    
class NeighbourTable:
    #             Current  up   up-right right down-right  down    down-left  left    up-left
    directions = [(0, 0), (0,1),  (1,0),    (0,-1),    (-1,0)]
    def __init__(self, colaborate_grid: np.ndarray,grid,world_dir=None):
        dimx, dimy = len(colaborate_grid), len(colaborate_grid[0])
        table = dict()
        for i in range(dimx):
            for j in range(dimy):
                neighbours = []
                if grid[i][j] == 0:  # Obstacle
                    continue
                for dx, dy in self.directions:
                    if np.all(world_dir != None):
                        if world_dir[i][j] == '@' and (dx,dy) == (0,-1):
                            continue
                        if world_dir[i][j] == '#' and (dx,dy) == (-1,0):
                            continue
                        if world_dir[i][j] == '$' and (dx,dy) == (0,1):
                            continue
                        if world_dir[i][j] == '!':
                            continue
                    x, y = i + dy, j + dx,
                    if x >= 0 and x < dimx and y >= 0 and y < dimy and not (grid[x][y] == 0):  # Obstacle
                        neighbours.append(colaborate_grid[x][y])
                table[self.hash(colaborate_grid[i][j])] = np.array(neighbours)  ##hash值作为key
        self.table = table

    def lookup(self, position: np.ndarray) -> np.ndarray:
        return self.table[self.hash(position)]
    
    @staticmethod
    def hash(grid_pos: np.ndarray) -> int:
        return tuple(grid_pos)
    
class Planner:

    def __init__(self,world,world_dir = None):
        self.grid = world
        print(world)
        # 生成值为坐标的矩阵
        self.colaborate_grid = np.zeros((world.shape[0],world.shape[1],2),dtype=int)
        for i in range(world.shape[0]):
            for j in range(world.shape[1]):
                self.colaborate_grid[i][j] = np.array([i,j],dtype=int)
        # Make a lookup table for looking up neighbours of a grid
        self.neighbour_table = NeighbourTable(self.colaborate_grid,world,world_dir)  ###找到的所有邻居


    '''
    An admissible and consistent heuristic for A*
    '''
    @staticmethod
    def h(start: np.ndarray, goal: np.ndarray) -> int:
        return int(np.linalg.norm(start-goal, 1))  # L1 norm   ###曼哈顿距离

    @staticmethod
    def l2(start: np.ndarray, goal: np.ndarray) -> int:
        return int(np.linalg.norm(start-goal, 2))  # L2 norm   ###欧几里得距离

    '''
    Check whether the nearest static obstacle is within radius
    '''
    def safe_static(self, grid_pos: np.ndarray) -> bool:
        return self.grid[grid_pos[0],grid_pos[1]] != -1

    '''
    Space-Time A*
    '''
    def plan(self, start: Tuple[int, int], # 坐标
                   goal: Tuple[int, int],  # 坐标
                   dynamic_obstacles: Dict[int, Set[Tuple[int, int]]],##The keys for are integers representing time and the values are sets of coordinates.
                   semi_dynamic_obstacles:Dict[int, Set[Tuple[int, int]]] = None,
                   max_iter:int = 5000,
                   debug:bool = False) -> np.ndarray:

        # Prepare dynamic obstacles
        dynamic_obstacles = dict((k, np.array(list(v))) for k, v in dynamic_obstacles.items())
        # Assume dynamic obstacles are agents with same radius, distance needs to be 2*radius
        def safe_dynamic(grid_pos: np.ndarray, time: int) -> bool:
            nonlocal dynamic_obstacles  ##nonlocal表示的就是非局部的意思
            return all((grid_pos != obstacle).any()
                       for obstacle in dynamic_obstacles.setdefault(time, np.array([])))

        # # Prepare semi-dynamic obstacles, consider them static after specific timestamp
        if semi_dynamic_obstacles is None:
            semi_dynamic_obstacles = dict()
        else:
            semi_dynamic_obstacles = dict((k, np.array(list(v))) for k, v in semi_dynamic_obstacles.items())
            
        def safe_semi_dynamic(grid_pos: np.ndarray, time: int) -> bool: #检查终止障碍冲突
            nonlocal semi_dynamic_obstacles
            pos_tuple = tuple(grid_pos)
            for timestamp, obstacles in semi_dynamic_obstacles.items():
                if time >= timestamp:
                    if pos_tuple in obstacles:
                        return False
            return True

        start = np.array(start)
        goal = np.array(goal)
        # Initialize the start state
        s = State(start, 0, 0, self.h(start, goal))

        open_set = [s]
        seen_in_open = {s} # Use set for O(1) lookup
        closed_set = set()

        # Keep track of parent nodes for reconstruction
        came_from = dict()

        iter_ = 0
        while open_set and iter_ < max_iter:
            iter_ += 1
            current_state = heappop(open_set)
            
            if current_state.pos_equal_to(goal):
                if debug:
                    print('STA*: Path found after {0} iterations'.format(iter_))
                return self.reconstruct_path(came_from, current_state)

            closed_set.add(current_state)
            epoch = current_state.time + 1
            
            for neighbour in self.neighbour_table.lookup(current_state.pos):
                neighbour_state = State(neighbour, epoch, current_state.g_score + 1, self.h(neighbour, goal))
                
                # Check if visited
                if neighbour_state in closed_set:
                    continue
                
                # Avoid obstacles
                if not self.safe_static(neighbour) or \
                   not safe_dynamic(neighbour, epoch) or \
                   not safe_semi_dynamic(neighbour, epoch): 
                    continue

                # Add to open set
                if neighbour_state not in seen_in_open:
                    came_from[neighbour_state] = current_state
                    seen_in_open.add(neighbour_state)
                    heappush(open_set, neighbour_state)

        if debug:
            print('STA*: Open set is empty, no path found.')
        return np.array([])

    '''
    Reconstruct path from A* search result
    '''
    def reconstruct_path(self, came_from: Dict[State, State], current: State) -> np.ndarray:
        total_path = [current.pos]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current.pos)
        return np.array(total_path[::-1])

# def test():
#     env = env_gym.MAPFEnv()
#     env.set_world_fix(fix_map.map1,fix_map.goals,3)
#     world = env.world.state
#     planner = Planner(world)
#     for index,i in enumerate(env.world.agents):
#         result = planner.plan(i,env.world.agent_goals[index],{4:[[4,4]]},{6:[[4,4]]})
#         print(result)

if __name__ == '__main__':
    test()


