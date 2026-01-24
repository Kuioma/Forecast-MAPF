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
    def __init__(self,con,sol,terminate_obstacle={},cost=0) -> None:
        self.con = con  ###约束--- [constraint]
        self.sol = sol  ###解 {id: sol}
        self.cost = cost
        self.terminate_obstacle = terminate_obstacle  ###agentid: time position
        for key,value in sol.items():
            self.cost += len(value)
        
    def __lt__(self,other):
        return self.cost < other.cost
    def __str__(self) -> str:
        return "constraint:"+str(self.con)+"solution:"+str(self.sol)+"cost:"+str(self.cost)
    def sol_add(self,sol,agent_id):
        sol_dict = self.sol.copy()
        sol_dict[agent_id] = sol
        return sol_dict

class CBS:
    def __init__(self,world_state,need_convert = False,converter = None,world_dir = None) -> None:
        self.planner = stastart.Planner(world_state,world_dir=world_dir)
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
        cost = 0
        new_terminate_obstacle = {}
        #        self.con_dict[self.agent_id] = {self.time:[tuple(self.position)]}

        # 未分配到任务的agent不动视为障碍
        # new_terminate_obstacle = {time:pos}
        if self.need_convert:
            for key,value in position.items():
                position[key] = self.converter.world_to_grid(value)
        for id,pos in position.items():
            if pos[1] == None:
                new_terminate_obstacle[id] = (0,pos[0])
        for agentID,pos in position.items():
            agentPos = pos[0]
            taskPos = pos[1]
            if taskPos == None:
                continue
            result = {agentID:self.planner.plan(agentPos,taskPos,{},dict_to(new_terminate_obstacle,None))}
            sol.update(result)
            if len(result)!=0:             ### 达到终点后视为障碍
                new_terminate_obstacle[agentID] = (len(result[agentID]),taskPos)
        heapq.heappush(self.OpenList,NTnode({},sol,new_terminate_obstacle))
        while(self.OpenList):
            _time += 1
            if _time%50==0:print("time:",_time)
            if(_time>max_iterations):
                return None
            current = heapq.heappop(self.OpenList)
            conflict = detect_conflicts(current)
            # 无冲突则返回
            if(conflict == []):
                if(all(len(i)>0 for i in current.sol.values())):
                    if self.need_convert:
                        print(current)
                        for key,value in current.sol.items():
                            current.sol[key] = self.converter.grid_to_world(value)
                    return current
            conflict_dict = defaultdict(list)
            # ((x,y,t),agent)
            for i in conflict:
                for agent in i[1]:
                    conflict_dict[agent].append(i[0])
            ###生成新的节点
            for agent,value in conflict_dict.items():
                ###之前的约束加上现在的约束
                pos_x,pos_y,time = value[0]
                new_constraint = constraint(agent,time,[pos_x,pos_y])+current.con
                new_result = self.planner.plan(position[agent][0],position[agent][1],new_constraint.con_dict[agent],dict_to(current.terminate_obstacle,agent))
                new_terminate_obstacle = current.terminate_obstacle.copy()
                if len(new_result)!=0:
                    new_terminate_obstacle[agent] = (len(new_result),position[agent][1])
                new_result = current.sol_add(new_result,agent)
                heapq.heappush(self.OpenList,NTnode(new_constraint,new_result,new_terminate_obstacle))


def dict_to(dict,agent):
    """
     Convert the terminate_obstacle dictionary to a new format excluding the specified agent.
 
     Parameters:
     dict (defaultdict): The dictionary containing agent termination obstacles.
     agent (int): The ID of the agent to exclude from the new dictionary.
 
     Returns:
     defaultdict: A new dictionary with the same structure, excluding the specified agent.
    """
    new_dict = defaultdict(set)
    ###不能带自己的否则当新解路径长过原本解的时候必定无解
    for key,value in dict.items():
        if key != agent:
            new_dict[value[0]].add(value[1])
    return new_dict


### return ((loc_x,loc_y,time),[agent1....])
def detect_conflicts(node):
    conflict = []
    max_len = max(len(p) for p in node.sol.values())

    past = {}
    ### 碰撞冲突检测
    for t in range(max_len):
        temp = {}
        locations = defaultdict(list)  ###（x,y,t）:(agentid)
        for i,path in node.sol.items():
            if len(path)>t:
                temp.update({i:(path[t][0],path[t][1])})
                #temp.append((i,(path[t][0],path[t][1])))
            else:
                temp.update({i:(-1,-1)})
                #temp.append((i,(-1,-1)))

            if len(path)>t:
                locations[(path[t][0],path[t][1],t)].append(i) ###i agent
                ###检查终止障碍
                for agent,value in node.terminate_obstacle.items():            ###agentid: time positio
                    if t>=value[0] and tuple(value[1])==(path[t][0],path[t][1]):
                            ###有终止冲突
                            locations[(value[1][0],value[1][1],t)].append(i)
        ### 交换冲突
        swap_index = detect_swap(past, temp)
        for i,j in swap_index:
            conflict.append(((temp[i][0],temp[i][1],t),[i]))
        past = temp
        ### 碰撞冲突
        for loc,agent in locations.items():
            if len(agent)>1:   ###
                conflict.append((loc,agent))  ###loc_x,loc_y,time,agent
    return conflict

def detect_swap(past, now):
    swap_indices = []
    for i, (p1, p2) in past.items():
        for j, (n1, n2) in now.items():
            if(p1,p2) == (-1,-1) or (n1,n2) == (-1,-1):
                continue
            if (p1, p2) == (n1, n2) and past[j] == now[i]:
                if i != j:
                    swap_indices.append((i, j))
    swap_indices = list(set(swap_indices))
    return swap_indices


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