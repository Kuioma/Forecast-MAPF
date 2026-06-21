import numpy as np
from collections import deque
def get_cost_matrix(grid, si, sj):
    """
    计算从 (si, sj) 出发到地图上所有可达点的最短路径距离。
    对应 C++ 中的 get_cost_matrix。
    """
    rows = len(grid)
    cols = len(grid[0])
    # 初始化结果矩阵，-1 表示不可达
    result = np.full((rows, cols), -1, dtype=int)
    
    # 定义移动方向：原地, 上, 下, 左, 右
    moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS 队列
    fringe = deque([(si, sj)])
    result[si, sj] = 0
    
    while fringe:
        curr_i, curr_j = fringe.popleft()
        
        for di, dj in moves:
            ni, nj = curr_i + di, curr_j + dj
            
            # 边界检查
            if 0 <= ni < rows and 0 <= nj < cols:
                # 如果是空地 (grid[ni][nj] == 0) 且尚未访问 (result[ni][nj] < 0)
                if grid[ni, nj] == 0 and result[ni, nj] < 0:
                    result[ni, nj] = result[curr_i, curr_j] + 1
                    fringe.append((ni, nj))
    
    return result

def precompute_cost2go(grid, obs_radius):
    """
    预计算网格中每个非障碍物点到其他点的 cost 矩阵。
    对应 C++ 中的 precompute_cost2go (L33-L41)。
    """
    grid = np.array(grid)
    rows, cols = grid.shape
    cost2go = {}
    
    # 遍历排除边界 radius 后的网格
    for i in range(0, rows):
        for j in range(0, cols):
            # 如果当前点是平地/通路 (0)，则计算其 cost 矩阵
            if grid[i, j] == 0:
                cost2go[(i, j)] = get_cost_matrix(grid, i, j)
                
    return cost2go

def generate_cost2go_obs(cost2go, pos, offset, limit, only_obstacles):
    """
    生成观测矩阵。
    处理边界情况，超出部分填充为 -1。
    """
    if offset == 0:
        return []

    # 转换输入为 numpy 数组提高性能
    if not isinstance(cost2go, np.ndarray):
        cost2go = np.array(cost2go)
    
    rows, cols = cost2go.shape
    pos_i, pos_j = pos
    
    # 目标窗口的理论边界
    i_start, i_end = pos_i - offset, pos_i + offset + 1
    j_start, j_end = pos_j - offset, pos_j + offset + 1

    # 初始化为 -1 (代表障碍/不可达)
    target_size = 2 * offset + 1
    obs = np.full((target_size, target_size), -1, dtype=int)

    # 计算在原图中的有效切片范围
    slice_i_start = max(0, i_start)
    slice_i_end = min(rows, i_end)
    slice_j_start = max(0, j_start)
    slice_j_end = min(cols, j_end)

    # 计算在目标 obs 矩阵中的放置位置
    if slice_i_start < slice_i_end and slice_j_start < slice_j_end:
        pad_i_start = slice_i_start - i_start
        pad_i_end = pad_i_start + (slice_i_end - slice_i_start)
        pad_j_start = slice_j_start - j_start
        pad_j_end = pad_j_start + (slice_j_end - slice_j_start)
        
        obs[pad_i_start:pad_i_end, pad_j_start:pad_j_end] = cost2go[slice_i_start:slice_i_end, slice_j_start:slice_j_end]

    if only_obstacles:
        return (obs < 0).astype(int).tolist()

    # 获取中心点的值来做相对距离转换
    middle_value = obs[offset, offset]

    # 可达区域的处理 (只有在原图内且非障碍的点其 cost 才是 >= 0)
    reachable = (obs >= 0)
    
    # 只有当中心点也是可达时，才进行减法运算
    if middle_value >= 0:
        obs[reachable] -= middle_value
    
    # 超出 limit 的范围处理
    obs[reachable & (obs > limit)] = limit * 2
    obs[reachable & (obs < -limit)] = -limit * 2

    # 不可达区域（包含填充的边界和图内障碍）的处理
    obs[~reachable] = -limit * 4

    return obs.astype(int).tolist()