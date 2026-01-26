from typing import List, Tuple, Dict
import dataclasses


@dataclasses.dataclass
class InputParameters:
    """
    一个用于保存配置参数的数据类。

    属性:
        cost2go_value_limit (int): cost2go 值的绝对值上限。
        num_agents (int): 系统中的最大代理数量。
        num_previous_actions (int): 每个代理要考虑的先前动作的数量。
        context_size (int): 编码输出的固定大小（上下文长度）。
    """
    cost2go_value_limit: int
    num_agents: int
    num_previous_actions: int
    context_size: int = 256

@dataclasses.dataclass
class AgentsInfo:
    """
    一个用于保存单个代理信息的数据类。

    属性:
        relative_pos (Tuple[int, int]): 代理的相对位置 (x, y)。
        relative_goal (Tuple[int, int]): 代理的相对目标 (x, y)。
        previous_actions (List[str]): 代理先前执行的动作列表。
        next_action (str): 代理将要执行的下一个动作。
    """
    relative_pos: Tuple[int, int]
    relative_goal: Tuple[int, int]
    previous_actions: List[str]
    next_action: str

class Encoder:
    """
    该类根据给定的 cost2go 地图和代理信息，将高级状态信息编码为整数序列。
    """
    def __init__(self, cfg: InputParameters):
        """
        初始化编码器并构建词汇表。

        参数:
            cfg (InputParameters): 包含编码器配置的对象。
        """
        self.cfg = cfg

        # 1. 定义词汇表中的符号范围
        coord_range = list(range(-cfg.cost2go_value_limit, cfg.cost2go_value_limit + 1))
        coord_range.extend([
            -cfg.cost2go_value_limit * 4,
            -cfg.cost2go_value_limit * 2,
            cfg.cost2go_value_limit * 2,
        ])

        actions_range = ['n', 'w', 'u', 'd', 'l', 'r']
        
        # 使用格式化字符串生成 4 位二进制表示
        next_action_range = [f'{i:04b}' for i in range(16)]

        # 2. 构建从符号到整数索引的映射
        self.int_vocab: Dict[int, int] = {}
        self.str_vocab: Dict[str, int] = {}
        idx = 0

        for token in coord_range:
            self.int_vocab[token] = idx
            idx += 1
        
        for token in actions_range:
            self.str_vocab[token] = idx
            idx += 1

        for token in next_action_range:
            self.str_vocab[token] = idx
            idx += 1
        
        self.str_vocab["!"] = idx # 用于填充的特殊符号

    def encode(self, agents: List[AgentsInfo], cost2go: List[List[int]]) -> List[int]:
        """
        将 cost2go 地图和代理信息编码为一个固定长度的整数向量。

        参数:
            agents (List[AgentsInfo]): 当前所有代理的信息列表。
            cost2go (List[List[int]]): 表示环境中每个点成本的 2D 网格。

        返回:
            List[int]: 组合并填充到 context_size 的编码整数序列。
        """
        # 1. 编码 cost2go 地图
        # 将 2D cost2go 列表扁平化，并将每个值映射到其词汇索引
        cost2go_indices = [
            self.int_vocab[value] for row in cost2go for value in row
        ]
        
        # 2. 编码每个代理的信息
        agents_indices = []
        for agent in agents:
            # 编码坐标（相对位置和目标）
            agents_indices.extend([
                self.int_vocab[agent.relative_pos[0]],
                self.int_vocab[agent.relative_pos[1]],
                self.int_vocab[agent.relative_goal[0]],
                self.int_vocab[agent.relative_goal[1]]
            ])
            # 编码先前的动作
            agents_indices.extend(
                self.str_vocab[action] for action in agent.previous_actions
            )
            # 编码下一个动作
            agents_indices.append(self.str_vocab[agent.next_action])

        # 3. 填充代理信息
        # 如果代理数量少于 cfg.num_agents，则用填充符号进行填充
        num_padding_agents = self.cfg.num_agents - len(agents)
        if num_padding_agents > 0:
            # 每个代理的 token 数量：4个坐标 + num_previous_actions + 1个下一个动作
            tokens_per_agent = 4 + self.cfg.num_previous_actions + 1
            padding_token_index = self.str_vocab["!"]
            agents_indices.extend(
                [padding_token_index] * (num_padding_agents * tokens_per_agent)
            )

        # 4. 组合并填充到最终的 context_size
        result = cost2go_indices + agents_indices
        
        padding_needed = self.cfg.context_size - len(result)
        if padding_needed > 0:
            result.extend([self.str_vocab["!"]] * padding_needed)
        
        # 注意：如果组合结果超过 context_size，此实现不会截断它，
        # 这与原始 C++ 代码的行为一致。
        return result

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 定义配置
    config = InputParameters(
        cost2go_value_limit=8,
        num_agents=4,
        num_previous_actions=1
    )

    # 2. 创建编码器实例
    encoder = Encoder(config)

    # 3. 创建示例输入数据
    # 假设 cost2go 是一个 5x5 的地图
    cost2go_map = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ]
    
    # 假设我们有两个代理
    agents_data = [
        AgentsInfo(
            relative_pos=(1, -1),
            relative_goal=(5, 2),
            previous_actions=['n'],
            next_action='0010'
        ),
        AgentsInfo(
            relative_pos=(0, 3),
            relative_goal=(-2, 4),
            previous_actions=['r'],
            next_action='1100'
        )
    ]

    # 4. 执行编码
    encoded_vector = encoder.encode(agents_data, cost2go_map)

    print(f"输入配置: {config}")
    print("-" * 20)
    print("输入代理信息:")
    for ag in agents_data:
        print(f"- {ag}")
    print("-" * 20)
    print(f"编码后的向量 (长度: {len(encoded_vector)}):")
    print(encoded_vector[:39])
    
    # 验证填充：由于 num_agents=4 而我们只有 2 个代理，
    # 编码向量中应包含 2 个代理的填充。
    # 每个代理的 token 数量 = 4 (坐标) + 1 (先前动作) + 1 (下一个动作) = 6
    # 填充 token 数量 = 2 (代理) * 6 (token/代理) = 12
    # cost2go token 数量 = 5 * 5 = 25
    # 真实代理 token 数量 = 2 * 6 = 12
    # 总 token 数量 = 25 + 12  = 37
    # 填充到 256 的总长度
    assert len(encoded_vector) == config.context_size
    print(f"\n断言成功：向量长度与配置的 context_size ({config.context_size}) 匹配。")