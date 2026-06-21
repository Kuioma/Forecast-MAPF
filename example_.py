import argparse
from pathlib import Path

import torch
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import run_episode
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
# from gpt.inference_multi_action import MultiActionGPTInference,MultiActionMAPFGPTInferenceConfig
# from gpt.inference_multi_action_with_real_action import MultiActionGPTInference,MultiActionMAPFGPTInferenceConfig
# from pogema_benchmark_main.algorithms.scrimp.inference import SCRIMPInference, SCRIMPInferenceConfig
# from pogema_benchmark_main.algorithms.follower.follower_python.inference import FollowerInference, FollowerInferenceConfig
# from pogema_benchmark_main.algorithms.follower.follower_python.preprocessing import follower_preprocessor



def main():
    parser = argparse.ArgumentParser(description='MAPF-GPT Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=32, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='validation-random-seed-001', help='Map name (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda, cpu, mps (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=128,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--model', type=str, choices=['2M', '6M', '85M'], default='2M',
                        help='Model to use: 2M, 6M, 85M (default: %(default)s)')

    # loading maps from eval folders
    for maps_file in Path("eval_configs_").rglob('maps.yaml'):
        with open(maps_file, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in ToolboxRegistry.get_maps():
            print(map_)
        return
    args.map_name = "puzzle-15"
    args.num_agents = 4
    env_cfg = Environment(
        with_animation=args.animation,
        observation_type="MAPF",
        on_target="nothing",
        map_name=args.map_name,
        max_episode_steps=args.max_episode_steps,
        num_agents=args.num_agents,
        seed=args.seed,
        obs_radius=5,
        collision_system="soft",
    )
    ### addition
    # env_cfg.targets_xy = [(2,3),(1,0),(2,0)]
    # env_cfg.agents_xy = [(2,3),(1,3),(2,4)]

    # pytorch seeding
    torch_seed = 1
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
    torch.backends.mps.is_available()
    torch.backends.cudnn.deterministic = True

    env = create_eval_env(env_cfg)
    # algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'/home/mapf-gpt/weights/model-85m.pt', device=args.device))
    # args.model = "MAPF-85m"
    # algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'/home/mapf-gpt/out_multi_action/9.15_ckpt .pt', device=args.device))
    # algo = MultiActionGPTInference(MultiActionMAPFGPTInferenceConfig(path_to_weights=f'/home/mapf-gpt/out_multi_action/9.15_ckpt .pt', device=args.device))
    algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'/home/mapf-gpt/out_multi_action/morden_style_nomask_40000/6m/ckpt.pt', device=args.device,predict_next_action=True))
    args.model = "Forecast-MAPF(6m)"
    # algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'weights/model-{args.model}.pt', device=args.device))
    algo.reset_states()
    results = run_episode(env, algo)

    svg_path = f"svg/{args.map_name}-{args.model}-seed-{args.seed}.svg"
    env.save_animation(svg_path)
    ToolboxRegistry.info(f'Saved animation to: {svg_path}')

    ToolboxRegistry.success(results)


if __name__ == "__main__":
    main()
