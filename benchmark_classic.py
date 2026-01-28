from pathlib import Path
import sys
import itertools
import copy
sys.path.append("/home/mapf-gpt/pogema_benchmark_main")
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
# from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

PROJECT_NAME = "Benchmark"
BASE_PATH = Path("eval_configs_")                  


def ensure_weights(eval_config):
    for algo_name, algo_cfg in eval_config['algorithms'].items():
        ToolboxRegistry.create_algorithm(algo_cfg['name'], **algo_cfg)


def main(disable_wandb=True):
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_eval_env, Environment)
    
    from pogema_in.cbs.inference import CBSInference, CBSInferenceConfig
    ToolboxRegistry.register_algorithm("CBS", CBSInference, CBSInferenceConfig)


    folder_names = [
        "01-random-CBS",
        "02-mazes-CBS",
        "03-warehouse-CBS",
        "04-movingai-CBS",
        "05-puzzles-CBS"
    ]

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        # ensuring model weights are downloaded
        ensure_weights(base_config)

        eval_dir = BASE_PATH / folder
        
        # Unroll grid search
        # env_config = base_config.get('environment', {})
        # grid_params = {}
        # fixed_params = {}
        
        # for k, v in env_config.items():
        #     if isinstance(v, dict) and 'grid_search' in v:
        #         grid_params[k] = v['grid_search']
        #     else:
        #         fixed_params[k] = v
                
        # keys = list(grid_params.keys())
        # values = list(grid_params.values())
        
        # combinations = list(itertools.product(*values))
        # total_runs = len(combinations)
        # print(f"Total experiments to run: {total_runs}")
        
        # for i, combo in enumerate(combinations):
        #     current_params = dict(zip(keys, combo))
        #     print(f"\nRunning experiment {i+1}/{total_runs}: {current_params}")
            
        #     # Create a specific config for this run
        #     run_config = copy.deepcopy(base_config)
            
        #     # Replace grid_search with specific value
        #     for k, v in current_params.items():
        #         run_config['environment'][k] = v
                
        #     # Run evaluation
        try:
            evaluation(base_config, eval_dir=eval_dir)
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")

if __name__ == "__main__":
    main()
