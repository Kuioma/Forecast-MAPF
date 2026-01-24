from pathlib import Path
import sys
sys.path.append("/home/mapf-gpt/pogema_benchmark_main")
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
# from algorithms.scrimp.inference import SCRIMPInference, SCRIMPInferenceConfig
# from algorithms.follower.follower_python.inference import FollowerInference, FollowerInferenceConfig
# from algorithms.follower.follower_python.preprocessing import follower_preprocessor

PROJECT_NAME = "Benchmark"
BASE_PATH = Path("eval_configs_")                  


def ensure_weights(eval_config):
    for algo_name, algo_cfg in eval_config['algorithms'].items():
        ToolboxRegistry.create_algorithm(algo_cfg['name'], **algo_cfg)


def main(disable_wandb=True):
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_eval_env, Environment)
    ToolboxRegistry.register_algorithm(
        "MAPF-GPT", MAPFGPTInference, MAPFGPTInferenceConfig
    )
    
    from pogema_in.cbs.inference import CBSInference, CBSInferenceConfig
    ToolboxRegistry.register_algorithm("CBS", CBSInference, CBSInferenceConfig)
    # ToolboxRegistry.register_algorithm('SCRIMP', SCRIMPInference, SCRIMPInferenceConfig)
    # ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)


    folder_names = [
        "01-random",
        "02-mazes",
        "03-warehouse",
        "04-movingai",
        "05-puzzles",
    ]

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        # ensuring model weights are downloaded
        ensure_weights(evaluation_config)

        eval_dir = BASE_PATH / folder
        # initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        # save_evaluation_results(eval_dir)


if __name__ == "__main__":
    main()
