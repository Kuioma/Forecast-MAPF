# Forecast-MAPF: Imitation Learning for Multi-Agent Pathfinding at Scale

<div align="center" dir="auto">
   <p dir="auto"><img src="svg/puzzles.svg" alt="Follower" style="max-width: 80%;"></p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Kuioma/Forecast-MAPF/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/Weights-Forecast--MAPF-blue?logo=huggingface)](https://huggingface.co/Kuioma00614/Forecast-MAPF/tree/main)
</div>

The repository consists of the following crucial parts:

- `example_.py` - an example of code to run the Forecast-MAPF approach.
- `example.py` - an example of code to run the standard CBS algorithm.
- `benchmark.py` - a script that launches the evaluation of the Forecast-MAPF model on the POGEMA benchmark set of maps.
- `generate_dataset.py` - a script that generates the training dataset.
- `train_multi_action.py` - a script that launches the training of the Forecast-MAPF model.
- `eval_configs` - a folder that contains configs from the POGEMA benchmark. Required by the `benchmark.py` script.
- `dataset_configs` - a folder that contains configs to generate training and validation datasets. Required by the `generate_dataset.py` script.

## Installation

It's recommended to utilize Docker to build the environment compatible with Forecast-MAPF code. The `docker` folder contains both `Dockerfile` and `requirements.txt` files to successfully build an appropriate container.

```
cd docker & sh build.sh
```

## Running an example

To test Forecast-MAPF, you can simply run the `example_.py` script. By default, it uses the Forecast-MAPF 6M model, but this can be adjusted.  

```
python3 example_.py
```

In addition to statistics about SoC, success rate, etc., you will also get an SVG file that animates the solution found by Forecast-MAPF, which will be saved to the `svg/` folder.


## Running evaluation

You can run the `benchmark.py` script, which will run the Forecast-MAPF model on all the scenarios from the POGEMA benchmark.

```
python3 benchmark.py
```

The results will be stored in the `eval_configs` folder near the corresponding configs. They can also be logged into wandb. The tables with average success rates will be displayed directly in the console.

## Dataset

To train Forecast-MAPF, we generated a training dataset consisting of tensor-action pairs. We used the LaCAM approach as the source of expert data. 90% of the data in the dataset was obtained from maze maps, while the remaining 10% was sourced from random maps.

### Generating the dataset

If you want to generate the dataset from scratch or create a modified version, use the provided script. It handles all necessary steps, including instance generation (via POGEMA), solving instances (via LaCAM), generating and filtering observations, shuffling the data, and saving it into multiple `.arrow` files for efficient in-memory operation.

```
python3 generate_dataset.py
```

Please note that generating the full training dataset requires significant disk space. Additionally, solving all instances with LaCAM takes significant time. You can reduce the time and space needed, as well as the final dataset size, by modifying the configuration files in `dataset_configs` (e.g., adjusting the number of seeds or reducing the number of maps).

## Running training of Forecast-MAPF

To train Forecast-MAPF from scratch or to fine-tune the existing models on other datasets (if you occasionally have such ones), you can use the `train_multi_action.py` script. By providing it a config, you can adjust the parameters of the model and training setup. The script utilizes DDP, which allows training the model on multiple GPUs simultaneously. By adjusting the `nproc_per_node` value, you can choose the number of GPUs that are used for training.

```
torchrun --standalone --nproc_per_node=1 train_multi_action.py gpt/config-multi-action.py
```

## Citation:

```bibtex
@inproceedings{andreychuk2025mapf,
  title={{MAPF-GPT}: Imitation learning for multi-agent pathfinding at scale},
  author={Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr and Skrynnik, Alexey},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={22},
  pages={23126--23134},
  year={2025}
}
```
