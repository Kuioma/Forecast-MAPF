CODE_ROOT="/disk/deepdata/workspace/sjj_ws/mapf-gpt"
export HYDRA_FULL_ERROR=1
export HF_HOME="${CODE_ROOT}/hugging_face"
conda init 
echo $SHELL

eval "$(/opt/conda/bin/conda shell.bash hook)"
source ~/.bashrc
cd "$CODE_ROOT" || echo "$(pwd)"
conda activate pogema
conda info --envs
nvidia-smi
echo pwd
python generate_dataset.py
#python benchmark.py
