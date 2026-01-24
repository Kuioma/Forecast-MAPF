CODE_ROOT="/disk/deepdata/workspace/sjj_ws/mapf-gpt"
export HYDRA_FULL_ERROR=1
export HF_HOME="${CODE_ROOT}/hugging_face"
cd "$CODE_ROOT" || echo "$(pwd)"
conda activate pogema
torchrun --nproc_per_node=8 train_multi_action.py
