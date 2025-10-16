set -x
source /home/c30061641/cann/83rc1b070/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

bash train.sh tasks/omni/train_qwen2_5_vl.py configs/multimodal/qwen2_vl/qwen2_5_vl.yaml
