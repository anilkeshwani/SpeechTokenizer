CONFIG="config/spt_base_cfg.json"

# NPROC_PER_NODE=4
# CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun \
#     --nnode 1 \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_port 50025  \
# train_example.py \
#     --config ${CONFIG} \

CUDA_VISIBLE_DEVICES=1,2,6,7 accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    scripts/train_example.py --config ${CONFIG}
