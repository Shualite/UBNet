# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 23613 tools/train_net.py \
	--config-file "configs/stack/stack_gpu_2.yaml" \
	--skip-test
