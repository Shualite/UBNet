# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=2
CUDA_VISIBLE_DEVICES=1,3 python2 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 23667 tools/train_net.py \
	--config-file "configs/ctw/debug_uncertainty_boundary.yaml" \
	--skip-test

