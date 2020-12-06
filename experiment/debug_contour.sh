# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 23633 tools/train_net.py \
	--config-file "configs/afub/ctw/debug_rrpn.yaml" \
	--skip-test
