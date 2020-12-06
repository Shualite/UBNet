# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=1
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
	--config-file "configs/afub/ctw/test_rrpn.yaml"
