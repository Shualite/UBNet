# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=2
CUDA_VISIBLE_DEVICES=4,5 python2 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
	--config-file "configs/ctw/ubnet_baseline.yaml" 
