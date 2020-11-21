export PYTHONPATH=$PYTHONPATH:/data/ContourNet
export NGPUS=1
CUDA_VISIBLE_DEVICES=4 python2 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
	--config-file "configs/slpr_gaussian/test_ub.yaml"
