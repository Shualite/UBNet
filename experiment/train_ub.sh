# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/DATA/disk1/fsy_scenetext/ContourNet_v2
# export NGPUS=2
# CUDA_VISIBLE_DEVICES=0,1 python2 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 24743 tools/train_net.py \
# 	--config-file "configs/slpr_gaussian/ctw/train_ub_rotate.yaml" \
# 	--skip-test
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 23644 tools/train_net.py \
	--config-file "configs/slpr_gaussian/ctw/train_ub_ciou.yaml" \
	--skip-test