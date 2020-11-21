# export NCCL_P2P_DISABLE=1
export PYTHONPATH=$PYTHONPATH:/DATA/disk1/fsy_scenetext/ContourNet_v2t
export NGPUS=1
CUDA_VISIBLE_DEVICES=1 python2 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 23633 tools/train_net.py \
	--config-file "configs/slpr_gaussian/train_ub_deformable.yaml" \
	--skip-test