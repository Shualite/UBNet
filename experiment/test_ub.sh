export PYTHONPATH=$PYTHONPATH:/DATA/disk1/fsy_scenetext/ContourNet_v2
export NGPUS=1
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
	--config-file "configs/afub/ic15/test_rrpn.yaml"
