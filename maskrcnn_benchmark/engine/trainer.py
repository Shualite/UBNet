# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
import torch.distributed as dist
from torch import nn

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# from apex import amp

from tensorboardX import SummaryWriter

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    distributed,
    eval_period=-1,
):  
    
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    if eval_period != -1:
        iou_types = ("bbox",)
        if cfg.MODEL.BOUNDARY_ON:
            iou_types = iou_types + ("bo",)
        if cfg.MODEL.UB_ON:
            iou_types = iou_types + ("ub",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(
            cfg,
            is_train=False, 
            is_distributed=distributed)

    
    writer_loss = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tensorboard_loss'))

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    
    # def inplace_relu(m):
    #     classname = m.__class__.__name__
    #     if classname.find('ReLU') != -1:
    #         m.inplace=True
            
    # model.apply(inplace_relu)

    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, img_path) in enumerate(data_loader, start_iter):      
        
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        if isinstance(targets[0], list):
            targets = [[target[0].to(device) for target in targets],
                       [target[1].to(device) for target in targets]]
        else:
            targets = [target.to(device) for target in targets]
        
        # print(img_path)
        loss_dict = model(images, targets)

        del targets
        del images
        

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        
        # with amp.scale_loss(losses, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        # for i, (name, param) in enumerate(model.named_parameters()):
        #     if 'bn' not in name and param.requires_grad:
        #         writer.add_histogram(str(iteration), param, i)
        

        # grad0 = [x['params'][0].grad.sum() for x in optimizer.param_groups]
        # all_grad0 = sum(grad0)
        # print(all_grad0)
        # grad0 = torch.tensor(grad0)

        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if cfg.SOLVER.GRADIENT_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRADIENT_CLIP)

        # grad = [x['params'][0].grad.sum() for x in optimizer.param_groups]
        # grad_tensors = [x['params'][0].grad for x in optimizer.param_groups]
        # all_grad = sum(grad)
        # grad = torch.tensor(grad)
        # print(grad)
        # print(all_grad)

        if cfg.LOCK:
            import ipdb;ipdb.set_trace()


        # if torch.isnan(grad).sum() == 0:
        #     optimizer.step()
        # else:
        #     print(img_path)
        
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        losses = losses.float()
        del losses, loss_dict, loss_dict_reduced, losses_reduced
        
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            
            if writer_loss is not None:
                if 'loss_objectness' in meters.meters.keys() and 'loss_rpn_box_reg' in meters.meters.keys():
                    writer_loss.add_scalars('rpn', {'cls': meters.meters['loss_objectness'].median, 'reg': meters.meters['loss_rpn_box_reg'].median}, iteration)
                if 'loss_classifier' in meters.meters.keys() and 'loss_box_reg' in meters.meters.keys():
                    writer_loss.add_scalars('boxhead', {'cls': meters.meters['loss_classifier'].median, 'reg': meters.meters['loss_box_reg'].median}, iteration)
                # if 'loss_ub' in meters.meters.keys():
                #     writer_loss.add_scalar('ubhead', meters.meters['loss_ub'].median, iteration)
                    
                writer_loss.flush()

            del meters
            meters = MetricLogger(delimiter="  ")
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            torch.cuda.empty_cache()

        
        
        if eval_period != -1 and iteration % eval_period == 0 and iteration >= cfg.SOLVER.EVAL_BEGIN:
            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                )
                synchronize()

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    writer_loss.close()
