# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import os
import time
import torch
import torch.nn as nn
import torch.distributed

from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from models import build_model, build_gdino_model
from data import (build_dataset, build_sampler, build_dataloader, 
                  build_dataloader_w_sen, build_dataloader_w_cat_cap)
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank, set_seed, is_main_process, \
    distributed_world_size
from utils.nested_tensor import tensor_list_to_nested_tensor
# from models.memotr import MeMOTR
from models.my_groundingdino_track import GroundingDINO as Tracknet
from structures.track_instances import TrackInstances
from models.criterion import build as build_criterion, ClipCriterion
from models.utils import get_model, save_checkpoint, load_checkpoint
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from log.logger import Logger, ProgressLogger
from log.log import MetricLog
from models.utils import load_pretrained_model

# Visualization
from utils.tools import visualize_a_batch, visualize_a_batch_w_sen

import wandb

def train(config: dict):
    train_logger = Logger(logdir=os.path.join(config["OUTPUTS_DIR"], "train"), only_main=True)
    train_logger.show(head="Configs:", log=config)
    train_logger.write(log=config, filename="config.yaml", mode="w")
    train_logger.tb_add_git_version(git_version=config["GIT_VERSION"])

    set_seed(config["SEED"])

    # model = build_model(config=config)
    # Build and load pretrained grounding DINO
    model = build_gdino_model(config=config)

    # magic appears plsss
    torch.autograd.set_detect_anomaly(True, check_nan=True)

    # Load Pretrained Model
    # if config["PRETRAINED_MODEL"] is not None:
        # model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)

    # Data process
    dataset_train = build_dataset(config=config, split="train")
    sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
    # dataloader_train = build_dataloader(dataset=dataset_train, sampler=sampler_train,
    #                                     batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])
    # dataloader_train = build_dataloader_w_sen(dataset=dataset_train, sampler=sampler_train,
    #                                           batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])

    dataloader_train = build_dataloader_w_cat_cap(
        dataset=dataset_train, sampler=sampler_train,
        batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"]
    )

    # Criterion
    criterion = build_criterion(config=config)
    criterion.set_device(torch.device("cuda", distributed_rank()))

    wandb.init(
        # Set the project where this run will be logged
        project="thien", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"baseline_w_gdino_", 
        # # Track hyperparameters and run metadata
        # config={
        #     "architecture": "Transformer",
        #     "epochs": 500,
        # }
    )

    # Optimizer
    param_groups, lr_names = get_param_groups(config=config, model=model)
    optimizer = AdamW(params=param_groups, lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    # Scheduler
    if config["LR_SCHEDULER"] == "MultiStep":
        scheduler = MultiStepLR(
            optimizer,
            milestones=config["LR_DROP_MILESTONES"],
            gamma=config["LR_DROP_RATE"]
        )
    elif config["LR_SCHEDULER"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config["EPOCHS"]
        )
    else:
        raise ValueError(f"Do not support lr scheduler '{config['LR_SCHEDULER']}'")

    # Training states
    train_states = {
        "start_epoch": 0,
        "global_iters": 0
    }

    # Resume
    if config["RESUME"] is not None:
        if config["RESUME_SCHEDULER"]:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states,
                            optimizer=optimizer, scheduler=scheduler)
        else:
            load_checkpoint(model=model, path=config["RESUME"], states=train_states)
            for _ in range(train_states["start_epoch"]):
                scheduler.step()

    # Set start epoch
    start_epoch = train_states["start_epoch"]

    if is_distributed():
        model = DDP(module=model, device_ids=[distributed_rank()], find_unused_parameters=False)

    multi_checkpoint = "MULTI_CHECKPOINT" in config and config["MULTI_CHECKPOINT"]

    # Training:
    for epoch in range(start_epoch, config["EPOCHS"]):
        if is_distributed():
            sampler_train.set_epoch(epoch)
        dataset_train.set_epoch(epoch)

        if epoch >= config["ONLY_TRAIN_QUERY_UPDATER_AFTER"]:
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[3]["lr"] = 0.0
        lrs = [optimizer.param_groups[_]["lr"] for _ in range(len(optimizer.param_groups))]
        assert len(lrs) == len(lr_names)
        lr_info = [{name: lr} for name, lr in zip(lr_names, lrs)]
        train_logger.show(head=f"[Epoch {epoch}] lr={lr_info}")
        train_logger.write(head=f"[Epoch {epoch}] lr={lr_info}")
        default_lr_idx = -1
        for _ in range(len(lr_names)):
            if lr_names[_] == "lr":
                default_lr_idx = _
        train_logger.tb_add_scalar(tag="lr", scalar_value=lrs[default_lr_idx], global_step=epoch, mode="epochs")

        no_grad_frames = None
        if "NO_GRAD_FRAMES" in config:
            for i in range(len(config["NO_GRAD_STEPS"])):
                if epoch >= config["NO_GRAD_STEPS"][i]:
                    no_grad_frames = config["NO_GRAD_FRAMES"][i]
                    break

        train_one_epoch(
            model=model,
            train_states=train_states,
            max_norm=config["CLIP_MAX_NORM"],
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            # metric_log=train_metric_log,
            logger=train_logger,
            accumulation_steps=config["ACCUMULATION_STEPS"],
            use_dab=config["USE_DAB"],
            multi_checkpoint=multi_checkpoint,
            no_grad_frames=no_grad_frames
        )
        scheduler.step()
        train_states["start_epoch"] += 1
        if multi_checkpoint is True:
            pass
        else:
            if config["EPOCHS"] < 100 or (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model=model,
                    path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler
                )

    return


from utils.box_ops import box_cxcywh_to_xyxy
import torchvision.transforms as T


def get_local_images_in_batch(batch: dict, device) -> List[torch.Tensor]:
    assert 'infos' in batch
    
    def transform_crop_img(img: torch.Tensor, input_size: Tuple[int, int] = (224, 224)):
        # including resize and normalize
        transforms = T.Compose([
            T.Resize(input_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet mean and std
        ])
        return transforms(img)
    

    local_images: list[torch.Tensor] = []
    for b in range(len(batch['infos'])):
        list_crop_imgs = []

        boxes = [batch['infos'][b][i]['boxes'] for i in range(len(batch['infos'][b]))] # (num_frame_idx, num_boxes, 4)
        hs, ws, int_boxes = [], [], []
        for i in range(len(boxes)):
            # TODO: need to review
            _, h, w = batch['infos'][b][i]['unnorm_img'].shape
            int_boxes.append((box_cxcywh_to_xyxy(boxes[i]) * torch.as_tensor([w, h, w, h])).int().tolist())
            hs.append(h)
            ws.append(w)
        
        for i in range(len(boxes)):
            _box_int = int_boxes[i]
            _crop_img = [batch['infos'][b][i]['unnorm_img'][:, _box_int[j][1]:_box_int[j][3], _box_int[j][0]:_box_int[j][2]] \
                         for j in range(len(_box_int))] # (num_boxes, 3, h, w)
            _crop_img = torch.stack([transform_crop_img(im) for im in _crop_img], dim=0) # (num_boxes, 3, 224, 224)
            list_crop_imgs.append(_crop_img) 
            
        # list_crop_imgs = torch.stack(list_crop_imgs, dim=0) # (num_frame_idx, num_boxes, 3, 224, 224)
        local_images.append(list_crop_imgs) # (bs, num_frame_idx, num_boxes, 3, 224, 224)

    return local_images


def train_one_epoch(model: Tracknet, train_states: dict, max_norm: float,
                    dataloader: DataLoader, criterion: ClipCriterion, optimizer: torch.optim,
                    epoch: int, logger: Logger,
                    accumulation_steps: int = 1, use_dab: bool = False,
                    multi_checkpoint: bool = False,
                    no_grad_frames: int | None = None):
    """
    Args:
        model: Model.
        train_states:
        max_norm: clip max norm.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Training optimizer.
        epoch: Current epoch.
        # metric_log: Metric Log.
        logger: unified logger.
        accumulation_steps:
        use_dab:
        multi_checkpoint:
        no_grad_frames:

    Returns:
        Logs
    """
    model.train()
    optimizer.zero_grad()
    device = next(get_model(model).parameters()).device

    dataloader_len = len(dataloader)
    metric_log = MetricLog()
    epoch_start_timestamp = time.time()

    for i, batch in enumerate(dataloader):
        category_caption = batch['cat_caption']
        cat_list = batch['cat_list']
        iter_start_timestamp = time.time()
        tracks = TrackInstances.init_tracks(batch=batch,
                                            hidden_dim=get_model(model).hidden_dim,
                                            num_classes=get_model(model).max_text_len, # (old) num_classes
                                            device=device, use_dab=use_dab)
        criterion.init_a_clip(batch=batch,
                              hidden_dim=get_model(model).hidden_dim,
                              num_classes=get_model(model).max_text_len, # (old) num_classes
                              device=device)

        for frame_idx in range(len(batch["imgs"][0])):
            # sentences = batch["infos"][0][frame_idx]["sentences"]
            if no_grad_frames is None or frame_idx >= no_grad_frames:
                frame = [fs[frame_idx] for fs in batch["imgs"]]
                for f in frame:
                    f.requires_grad_(False)
                frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)

                # (new)
                # local_images_input = [local_images[b][frame_idx] for b in range(len(local_images))]
                local_images_input = None

                res = model(
                    samples=frame, 
                    tracks=tracks, 
                    captions=category_caption,
                    local_images=local_images_input, # (new) (num_boxes, 3, 224, 224)
                )
                previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                    model_outputs=res,
                    tracked_instances=tracks,
                    frame_idx=frame_idx,
                    cat_list=cat_list,
                    caption=category_caption,
                )
                if frame_idx < len(batch["imgs"][0]) - 1:
                    tracks = get_model(model).postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets)
                    
            else:
                with torch.no_grad():
                    frame = [fs[frame_idx] for fs in batch["imgs"]]
                    for f in frame:
                        f.requires_grad_(False)
                    frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                    res = model(frame=frame, tracks=tracks, sentences=sentence)
                    previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                        model_outputs=res,
                        tracked_instances=tracks,
                        frame_idx=frame_idx
                    )
                    if frame_idx < len(batch["imgs"][0]) - 1:
                        tracks = get_model(model).postprocess_single_frame(
                            previous_tracks, new_tracks, unmatched_dets, \
                                no_augment=frame_idx < no_grad_frames-1)

        loss_dict, log_dict = criterion.get_mean_by_n_gts()
        loss = criterion.get_sum_loss_dict(loss_dict=loss_dict)

        # Metrics log
        metric_log.update(name="total_loss", value=loss.item())
        wandb.log({f"loss_epoch_{epoch+1}": loss.item()})
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            else:
                pass
            optimizer.step()
            optimizer.zero_grad()

        # For logging
        for log_k in log_dict:
            metric_log.update(name=log_k, value=log_dict[log_k][0])

        iter_end_timestamp = time.time()
        metric_log.update(name="time per iter", value=iter_end_timestamp-iter_start_timestamp)
        # Outputs logs
        if True or i<=5 or (i+1) % 20 == 0:
            metric_log.sync()
            max_memory = max([torch.cuda.max_memory_allocated(torch.device('cuda', i))
                              for i in range(distributed_world_size())]) // (1024**2)
            second_per_iter = metric_log.metrics["time per iter"].avg
            logger.show(head=f"[Epoch={epoch}, Iter={i}, "
                             f"{second_per_iter:.2f}s/iter, "
                             f"{i}/{dataloader_len} iters, "
                             f"rest time: {int(second_per_iter * (dataloader_len - i) // 60)} min, "
                             f"Max Memory={max_memory}MB]",
                        log=metric_log)
            logger.write(head=f"[Epoch={epoch}, Iter={i}/{dataloader_len}]",
                         log=metric_log, filename="log.txt", mode="a")
            logger.tb_add_metric_log(log=metric_log, steps=train_states["global_iters"], mode="iters")

        if multi_checkpoint:
            if i % 100 == 0 and is_main_process():
                logger.show('Saving checkpoint for iteration {}'.format(i))
                save_checkpoint(
                    model=model,
                    path=os.path.join(logger.logdir[:-5], f"checkpoint_{int(i // 100)}.pth")
                )

        train_states["global_iters"] += 1

    # Epoch end
    metric_log.sync()
    epoch_end_timestamp = time.time()
    epoch_minutes = int((epoch_end_timestamp - epoch_start_timestamp) // 60)
    logger.show(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                log=metric_log)
    logger.write(head=f"[Epoch: {epoch}, Total Time: {epoch_minutes}min]",
                 log=metric_log, filename="log.txt", mode="a")
    logger.tb_add_metric_log(log=metric_log, steps=epoch, mode="epochs")

    return


def get_param_groups(config: dict, model: nn.Module) -> Tuple[List[Dict], List[str]]:
    """
    用于针对不同部分的参数使用不同的 lr 等设置
    Args:
        config: 实验的配置信息
        model: 需要训练的模型

    Returns:
        params_group: a list of params groups.
        lr_names: a list of params groups' lr name, like "lr_backbone".
    """
    def match_keywords(name: str, keywords: List[str]):
        matched = False
        for keyword in keywords:
            if keyword in name:
                matched = True
                break
        return matched
    # keywords
    backbone_keywords = ["backbone"] # (old) backbone.backbone
    points_keywords = ["reference_points", "sampling_offsets"]  # 在 transformer 中用于选取参考点和采样点的网络参数关键字
    query_updater_keywords = ["query_updater"]
    bert_layer_keywords = ["bert"]
    param_groups = [
        {   # backbone 学习率设置
            "params": [p for n, p in model.named_parameters() if match_keywords(n, backbone_keywords) and p.requires_grad],
            "lr": config["LR_BACKBONE"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, points_keywords)
                       and p.requires_grad],
            "lr": config["LR_POINTS"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, query_updater_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if not match_keywords(n, backbone_keywords)
                       and not match_keywords(n, points_keywords)
                       and not match_keywords(n, query_updater_keywords)
                       and not match_keywords(n, bert_layer_keywords)
                       and p.requires_grad],
            "lr": config["LR"]
        },
        {
            "params": [p for n, p in model.named_parameters() if match_keywords(n, bert_layer_keywords)
                       and p.requires_grad],
            "lr": 1e-7
        }, # (new) put at this position to avoid being #0, #1, #3 in the training scheme.
    ]
    return param_groups, ["lr_backbone", "lr_points", "lr_query_updater", "lr", "lr_bert_layer"] # (new) add "lr_bert_layer"
