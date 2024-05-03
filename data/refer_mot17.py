# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import json
import data.transforms as T
from structures.track_instances import TrackInstances


class MOT17Dataset:
    def __init__(self, args, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.num_frames_per_batch = max(args['SAMPLER_LENGTHS'])
        self.sample_mode = args['SAMPLE_MODE']
        self.sample_interval = args['SAMPLE_INTERVAL']
        self.vis = args['VIS']

        with open('mot17_train_coco.json', 'r') as f:
            self.mot17_data_annotations = json.load(f)
            img_files = self.mot17_data_annotations['images']
            self.img_files = ["{}/{}".format(
                seqs_folder,
                img_file['file_name'][5:]).replace('mot17_train_coco','train').split('_')[0] + '/img1/' + img_file['file_name'].split('_')[-1]
                
                for img_file in img_files
            ] # remove 'data/' from the file name

            # self.img_files = [
            #     os.path.join(seqs_folder,
            #                  img_file['file_name'][5:].split('/mot17_train_coco/')[0],
            #                  "images",
            #                  "train",
            #                  img_file['file_name'][5:].split('/mot17_train_coco/')[1].split('')[0], 
            #                  'img1', 
            #                  img_file['file_name'].split('')[-1])
            #     # "{}/{}".format(
            #     # seqs_folder,
            #     # img_file['file_name'][5:].replace('mot17_train_coco',os.path.join("images","train")).split('')[0] + '/img1/' + img_file['file_name'].split('')[-1])

            #     for img_file in img_files
            # ] # remove 'data/' from the file name
            self.img_files = [{**img_file, "file_name":_file_name} for img_file,_file_name in zip(img_files,self.img_files)]

        # The number of images per sample: 1 + (num_frames - 1) * interval.
        # The number of valid samples: num_images - num_image_per_sample + 1.
        self.item_num = len(self.img_files) - (self.num_frames_per_batch - 1) * self.sample_interval


        # video sampler.
        self.sampler_steps: list = args['SAMPLER_STEPS']
        self.sampler_lengths: list = args['SAMPLER_LENGTHS']
        self.sampler_intervals: list = args['SAMPLER_INTERVALS']
        self.sampler_modes: list = args['SAMPLER_MODES']

        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.sampler_lengths))
        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.sampler_lengths) > 0
            assert len(self.sampler_lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.item_num = len(self.img_files) - (self.sampler_lengths[-1] - 1) * self.sample_interval
            self.period_idx = 0
            self.num_frames_per_batch = self.sampler_lengths[0]
            self.current_epoch = 0

    def set_epoch(self, epoch: int):
        # Copy from dancetrack.py
        self.sample_begin_frame_paths = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sampler_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sampler_steps) + 1
        self.sample_length = self.sampler_lengths[min(len(self.sampler_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sampler_modes[min(len(self.sampler_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sampler_intervals[min(len(self.sampler_intervals) - 1, self.sample_stage)]
        # End of Copy
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.sampler_lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        gt_instances.is_ref = targets['is_ref']
        return gt_instances

    def _pre_single_frame(self, idx: int, annotations: list):

        img_path = self.img_files[idx]['file_name']

        img_id = self.img_files[idx]['id']
        _ann = [ann for ann in annotations if img_id == ann['image_id']]

        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)

        targets['dataset'] = 'MOT17'

        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(img_id)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        targets['is_ref'] = []

        for _a in _ann:
            targets['boxes'].append([
                _a['bbox'][0], 
                _a['bbox'][1], 
                _a['bbox'][0]+_a['bbox'][2],
                _a['bbox'][1]+_a['bbox'][3]]) # xywh -> xyxy           
            targets['area'].append(_a['area'])
            targets['iscrowd'].append(_a['iscrowd'])
            targets['labels'].append(_a['category_id'])
            targets['obj_ids'].append(_a['id'])  # relative id
            targets['is_ref'].append(1.0)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'], dtype=torch.int64)
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['is_ref'] = torch.as_tensor(targets['is_ref'])
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, start, end, interval=1, annotations=None):
        targets = []
        images = []
        for i in range(start, end, interval):
            img_i, targets_i = self._pre_single_frame(i, annotations)
            images.append(img_i)
            targets.append(targets_i)
        return images, targets

    def __getitem__(self, idx):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)

        annotations = self.mot17_data_annotations['annotations']
        images, targets = self.pre_continuous_frames(sample_start, sample_end, sample_interval, annotations)

        sentence = ["a photo of a person"]

        data = {}
        dataset_name = targets[0]['dataset']
        transform = self.dataset2transform[dataset_name]
        if transform is not None:
            images, targets = transform(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
        data.update({
            'imgs': images,
            'sentences': sentence, # change here if we need captions
            'gt_instances': gt_instances,
            'dataset_name': dataset_name,
        })
        if self.args['VIS']:
            data['ori_img'] = [target_i['ori_img'] for target_i in targets]
        return data

    def __len__(self):
        # return self.item_num
        assert self.sample_begin_frame_paths is not None, "Please use set_epoch to init MOT17 Dataset."
        return len(self.sample_begin_frame_paths)



class DetMOTDetectionValidation(MOT17Dataset):
    def __init__(self, args, seqs_folder, dataset2transform):
        args['DATA_TXT_PATH'] = args['VAL_DATA_TXT_PATH'].val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)


def make_transforms_for_mot17(image_set, 
                              args=None, 
                              coco_size: bool=False,
                              overflow_bbox: bool = False,
                              reverse_clip: bool = False):

    normalize = T.MultiCompose([
        T.MultiToTensor(),
        T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from COCO/MOTR
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MultiCompose([
            T.MultiRandomHorizontalFlip(),
            T.MultiRandomSelect(
                T.MultiRandomResize(scales, max_size=1536),
                T.MultiCompose([
                    T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                    T.MultiRandomCrop(
                        min_size=384 if coco_size else 800,
                        max_size=600 if coco_size else 1200,
                        overflow_bbox=overflow_bbox),
                    T.MultiRandomResize(scales, max_size=1536)
                ])
            ),
            T.MultiHSV(),
            normalize,
            T.MultiReverseClip(reverse=reverse_clip)
        ])

    if image_set == 'val':
        return T.MultiCompose([
            T.MultiRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    dataset2transform_train = {'MOT17': mot17_train}
    if image_set == 'train':
        return dataset2transform_train
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args['DATA_ROOT'])
    assert root.exists(), f'provided DATA_ROOT={root} does not exist'
    dataset2transform = build_dataset2transform(args, 
                                                image_set,
                                                coco_size=args["COCO_SIZE"],
                                                overflow_bbox=args["OVERFLOW_BBOX"],
                                                reverse_clip=args["REVERSE_CLIP"])
    if image_set == 'train':
        return MOT17Dataset(args, 
                               seqs_folder=root, 
                               dataset2transform=dataset2transform)
    else:
        raise NotImplementedError()

