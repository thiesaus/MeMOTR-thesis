# @Author       : Ruopeng Gao
# @Date         : 2022/12/2
import math
import os
import torch
import random

from collections import defaultdict
from random import randint
from PIL import Image

import data.transforms as T
from .mot import MOTDataset
import json

def dd():
    return defaultdict(list)
def ddd():
    return defaultdict(dd)




class MOT17_COCO(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        super(MOT17_COCO, self).__init__(config=config, split=split, transform=transform)
        self.spliter = "/"
        if os.name == "nt":
            self.spliter="\\"
        self.config = config
        self.transform = transform
        self.use_motsynth = config["USE_MOTSYNTH"]
        self.use_crowdhuman = config["USE_CROWDHUMAN"]
        self.motsynth_rate = config["MOTSYNTH_RATE"]
        with open(config["TRAIN_COCO"], 'r') as f:
            self.mot17_data_annotations = json.load(f)
            img_files = self.mot17_data_annotations['images']
            self.img_files = [
                os.path.join(config["DATA_ROOT"],img_file['file_name'][5:].split('/mot17_train_coco/')[0],"images","train",img_file['file_name'][5:].split('/mot17_train_coco/')[1].split('_')[0], 'img1', img_file['file_name'].split('_')[-1])
                # "{}/{}".format(
                # seqs_folder,
                # img_file['file_name'][5:].replace('mot17_train_coco',os.path.join("images","train")).split('_')[0] + '/img1/' + img_file['file_name'].split('_')[-1])

                for img_file in img_files
            ] # remove 'data/' from the file name
            self.img_files = [{**img_file, "file_name":_file_name} for img_file,_file_name in zip(img_files,self.img_files)]

        if self.use_motsynth:
            multi_random_state = random.getstate()
            random.seed(config["SEED"])
            self.unified_random_state = random.getstate()
            random.setstate(multi_random_state)
        else:
            self.unified_random_state = None

        assert split == "train", f"Split {split} is NOT supported."
        self.mot17_seqs_dir = os.path.join(config["DATA_ROOT"], config["DATASET"].split("_")[0], "images", split)
        self.mot17_gts_dir = os.path.join(config["DATA_ROOT"], config["DATASET"].split("_")[0], "gts", split)
        self.crowdhuman_seq_dir = os.path.join(config["DATA_ROOT"], "CrowdHuman", "images", "val")
        self.crowdhuman_gts_dir = os.path.join(config["DATA_ROOT"], "CrowdHuman", "gts", "val")
        # Training MOT17, using MOT17 train split and crowdhuman val splits
        self.motsynth_seqs_dir = os.path.join(config["DATA_ROOT"], "MOTSynth", "frames")
        self.motsynth_gts_dir = os.path.join(config["DATA_ROOT"], "MOTSynth", "gts")

        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.sample_mot17_join: int = config["SAMPLE_MOT17_JOIN"]
        self.sample_stage = None
        self.sample_begin_frame_paths = None
        self.sample_sentences = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.mot17_gts = defaultdict(ddd)
        # self.mot17_gts_coco=defaultdict(ddd)
        self.crowdhuman_gts = defaultdict(list)
        self.motsynth_gts = defaultdict(dd)

        # self.mot17_seq_names = [seq for seq in os.listdir(self.mot17_seqs_dir) if "SDP" in seq]
        self.mot17_seq_names = [seq for seq in list(self.mot17_data_annotations["text_key"].keys()) ]
        # for vid in self.mot17_seq_names:
        #     mot17_gts_dir = os.path.join(self.mot17_gts_dir, vid,"img1")
        #     mot17_gt_paths = [os.path.join(mot17_gts_dir, filename) for filename in os.listdir(mot17_gts_dir)]
        #     for mot17_gt_path in mot17_gt_paths:
        #         for line in open(mot17_gt_path,"r"):
        #             _, i, x, y, w, h, v = line.strip("\n").split(" ")
        #             i, x, y, w, h, v = map(float, (i, x, y, w, h, v))
        #             i, x, y, w, h = map(int, (i, x, y, w, h))
        #             t = int(mot17_gt_path.split(self.spliter)[-1].split(".")[0])
        #             self.mot17_gts[vid][t].append([i, x, y, w, h])
        text_key = self.mot17_data_annotations['text_key']
        for vid in self.mot17_seq_names:
            for t in text_key[vid]:
                num=int(t)+1
                sentences= list(text_key[vid][t].keys())
                for sentence in sentences:
                    
                    if sentence == 'null': # sentence is null when there is lack of either appearance or action caption for the person
                        continue

                    for line in text_key[vid][t][sentence]:
                        i= line["track_id"]
                        x, y, w, h = line["bbox"]
                        i, x, y, w, h = map(int, (i, x, y, w, h))
                        self.mot17_gts[vid][num][sentence].append([i, x, y, w, h])

        
        # Prepare for MOTSynth
        if self.use_motsynth:
            self.motsynth_seq_names = [seq for seq in os.listdir(self.motsynth_seqs_dir)]
            for vid in self.motsynth_seq_names:
                motsynth_gt_path = os.path.join(self.motsynth_gts_dir, vid, "gt", "gt.txt")
                for line in open(motsynth_gt_path,"r"):
                    t, i, *xywh, a, b, c = line.strip().split(",")[:9]
                    if int(a) == 0 or not int(b) == 1 or float(c) == 0:
                        continue
                    x, y, w, h = map(float, xywh)
                    self.motsynth_gts[vid][int(t)].append([int(i), x, y, w, h])
        # crowdhuman_gt_filenames = os.listdir(self.crowdhuman_gts_dir)
        # for filename in crowdhuman_gt_filenames:
        #     crowdhuman_gt_path = os.path.join(self.crowdhuman_gts_dir, filename)
        #     image_name = filename.split(".")[0]
        #     for line in open(crowdhuman_gt_path):
        #         _, i, x, y, w, h = line.strip("\n").split(" ")
        #         i, x, y, w, h = map(int, (i, x, y, w, h))
        #         self.crowdhuman_gts[image_name].append([i, x, y, w, h])

        self.set_epoch(epoch=0)     # init datasets
     
        return

    def __len__(self):
        assert self.sample_begin_frame_paths is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frame_paths )

    def __getitem__(self, item):
        # begin_sentence = self.sample_begin_frame_paths[item]["sentence"]
        # begin_frame_path = self.sample_begin_frame_paths[item]["path"]
        begin_frame_path = self.sample_begin_frame_paths[item]
        frame_paths = self.sample_frame_paths(begin_frame_path=begin_frame_path)
        imgs, infos = self.get_multi_frames(frame_paths=frame_paths,sentence=None)

        if infos[0]["dataset"] == "MOT17":
            imgs, infos = self.transform["MOT17"](imgs, infos)
        else:
            imgs, infos = self.transform["CrowdHuman"](imgs, infos)

        return {
            "imgs": imgs,
            "infos": infos
        }

    def set_epoch(self, epoch: int):
        # Copy from dancetrack.py
        self.sample_begin_frame_paths = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        # End of Copy
        # Add Crowdhuman:
        if self.use_crowdhuman:
            for image_name in self.crowdhuman_gts:
                self.sample_begin_frame_paths.append(os.path.join(self.crowdhuman_seq_dir, f"{image_name}.jpg"))
        if epoch >= self.sample_mot17_join:
          
            for vid in self.mot17_gts.keys():
                t_min = min(map(int,self.mot17_gts[vid].keys()))
                t_max = max(map(int,self.mot17_gts[vid].keys()))
                self.sample_vid_tmax[vid] = t_max
                for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                    # for sen in list(self.mot17_gts[vid][t].keys()):
                    #     sample=defaultdict()
                    #     sample["path"] = os.path.join(self.mot17_seqs_dir, vid, "img1", str(t).zfill(6) + ".jpg")
                    #     sample["sentence"] = sen
                    #     self.sample_begin_frame_paths.append(
                    #        sample
                    #     )
                    self.sample_begin_frame_paths.append(
                        os.path.join(self.mot17_seqs_dir, vid, "img1", str(t).zfill(6) + ".jpg")
                    )
                
        if self.use_motsynth:
            multi_random_state = random.getstate()
            random.setstate(self.unified_random_state)
            for vid in self.motsynth_gts.keys():
                t_min = min(self.motsynth_gts[vid].keys())
                t_max = max(self.motsynth_gts[vid].keys())
                self.sample_vid_tmax[vid] = t_max
                for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                    if random.random() > self.motsynth_rate:
                        continue
                    self.sample_begin_frame_paths.append(
                        os.path.join(self.motsynth_seqs_dir, vid, "rgb", str(t).zfill(4) + ".jpg")
                    )
            self.unified_random_state = random.getstate()
            random.setstate(multi_random_state)

        return

    def sample_frame_paths(self, begin_frame_path: str) -> list[str]:
        if "CrowdHuman" in begin_frame_path:
            return [begin_frame_path] * self.sample_length
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample Length is less than 2."
            vid = begin_frame_path.split(self.spliter)[-3]
            begin_t = int(begin_frame_path.split(self.spliter)[-1].split(".")[0])
            remain_frames = self.sample_vid_tmax[vid] - begin_t
            max_interval = math.floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idx = [begin_t + interval * i for i in range(self.sample_length)]
            if "MOTSynth" in begin_frame_path:
                frame_paths = [os.path.join(self.motsynth_seqs_dir, vid, "rgb", str(t).zfill(4) + ".jpg") for t in frame_idx]
            else:
                frame_paths = [os.path.join(self.mot17_seqs_dir, vid, "img1", str(t).zfill(6) + ".jpg") for t in frame_idx]

            return frame_paths
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")

    def get_single_frame(self, frame_path: str,sentence:str=None):
        if "CrowdHuman" in frame_path:
            frame_name = frame_path.split(self.spliter)[-1].split(".")[0]
            gt = self.crowdhuman_gts[frame_name]
        elif "MOT17" in frame_path or "MOTSynth" in frame_path:
            frame_idx = int(frame_path.split(self.spliter)[-1].split(".")[0])
            vid = frame_path.split(self.spliter)[-3]
            if "MOTSynth" in frame_path:
                gt = self.motsynth_gts[vid][frame_idx]
            else:
                # gt = self.mot17_gts[vid][frame_idx][sentence]
                gt = self.mot17_gts[vid][frame_idx]
        else:
            raise RuntimeError(f"Frame path '{frame_path}' has no GTs.")
        img = Image.open(frame_path)

        crowdhuman_ids_offset = 100000

        info = {}

        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["sentences"] = list()
        info["dataset"] = "MOT17" if ("MOT17" in frame_path or "MOTSynth" in frame_path) else "CrowdHuman"

        for sen in gt:
            for i, x, y, w, h in gt[sen]:
                info["boxes"].append(list(map(float, (x, y, w, h))))
                info["areas"].append(w * h)
                info["ids"].append(i if "MOT17" in frame_path else i + crowdhuman_ids_offset)
                info["labels"].append(0)
                info["sentences"].append(sen)

        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"], dtype=torch.long)
        info["labels"] = torch.as_tensor(info["labels"], dtype=torch.long)
        
        # xywh to xyxy
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0, ), dtype=torch.long)
            info["labels"] = torch.zeros((0, ), dtype=torch.long)

        info['frame_path'] = frame_path # for debugging

        return img, info

    def get_multi_frames(self, frame_paths: list[str],sentence:str=None):
        return zip(*[self.get_single_frame(frame_path=path,sentence=sentence) for path in frame_paths])


def transforms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    return {
        "MOT17": T.MultiCompose([
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
            T.MultiCompose([
                T.MultiToTensor(),
                T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from COCO/MOTR
            ]),
            T.MultiReverseClip(reverse=reverse_clip)
        ]),
        "CrowdHuman": T.MultiCompose([
            T.MultiRandomHorizontalFlip(),
            T.MultiRandomShift(),
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
            T.MultiCompose([
                T.MultiToTensor(),
                T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from COCO/MOTR
            ]),
            T.MultiReverseClip(reverse=reverse_clip)
        ])
    }


def build(config: dict, split: str):
    if split == "train":
        return MOT17_COCO(
            config=config,
            split=split,
            transform=transforms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    else:
        raise NotImplementedError(f"MOT Dataset 'build' function do not support split {split}.")
