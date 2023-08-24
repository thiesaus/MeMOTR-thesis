# MeMOTR

The official implementation of [MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking](https://arxiv.org/abs/2307.15700), ICCV 2023.

Authors: [Ruopeng Gao](https://ruopenggao.com), [Limin Wang](https://wanglimin.github.io/).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memotr-long-term-memory-augmented-transformer/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=memotr-long-term-memory-augmented-transformer)

![MeMOTR](./assets/overview.png)

**MeMOTR** is a fully-end-to-end memory-augmented multi-object tracker based on Transformer. We leverage long-term memory injection with a customized memory-attention layer, thus significantly improving the association performance.



## News :fire:

- 2023.8.9: We release the main code. More configurations, scripts and checkpoints will be released soon :soon:.



## Installation

```shell
conda create -n MeMOTR python=3.10
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python
```

You also need to compile the Deformable Attention CUDA ops:

```shell
# From https://github.com/fundamentalvision/Deformable-DETR
cd ./models/ops/
sh make.sh
# You can test this ops if you need:
python test.py
```

## Pretrain

We initialize our model with the official DAB-Deformable-DETR ( with R50 backbone) weights pretrained on the COCO dataset, you can also download the checkpoint we used [here](https://drive.google.com/file/d/17FxIGgIZJih8LWkGdlIOe9ZpVZ9IRxSj/view?usp=sharing). And then put the checkpoint at the root of this project dir.

## Scripts on DanceTrack

### Training
Train MeMOTR with 8 GPUs on DanceTrack (recommended to use GPUs with >= 32 GB Memory, like V100-32GB or some else):
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --use-distributed --config-path ./configs/train_dancetrack.yaml --outputs-dir ./outputs/memotr_dancetrack/ --batch-size 1 --data-root <your data dir path>
```
if your GPU's memory is below than 32 GB, we also implement a memory-optimized version (by running option `--use-checkpoint`) as discussed in the paper, we use [gradient checkpoint](https://pytorch.org/docs/1.13/checkpoint.html?highlight=checkpoint#torch.utils.checkpoint.checkpoint) to reduce the allocated GPU memory. This following training script will only take about 10 GB GPU memory:
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --use-distributed --config-path ./configs/train_dancetrack.yaml --outputs-dir ./outputs/memotr_dancetrack/ --batch-size 1 --data-root <your data dir path> --use-checkpoint
```

### Submit and Evaluation
You can use this script to evaluate the trained model on the DanceTrack val set:
```shell
python main.py --mode eval --data-root <your data dir path> --eval-mode specific --eval-model <filename of the checkpoint> --eval-dir ./outputs/memotr_dancetrack/ --eval-threads 8
```
for submitting, you can use the following scripts:
```shell
python -m torch.distributed.run --nproc_per_node=8 main.py --mode submit --submit-dir ./outputs/memotr_dancetrack/ --submit-model <filename of the checkpoint> --use-distributed --data-root <your data dir path>
```
Besides, if you just want to directly eval or submit through our trained checkpoint, you can get the checkpoint we used in the paper [here](https://drive.google.com/file/d/1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2/view?usp=sharing). Then put this checkpoint into [./outputs/memotr_dancetrack/](./outputs/memotr_dancetrack/) and run the above scripts.

## Results

### Multi-Object Tracking on the DanceTrack test set

| Methods                  | HOTA | DetA | AssA | checkpoint                                                                                         |
| ------------------------ | ---- | ---- | ---- |----------------------------------------------------------------------------------------------------|
| MeMOTR                   | 68.5 | 80.5 | 58.4 | [Google Drive](https://drive.google.com/file/d/1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2/view?usp=sharing) |
| MeMOTR (Deformable DETR) | 63.4 | 77.0 | 52.3 | Coming Soon...                                                                                     |



### Multi-Object Tracking on the MOT17 test set

| Methods | HOTA | DetA | AssA | checkpoint     |
| ------- | ---- | ---- | ---- | -------------- |
| MeMOTR  | 58.8 | 59.6 | 58.4 | Coming Soon... |



### Multi-Category Multi-Object Tracking on the BDD100K val set

| Methods | mTETA | mLocA | mAssocA | checkpoint     |
| ------- | ----- | ----- | ------- | -------------- |
| MeMOTR  | 53.6  | 38.1  | 56.7    | Coming Soon... |



## Contact

- Ruopeng Gao: ruopenggao@gmail.com



## Acknowledgement

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [DAB DETR](https://github.com/IDEA-Research/DAB-DETR)
- [MOTR](https://github.com/megvii-research/MOTR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [CV-Framework](https://github.com/HELLORPG/CV-Framework)
