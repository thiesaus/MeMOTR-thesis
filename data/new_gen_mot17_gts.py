# @Author       : Ruopeng Gao
# @Date         : 2022/12/19
# @Reference    : https://github.com/ifzhang/FairMOT/blob/master/src/gen_labels_20.py

import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# You should change the path to your own path:
seq_root = "./dataset/MOT17/images/train"
label_root = "./dataset/MOT17/gts/train"
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        # x += w / 2    # maintain xywh format, same as DanceTrack.
        # y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:d} {:d} {:d} {:d} {:f}\n'.format(
            tid_curr, int(x), int(y), int(w), int(h), float(_))
        with open(label_fpath, 'a') as f:
            f.write(label_str)