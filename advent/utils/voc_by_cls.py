import os
from PIL import Image
import numpy as np
from tqdm import tqdm

voc_root = '/disk1/datasets/VOCdevkit/VOC2012'
voc_lst = os.path.join(voc_root, 'list/train_aug.txt')
# lstbylb_dir = os.path.join(voc_lst, 'SegmentationClassAug')
new_lst = os.path.join(voc_root, 'list/train_aug_cls.txt')

f = open(voc_lst, 'r')
lines = f.readlines()
f.close()

cls_list = {}
for l in tqdm(lines):
    label_path = os.path.join(voc_root, l.split(' ')[-1].strip()[1:])
    label = np.asarray(Image.open(label_path), dtype=np.int32)
    cur_clses = list(set(list(label.flatten())))
    for cls_idx in cur_clses:
        if cls_idx == 0 or cls_idx == 255:
            continue
        else:
            if cls_idx not in cls_list:
                cls_list[cls_idx] = [l.strip() + ' ' + str(cls_idx)]
            else:
                cls_list[cls_idx].append(l.strip() + ' ' + str(cls_idx))

rst_lst = []
for key in cls_list.keys():
    rst_lst += cls_list[key]

with open(new_lst, 'w') as f:
    for l in rst_lst:
        f.write(l + '\n')
