from advent.utils.viz_segmask import colorize_mask
from entropy import colorize_save_voc
import os
from PIL import Image
from advent.utils import pallete
import shutil
import numpy as np

data_id = 1
root_dir = "/disk1/datasets/personal/id{}".format(data_id)
val_show = os.path.join(root_dir, 'val_color_all')
val_list = os.path.join(root_dir, 'val_all.txt')
palette = pallete.get_voc_pallete(21)

vals = open(val_list, 'r').readlines()
for l in vals:
    im_dir = os.path.join(root_dir, l.split(' ')[0][1:])
    seg_dir = os.path.join(root_dir, l.split(' ')[1][1:])
    label = np.asarray(Image.open(seg_dir), dtype=np.int32)
    colorize_save_voc(label, l.split(' ')[0].split('/')[-1].split('.')[0],
                        palette, val_show)
    shutil.copyfile(im_dir, os.path.join(val_show, l.split(' ')[0].split('/')[-1]))