#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_utils.py
@Time    :   2020/06/19 20:55:31
@Author  :   ZHANG Yu, Nankai University 
@Contact :   zhangyuygss@gmail.com
'''
import numpy as np
import random
import pickle as pkl
import os
from tqdm import tqdm
import shutil
from advent.utils.viz_segmask import colorize_mask
from PIL import Image
from advent.utils import pallete


def colorize_voc(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    palette[-3:] = [255, 255, 255]
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def colorize_save_voc(output, name, palette, save_root):
    os.makedirs(save_root, exist_ok=True)

    mask_Img         = Image.fromarray(np.asarray(output, dtype=np.uint8))
    mask_color = colorize_voc(output, palette)
    mask_Img.save(os.path.join(save_root, '%s.png' % (name)))
    mask_color.save(os.path.join(save_root, '%s_color.png' % (name)))


def get_ids_from_lst(lst_file):
    # print(lst_file)
    ids = set()
    try:
        with open(lst_file, 'r') as f:
            lines = f.readlines()
            for l in lines:
                im_name = l.split(' ')[0].split('/')[-1].split('.')[0]
                ids.add(im_name)
        return list(ids)
    except FileNotFoundError:
        return None


def get_info_from_txt(lst_f):
    lines = open(lst_f, 'r').readlines()
    cls_lst = {}
    for line in lines:
        line = line.strip()
        img, labels = line.split(':')
        img = img.split('/')[-1]
        lb_lst = labels.split(',')
        for lb in lb_lst:
            if lb not in cls_lst:
                cls_lst[lb] = [img]
            else:
                cls_lst[lb].append(img)
    for cls in cls_lst.keys():
        cls_lst[cls] = list(set(cls_lst[cls]))
    return cls_lst


def prepare_lst_by_class(root_dir, lst_f, val_rate=0.35):
    cls_lst = get_info_from_txt(lst_f)
    sorted_lst = sorted(cls_lst.items(), key=lambda kv:len(kv[1]), reverse=True)
    random.seed(1)
    info = {'cls_lst': sorted_lst}
    val_info = {}
    tr_lst, val_lst = [], []
    for item in sorted_lst:
        clas, lst = item
        if not clas in ['', 'background']:
            lst_cur = ['/JPEGImages/{x}.jpg /SegmentationClass/{x}.png {cls}'.format(x=name, cls=clas) for name in lst]
            tr_lst += lst_cur
            offset = len(lst) - int(len(lst) * val_rate)
            val_lst += lst_cur[offset:]
            val_info[clas] = lst_cur[offset:]
    info['val'] = val_info

    trlst_wo_val = tr_lst.copy()
    for l in val_lst:
        if l in trlst_wo_val:
            trlst_wo_val.remove(l)
    # write to pickle and list file
    with open(os.path.join(root_dir, 'info.pkl'), 'wb') as info_f:
        pkl.dump(info, info_f)
    with open(os.path.join(root_dir, 'train_sm.txt'), 'w') as tr_sm:
        for l in trlst_wo_val:
            tr_sm.write(l + '\n')
    with open(os.path.join(root_dir, 'train.txt'), 'w') as tr_f:
        for l in tr_lst:
            tr_f.write(l + '\n')
    with open(os.path.join(root_dir, 'val.txt'), 'w') as val_f:
        for l in val_lst:
            val_f.write(l + '\n')


def random_string(length):
    chars = 'abcdefghijklmnopqrstuvwxyz'
    return random.sample(chars, length)


def rename_imgs(root_dir):
    from os.path import join as pjoin
    imgdir = pjoin(root_dir, 'JPEGImages')
    sgclsdir, sgobdir = pjoin(root_dir, 'SegmentationClass'), pjoin(root_dir, 'SegmentationObject')
    list_file = pjoin(root_dir, 'imgLabelList.txt')
    lines = open(list_file, 'r').readlines()
    img2label = dict()
    for l in lines:
        im_name = l.split(':')[0].split('/')[-1]
        img2label[im_name] = l.split(':')[1]
    img_lst = os.listdir(imgdir)
    new_lines, name_map = [], dict()
    for img in tqdm(img_lst):
        im_name = img.split('.jpg')[0]
        new_name = ''.join(random_string(10))
        name_map[im_name] = new_name
        shutil.move(pjoin(imgdir, im_name + '.jpg'), pjoin(imgdir, new_name + '.jpg'))
        try:
            shutil.move(pjoin(sgclsdir, im_name + '.png'), pjoin(sgclsdir, new_name + '.png'))
            shutil.move(pjoin(sgobdir, im_name + '.png'), pjoin(sgobdir, new_name + '.png'))
        except FileNotFoundError: pass
        new_lines.append("{}.jpg:{}".format(new_name, img2label.get(im_name, 'UNKNOWN\n')))
    with open(pjoin(root_dir, 'imgLabelListnew.txt'), 'w') as f:
        for l in new_lines:
            f.write(l)
    pkl.dump(name_map, open(pjoin(root_dir, 'img_name_mapping.pkl'), 'wb'))
    
    # late processing for id1 and id2
    def _rename_imgs_in_list(list_name):
        new_lines = []
        with open(list_name, 'r') as f:
            for l in f.readlines():
                pre_name = l.split(' ')[0].split('/')[-1].split('.')[0]
                cluster_id = l.split(' ')[-1]
                new_name = name_map[pre_name]
                new_l = "/JPEGImages/{}.jpg /SegmentationClass/{}.png {}".format(new_name, new_name, cluster_id)
                new_lines.append(new_l)
        with open(list_name, 'w') as f:  # overwrite pre-list
            for l in new_lines: f.write(l)

    _rename_imgs_in_list(pjoin(root_dir, 'val_all.txt'))
    _rename_imgs_in_list(pjoin(root_dir, 'train_cluster80.txt'))


def colorize_data(root_dir, lst):
    palette = pallete.get_voc_pallete(21)
    val_show = os.path.join(root_dir, 'val_color_all')
    lst = [name.split('.png')[0] for name in lst]
    for l in tqdm(lst):
        im_dir = os.path.join(root_dir, 'JPEGImages', l + '.jpg')
        seg_dir = os.path.join(root_dir, 'SegmentationClass', l + '.png')
        label = np.asarray(Image.open(seg_dir), dtype=np.int32)
        img = np.asarray(Image.open(im_dir), dtype=np.int32)
        if img.shape[0] != label.shape[0]:
            print(im_dir)
        colorize_save_voc(label, l, palette, val_show)
        shutil.copyfile(im_dir, os.path.join(val_show, l + '.jpg'))


def get_val_list(root_dir, dst_file):
    lst = os.listdir(os.path.join(root_dir, 'SegmentationClass'))
    with open(os.path.join(root_dir, dst_file), 'w') as f:
        for name in lst:
            name = name.split('.png')[0]
            line = '/JPEGImages/{x}.jpg /SegmentationClass/{x}.png {cls}'.format(x=name, cls=0)
            f.write(line + '\n')


def get_val_statistics(val_dir):
    lst = os.listdir(val_dir)
    gps = [[] for i in range(21)]
    for anno in lst:
        anno_dir = os.path.join(val_dir, anno)
        label = np.asarray(Image.open(anno_dir), dtype=np.int32)
        classes = list(set(list(label.flatten())))
        for clss in classes:
            if not clss == 0:
                gps[clss].append(anno.strip())
    print('Number of classes:')
    for i in range(21):
        print(len(gps[i]))


if __name__ == "__main__":
    from pre_cluster import clust_images
    root_dir = 'data/personal'


    num_gps = [10]  # [1, 10, 20, 100, 200]
    for i in range(1, 3):
        sub_root = os.path.join(root_dir, 'id{}'.format(i))
        rename_imgs(sub_root)  # rename image files
        # colorize_data(sub_root, os.listdir(os.path.join(sub_root, 'SegmentationClass')))
        # get_val_list(sub_root, 'val_all.txt')
        # for ngp in num_gps:
        #     clust_images(os.path.join(sub_root, 'JPEGImages'), sub_root, num_gp=ngp)
        # get_val_statistics(os.path.join(sub_root, 'SegmentationClass'))
    
    # sub_root = "/disk1/datasets/VOCdevkit/VOC2012"
    # clust_images(os.path.join(sub_root, 'JPEGImages'), sub_root, num_gp=500, join_lb=True)

    # ids = os.listdir(root_dir)
    # for cur_id in ids:
    #     id_dir = os.path.join(root_dir, cur_id.split('_')[1].split('.')[0])
    #     os.makedirs(id_dir)
    #     lst_f = os.path.join(root_dir, cur_id)
    #     prepare_lst_by_class(id_dir, lst_f)
