# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from advent.utils.helpers import Logger
import sys
import cv2

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load
# from advent.domain_adaptation.train_UDA import eval_net
from entropy import colorize_save_voc
from advent.utils.func import prob_2_entropy
import torch.nn.functional as F


def cluster_subdomain(entropy_list, lambda1, save_root):
    entropy_list = sorted(entropy_list.items(), key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]
    # entropy_rank = pickle.load(open('tmp.pkl', 'rb'))

    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

    # merge with group list
    fin_easy, fin_hard = [], []
    ext = save_root.split('/')[-1]
    gp_list = open(os.path.join(save_root.split(ext)[0], 'train_cluster80.txt'), 'r').readlines()
    for l in gp_list:
        img, lb, gp_id = l.strip().split(' ')
        name = img.split('/')[-1].split('.')[0]
        new_l = "{} /color_masks/{}.png {}".format(img, name, gp_id)
        if name in hard_split:
            fin_hard.append(new_l)
        else:
            fin_easy.append(new_l)

    with open(os.path.join(save_root, 'easy_split.txt'),'w') as f:
        for item in fin_easy:
            f.write('%s\n' % item)

    with open(os.path.join(save_root, 'hard_split.txt'),'w') as f:
        for item in fin_hard:
            f.write('%s\n' % item)

    return copy_list


def eval_net(model, val_loader, cfg, group=False, cls_mask=None, save_dir=None,
            per_img_iou=False, val_aux=False, ent_rank=False, save_all=False,
            set_bg_ign=False):
    val_iter = iter(val_loader)
    palette = val_loader.dataset.palette
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    valed_lst = []
    entropy_list = {}
    img_iou = {}
    for index in tqdm(range(len(val_iter))):
        (image, label, img_names, gp_ids, im_szs) = next(val_iter)
        if type(img_names[0]) == list:
            img_names = [name[0] for name in img_names]
        with torch.no_grad():
            pred_main, pred_aux = model(image.cuda(), group=group)
            prediction = pred_main if not val_aux else pred_aux
            output = prediction.cpu().data.numpy()
            output = output.transpose(0, 2, 3, 1)
            output = np.argmax(output, axis=3)
        label = label.numpy()

        # save all training results for step2
        if save_all:
            for idx in range(output.shape[0]):
                cur_out = cv2.resize(output[idx], (im_szs[idx][1], im_szs[idx][0]), interpolation=cv2.INTER_NEAREST)
                if set_bg_ign:
                    msk = cur_out == 0
                    cur_out[msk] = 255
                colorize_save_voc(cur_out, img_names[idx], palette, save_dir)
        
        eval_mask = np.zeros(label.shape[0])  # select labeled and un-evaluated images
        for idx, name in enumerate(img_names):
            if name in val_loader.val_list and not name in valed_lst:
                eval_mask[idx] = 1
                valed_lst.append(name)
        eval_mask = eval_mask.astype(np.bool)
        if eval_mask.any():  # evaluate labeled images
            output, label = output[eval_mask], label[eval_mask]
            mask_names = list(np.array(img_names)[eval_mask])
            mask_szs = np.array(im_szs)[eval_mask]
            for idx in range(output.shape[0]):
                out, lb = output[idx], label[idx]
                if out.shape != lb.shape:
                    out = cv2.resize(out, lb.shape[::-1], interpolation=cv2.INTER_NEAREST)
                if per_img_iou:
                    img_iou[mask_names[idx]] = single_image_iou(out, lb)
                hist += fast_hist(lb.flatten(), out.flatten(), cfg.NUM_CLASSES)
                if save_dir is not None and not save_all:
                    save_sz = tuple(mask_szs[idx][::-1])
                    out_save = cv2.resize(out, save_sz, interpolation=cv2.INTER_NEAREST)
                    colorize_save_voc(out_save, mask_names[idx], palette, save_dir)
        
        if ent_rank:  # entropy rank for intra-domain step
            pred_entropy = prob_2_entropy(F.softmax(pred_main, 1))
            for idx in range(pred_main.shape[0]):
                name, pred_ent = img_names[idx], pred_entropy[idx]
                entropy_list[name] = pred_ent.mean().item()

    if ent_rank:
        save_dir = os.path.join('/disk1/datasets/personal/id{}'.format(cfg.PERSON_ID), 'entrank')
        os.makedirs(save_dir)
        cluster_subdomain(entropy_list, 0.6, save_dir)
    
    inters_over_union_classes = per_class_iu(hist)
    if per_img_iou:
        pickle_dump(img_iou, osp.join(save_dir, 'per_img_iou.pkl'))
        FBIoU = np.array(list(img_iou.values())).mean()
        print("FBIoU:", FBIoU)
    return inters_over_union_classes, FBIoU


def single_image_iou(pred, target):
    p1d = pred.flatten()
    t1d = target.flatten()
    plabeled, tlabeled = p1d > 0, t1d > 0
    inter = p1d == t1d
    inter = inter * plabeled
    union = plabeled + tlabeled
    if np.sum(union) == 0:
        iou = 0
    else:
        iou = np.sum(inter) / np.sum(union) 
    return iou


def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True, save=False):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, save)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
              device, test_loader, interp=None,
              fixed_test_size=None, verbose=None, save=False):
    restore_from = osp.join(cfg.TEST.RESTORE_FROM)
    print("Evaluating model", restore_from)
    load_checkpoint_for_evaluation(models[0], restore_from, device)
    if cfg.TEST.SAVE_PRED:
        save_dir = osp.join(cfg.TEST.RESTORE_FROM.split('id')[0],
                            cfg.TEST.RESTORE_FROM.split('/')[-1].split('.')[0]+ '_'+cfg.TEST.EXT)
        print("Save to ===========>", save_dir)
    else: save_dir = None
    if not cfg.TEST.FULLMAP:
        IoUs, _ = eval_net(models[0], test_loader, cfg, group=cfg.PERSON_GROUP,
                                save_dir=save_dir, per_img_iou=True,
                                ent_rank=False, save_all=False, set_bg_ign=False)
    else:
        IoUs, _ = eval_net(models[0], test_loader, cfg, group=cfg.PERSON_GROUP,
                                save_dir=save_dir, per_img_iou=True,
                                ent_rank=cfg.TEST.ENTRANK, save_all=True, set_bg_ign=False)

    # pickle_dump(img_ious, os.path.join(save_dir, 'per_img_iou.pkl'))
    print(IoUs.mean())
    display_stats(cfg, IoUs)


def eval_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP * 10  # start from step 20000
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res_{}.pkl'.format(cfg.TEST.EXT))
    save_txt = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res_{}.txt'.format(cfg.TEST.EXT))
    sys.stdout = Logger(save_txt)
    # if osp.exists(cache_path):
    #     all_res = pickle_load(cache_path)
    # else:
    #     all_res = {}
    cur_best_fiou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
    # for i_iter in range(1,16):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 
                            f'{cfg.TEST.SNAPSHOT_DIR[1]}{i_iter}.pth')
        print(restore_from)
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                continue
        print("Evaluating model", restore_from)
        load_checkpoint_for_evaluation(models[0], restore_from, device)
        # eval
        if cfg.TEST.SAVE_PRED:
            save_dir = osp.join(cfg.TEST.SNAPSHOT_DIR[0],
                                restore_from.split('/')[-1].split('.')[0]+'_'+cfg.TEST.EXT)
        else: save_dir = None
        inters_over_union_classes, FIoU = eval_net(models[0], test_loader, cfg, group=cfg.PERSON_GROUP,
                                    save_dir=save_dir, per_img_iou=True, ent_rank=cfg.TEST.ENTRANK, 
                                    save_all=True, set_bg_ign=False)

        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_fiou < FIoU:
            cur_best_fiou = FIoU
            cur_best_model = restore_from
        print('\tCurrent FIoU:{}; Current MIoU:{}'.format(FIoU, computed_miou))
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best FIoU:', cur_best_fiou)
        if verbose:
            display_stats(cfg, inters_over_union_classes)
        

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(saved_state_dict, strict=False)
    model.eval()
    model.cuda(device)


def display_stats(cfg, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(str(ind_class)
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
