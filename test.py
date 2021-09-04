# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import sys
import pdb

import argparse
import os
import os.path as osp
import pprint
import warnings
from advent.dataset import voc
from advent.model.psp import PSPGroupOCROnBase

from torch.utils import data
import torch
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.eval_UDA import evaluate_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_dataset_cfg(cfg):
    cfg_sup = {
        "data_dir": "./data/VOCdevkit/VOC2012",
        "list_file": "list/train_aug.txt",
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": 320,
        "shuffle": True,
        "base_size": 400,
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_supervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_sup_gp = {
        "data_dir": "./data/VOCdevkit/VOC2012",
        "list_file": cfg.VOC_GP_LIST,
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": 320,
        # "shuffle": True,
        "base_size": 400,
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_supervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_unsup = {
        "data_dir": "./data/personal/id{}".format(cfg.PERSON_ID),
        # "list_file": "train.txt",
        "list_file": cfg.PERSON_LIST,
        "weak_labels_output": "pseudo_labels/result/pseudo_labels",
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": 320,
        # "shuffle": True,
        "base_size": 400,
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_unsupervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_val = {
        "data_dir": "./data/personal/id{}".format(cfg.PERSON_ID),
        # "list_file": "train.txt",
        "list_file": cfg.PERSON_LIST,
        "batch_size": cfg.BATCH_SIZE,
        "val": True,
        "split": "val",
        "base_size": cfg.TRAIN.INPUT_SIZE_TARGET[1],
        "crop_size": cfg.TRAIN.INPUT_SIZE_TARGET[1],
        # "shuffle": True,
        # "shuffle_seed": 1,
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_val_fullim = {
        "data_dir": "./data/personal/id{}".format(cfg.PERSON_ID),
        "list_file": cfg.PERSON_LIST,
        "batch_size": cfg.BATCH_SIZE,
        "base_size": cfg.TRAIN.INPUT_SIZE_TARGET,
        "collate_fn": True,
        "split": "infer",
        "num_workers": cfg.NUM_WORKERS
    }
    source = cfg_sup if not cfg.VOC_GROUP else cfg_sup_gp
    target = cfg_unsup
    val = cfg_val_fullim if cfg.TEST.FULLMAP else cfg_val
    return source, target, val


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(exp_suffix):
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'DeepLabv2':
            model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        elif cfg.TRAIN.MODEL == 'psp':
            model = PSPGroupOCROnBase(pretrained=False, res=cfg.RES_GROUP, gpocr=cfg.GPOCR,
                                    ecd=cfg.ARCH)
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    src, trg, val = get_dataset_cfg(cfg)
    test_loader = voc.GroupLoader(val)
    # eval
    evaluate_domain_adaptation(models, test_loader, cfg)


if __name__ == '__main__':
    # eval all ID
    for i in range(1, 15):  # loop user IDs
        args = get_arguments()
        print('Called with args:')
        print(args)
        # LOAD ARGS
        assert args.cfg is not None, 'Missing cfg file'
        cfg_from_file(args.cfg)
        cfg.PERSON_ID = i
        cfg.TEST.RESTORE_FROM = "./data/final_res50_step2/id{}.pth".format(i)
        cfg.PERSON_LIST = "train_cluster80.txt"
        print("Evaluating ID {}...".format(i))
        print(cfg)
        cfg.TEST.EXT = cfg.TEST.EXT + 'id{}'.format(i)
        main(args.exp_suffix)
    
