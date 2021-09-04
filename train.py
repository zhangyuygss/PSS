# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import sys
import numpy as np
import yaml
import torch
from torch.utils import data
from advent.utils.helpers import Logger
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.psp import PSPGroupOCROnBase
# from advent.model.group_modules import AggregateFuse
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset import voc
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.train_UDA import train_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_dataset_cfg(cfg):
    cfg_sup = {
        "data_dir": "/disk1/datasets/VOCdevkit/VOC2012",
        "list_file": "list/train_aug.txt",
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": cfg.TRAIN.INPUT_SIZE_SOURCE[1],
        "shuffle": True,
        "base_size": cfg.TRAIN.INPUT_SIZE_SOURCE[0],
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_supervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_sup_gp = {
        "data_dir": cfg.DATA_DIRECTORY_SOURCE,
        "list_file": cfg.VOC_GP_LIST,
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": cfg.TRAIN.INPUT_SIZE_SOURCE[1],
        # "shuffle": True,
        "base_size": cfg.TRAIN.INPUT_SIZE_SOURCE[0],
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_supervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_unsup = {
        "data_dir": "/disk1/datasets/personal/id{}".format(cfg.PERSON_ID),
        # "list_file": "train.txt",
        "list_file": cfg.PERSON_LIST,
        "weak_labels_output": "pseudo_labels/result/pseudo_labels",
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": cfg.TRAIN.INPUT_SIZE_TARGET[1],
        # "shuffle": True,
        "base_size": cfg.TRAIN.INPUT_SIZE_TARGET[0],
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_unsupervised",
        "num_workers": cfg.NUM_WORKERS
    }
    cfg_val = {
        "data_dir": "/disk1/datasets/personal/id{}".format(cfg.PERSON_ID),
        # "list_file": "train.txt",
        "list_file": cfg.VAL_LIST,
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
        "data_dir": "/disk1/datasets/personal/id{}".format(cfg.PERSON_ID),
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
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''

    log_file = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_log.txt')
    sys.stdout = Logger(log_file)

    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    group = None
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    elif cfg.TRAIN.MODEL == 'psp':
        # model = PSPNet(pretrained=True, res=cfg.RES_GROUP, weight_gp=cfg.WEIGHT_GROUP)
        model = PSPGroupOCROnBase(pretrained=True, res=cfg.RES_GROUP, gpocr=cfg.GPOCR,
                                ecd=cfg.ARCH)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    src, trg, val = get_dataset_cfg(cfg)
    source_loader = voc.GroupLoader(src) if cfg.VOC_GROUP else voc.VOC(src)
    target_loader = voc.GroupLoader(trg)
    if cfg.PERSON_GROUP == False:
        val['batch_size'] = 1  # batch size set to 1 for evaluation of non-group methods
    val_loader = voc.GroupLoader(val)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    group = cfg.PERSON_GROUP
    fake_loder = None
    if cfg.FAKELB_LIST != "":
        cfg_fake_gp = {
        "data_dir": "/disk1/datasets/personal/id{}".format(cfg.PERSON_ID),
        "list_file": cfg.FAKELB_LIST,
        "batch_size": cfg.BATCH_SIZE,
        "crop_size": cfg.TRAIN.INPUT_SIZE_SOURCE[1],
        # "shuffle": True,
        "base_size": cfg.TRAIN.INPUT_SIZE_SOURCE[0],
        "scale": True,
        "augment": True,
        "flip": True,
        "rotate": False,
        "blur": False,
        "split": "train_supervised",
        "num_workers": cfg.NUM_WORKERS,
        }
        fake_loder = voc.GroupLoader(cfg_fake_gp)

    train_domain_adaptation(model, source_loader, target_loader, val_loader,
                            cfg, group, fk_loader=fake_loder)


if __name__ == '__main__':
    main()
