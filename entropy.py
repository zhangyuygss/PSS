##----------------------------------------------------------
# written by Fei Pan
#
# to get the entropy ranking from Inter-domain adaptation process 
#-----------------------------------------------------------
import sys
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
import pickle
from torch.utils import data
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.psp import PSPGroupOCROnBase
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet
from advent.utils.func import prob_2_entropy
from advent.dataset import voc
import torch.nn.functional as F
from advent.utils.func import loss_calc, bce_loss
from advent.domain_adaptation.config import cfg, cfg_from_file
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg
from advent.utils.func import per_class_iu, fast_hist

cfg_unsup = {
    "data_dir": "/disk1/datasets/personal/id{}".format(cfg.PERSON_ID),
    "list_file": cfg.PERSON_LIST,
    "batch_size": cfg.BATCH_SIZE,
    # "shuffle": True,
    "base_size": cfg.TRAIN.INPUT_SIZE_TARGET,
    "collate_fn": True,
    "split": "infer",
    "num_workers": cfg.NUM_WORKERS
}
# cfg_unsup = {
#     "data_dir": "/disk1/datasets/personal/id2",
#     # "list_file": "train.txt",
#     "list_file": "train_cluster40.txt",
#     "batch_size": 8,
#     "val": True,
#     "split": "val",
#     "base_size": 320,
#     "crop_size": 320,
#     # "shuffle": True,
#     # "shuffle_seed": 1,
#     "num_workers": 0
# }

rare_class = [3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17]


def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    
    return new_mask    

def colorize_save(output_pt_tensor, name):
    output_np_tensor = output_pt_tensor.cpu().data.numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0) 
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_Img         = Image.fromarray(mask_np_tensor)
    mask_color       = colorize(mask_np_tensor)  

    name = name.split('/')[-1]
    mask_Img.save('./color_masks/%s' % (name))
    mask_color.save('./color_masks/%s_color.png' % (name.split('.')[0]))

def find_rare_class(output_pt_tensor):
    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0)
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_np_tensor   = np.reshape(mask_np_tensor, 512*1024)
    unique_class     = np.unique(mask_np_tensor).tolist()
    commom_class     = set(unique_class).intersection(rare_class)
    return commom_class


def cluster_subdomain(entropy_list, lambda1, save_root):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    # entropy_rank = [item[0] for item in entropy_list]

    entropy_rank = pickle.load(open('tmp.pkl', 'rb'))
    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

    # merge with group list
    fin_easy, fin_hard = [], []
    ext = save_root.split('/')[-1]
    gp_list = open(os.path.join(save_root.split(ext)[0], 'train.txt'), 'r').readlines()
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

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def get_arguments():
    """
    Parse input arguments 
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")

    parser.add_argument('--best_iter', type=int, default=70000,
                        help='iteration with best mIoU')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='add normalizor to the entropy ranking')
    parser.add_argument('--lambda1', type=float, default=0.67, 
                        help='hyperparameter lambda to split the target domain')
    parser.add_argument('--cfg', type=str, default='../ADVENT/advent/scripts/configs/advent_ent.yml',
                        help='optional config file' )
    return parser.parse_args()


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


def main(args):

    # load configuration file 
    device = cfg.GPU_ID    
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    sub_dir = 'intra-gp'
    save_root = '/disk1/datasets/personal/id2'
    mask_dir = os.path.join(save_root, sub_dir, 'color_masks')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
    # load model with parameters trained from Inter-domain adaptation
    model_gen = PSPGroupOCR(pretrained=True, res=cfg.RES_GROUP, weight_gp=cfg.WEIGHT_GROUP)
    
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{args.best_iter}.pth')
    print("Loading the generator:", restore_from)
    
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    
    # load data
    target_loader = voc.GroupLoader(cfg_unsup)
    target_loader_iter = enumerate(target_loader)

    # eval_net(model_gen, target_loader, cfg, group=True)

    # upsampling layer
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
    #                             align_corners=True)

    entropy_list = []
    inferred_list = []
    for index in tqdm(range(len(target_loader))):
        # _, batch = target_loader_iter.__next__()
        _, (images, person_labels, img_names, gp_ids) = target_loader_iter.__next__()
        images = images.squeeze(0)
        # image, _, _, name = batch
        with torch.no_grad():
            _, pred_trg_main = model_gen(images.cuda(device), group=True)
            # pred_trg_main    = interp_target(pred_trg_main)
            if args.normalize == True:
                normalizor = (11-len(find_rare_class(pred_trg_main))) / 11.0 + 0.5
            else:
                normalizor = 1
            eval_mask = np.zeros(images.shape[0])
            for idx, name in enumerate(img_names):
                if not name[0] in inferred_list:
                    eval_mask[idx] = 1
                    inferred_list.append(name[0])
            eval_mask = eval_mask.astype(np.bool)
            if not eval_mask.any():
                continue
            else:
                pred_trg_main, img_names = pred_trg_main[eval_mask], img_names[eval_mask]
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            for idx in range(pred_trg_main.shape[0]):
                name, pred_ent = img_names[idx][0], pred_trg_entropy[idx]
                entropy_list.append((name, pred_ent.mean().item() * normalizor))
                palette = target_loader.dataset.palette
                output_np = pred_trg_main[idx].cpu().data.numpy()
                mask_np   = output_np.transpose(1,2,0) 
                mask_np   = np.asarray(np.argmax(mask_np, axis=2), dtype=np.uint8)
                colorize_save_voc(mask_np, name, palette, mask_dir)

    # split the enntropy_list into 
    cluster_subdomain(entropy_list, args.lambda1, os.path.join(save_root, sub_dir))

if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    main(args)
