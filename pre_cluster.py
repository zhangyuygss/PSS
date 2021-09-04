from advent.model.backbones.resnet_backbone import ResNetBackbone
from advent.utils.helpers import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math , time
from itertools import chain
import contextlib
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.models as models
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
import shutil
import random
import pickle


class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()
        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.prefix = nn.Sequential(
            orig_resnet.conv1,
            orig_resnet.bn1,
            orig_resnet.relu,
            orig_resnet.maxpool
        )
        # self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.gap = orig_resnet.avgpool

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        x = self.prefix(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x


def clust_images(img_dir, root_dir, num_gp=40, sample_val=None, join_val=False):
    # get imagenet feature
    im_lst = os.listdir(img_dir)
    if not os.path.isfile(os.path.join(root_dir, 'imgnet_fts.pkl')):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean, std)
        resnet = models.resnet101(pretrained=True)
        model = NormalResnetBackbone(resnet).cuda()
        im_fts = []
        for im in tqdm(im_lst):
            image_path = os.path.join(img_dir, im)
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            if image.shape[-1] > 3:  # remove 4th layer of image, keep RGB
                image = image[:,:,:3]
            img = Image.fromarray(np.uint8(image)).resize((400, 300), Image.BICUBIC)
            img = norm(to_tensor(img)).cuda().unsqueeze(0)
            im_ft = model(img).squeeze().cpu().data.numpy()
            im_fts.append(im_ft)
        res = [im_lst, im_fts]
        pickle.dump(res, open(os.path.join(root_dir, 'imgnet_fts.pkl'), 'wb'))
    else:
        res = pickle.load(open(os.path.join(root_dir, 'imgnet_fts.pkl'), 'rb'))
        im_lst, im_fts = res

    # clust
    lb = KMeans(n_clusters=num_gp, random_state=0).fit(im_fts)
    tmp_dir = os.path.join(root_dir, 'cluster_{}'.format(num_gp))
    cluster_lst = os.path.join(root_dir, 'train_cluster{}.txt'.format(num_gp))
    os.makedirs(tmp_dir, exist_ok=True)
    gps = {}
    for idx, im in tqdm(enumerate(im_lst)):
        im_gp = lb.labels_[idx]
        if im_gp in gps:
            gps[im_gp].append(im)
        else:
            gps[im_gp] = [im]
        # show clust result
        shutil.copyfile(os.path.join(img_dir, im),
                    os.path.join(tmp_dir, '{}_{}'.format(im_gp, im)))
    rst_lst, sp_val_lst = [], []
    for key in gps:  # write clust result to list for training
        cur_gp_lst = gps[key]
        lines = ['/JPEGImages/{x}.jpg /SegmentationClass/{x}.png {cls}'.format(x=name.split('.')[0], cls=key) for name in cur_gp_lst]
        rst_lst += lines
        if sample_val is not None:
            sp_lines = random.sample(lines, int(len(lines) * sample_val))
            sp_val_lst += sp_lines
    with open(cluster_lst, 'w') as f:
        for l in rst_lst:
            f.write(l + '\n')
    if sample_val is not None:  # select validation list from clusts
        val_list_f = os.path.join(root_dir, '{}val_cluster{}.txt'.format(root_dir.split('/')[-1], num_gp))
        with open(val_list_f, 'w') as f:
            for l in sp_val_lst:
                f.write(l + '\n')
    
    # sample validation list from groups
    if join_val:
        val_rate = 0.3
        group_val = os.path.join(root_dir, 'val_{}.txt'.format(val_rate))
        gps = {}
        for idx, line in tqdm(enumerate(rst_lst)):
            im_name = line.strip().split(' ')[0].split('/')[-1].split('.')[0]
            im_gp = line.strip().split(' ')[-1]
            if im_gp in gps:
                gps[im_gp].append(line.strip())
            else:
                gps[im_gp] = [line.strip()]
        with open(group_val, 'w') as f:
            for gp in gps:
                gp_imgs = gps[gp]
                selected_imgs = random.sample(gp_imgs, k=int(len(gp_imgs) * val_rate))
                for im in selected_imgs:  # filter out unlabeled samples
                    tmp = os.path.join(root_dir, im.split(' ')[1][1:])
                    if os.path.exists(tmp):
                        f.write(im + '\n')



# # # select val from groups
# # gped_list = os.path.join(root_dir, 'train_cluster{}.txt'.format(num_gp))
# # rst_lst = open(gped_list, 'r').readlines()



