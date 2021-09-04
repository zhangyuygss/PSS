from advent.dataset.my_base_dataset import BaseDataSet
from advent.dataset.base_loader import BaseDataLoader
from advent.utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json
from advent.utils.data_utils import get_ids_from_lst
import random

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):

        # self.root = os.path.join(self.root)
        file_list = os.path.join(self.root, self.list_file)

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if image.shape[-1] > 3:  # remove 4th layer of image, keep RGB
            image = image[:,:,:3]
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        if label.max() > 21 and not label.max() == 255:
            print('bad label: {}'.format(label_path))
        return image, label, image_id, image.shape[:2]

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)


def group_collect(batch):
    """
    Modified collect_fn for data_loader, for the sizes of items in each sample
    might not be the same
    :param batch:
    :return:
    """
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    batch = [item for item in batch if item is not None]
    result_batch = {}
    avg_h, avg_w, sizes = 0, 0, []
    for item in batch:
        avg_h += item[0].shape[0]
        avg_w += item[0].shape[1]
        sizes.append([item[0].shape[0], item[0].shape[1]])
    base_size = 480
    if avg_h > avg_w:
        h, w = int(avg_h * base_size / avg_w), base_size
    else:
        h, w = base_size, int(avg_w * base_size / avg_h)
    
    resz_im, resz_lb = [], []
    for item in batch:
        img = normalize(to_tensor(np.asarray(Image.fromarray(np.uint8(item[0])).resize((w, h), Image.BICUBIC))))
        resz_im.append(img)
        resz_lb.append(cv2.resize(item[1].numpy(), (w, h), interpolation=cv2.INTER_NEAREST))
    bt_im = torch.stack(resz_im)
    bt_lb = torch.from_numpy(np.array(resz_lb)).long()
    bt_name = [item[2] for item in batch]
    bt_gp = [item[3] for item in batch]
    return [bt_im, bt_lb, bt_name, bt_gp, sizes]


class GroupLoader(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        batch_size = kwargs['batch_size']
        
        num_workers = kwargs.pop('num_workers')
        try:
            custm_fn = kwargs.pop('collate_fn')
        except KeyError:
            custm_fn = False
        collate_fn = group_collect if custm_fn else None

        self.dataset = GroupDataset(**kwargs)
        self.val_list = get_ids_from_lst(os.path.join(kwargs['data_dir'], 'val_all.txt'))

        super(GroupLoader, self).__init__(self.dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, val_split=None, collate_fn=collate_fn)


class GroupDataset(BaseDataSet):
    def __init__(self, **kwargs):
        super(GroupDataset, self).__init__(**kwargs)
        self.num_classes = 21
        self.batch_sz = kwargs['batch_size']
        self._init_groups()
        self.num_groups = len(self.groups)
        self._reset_list()
        self.palette = pallete.get_voc_pallete(self.num_classes)
    
    def _set_files(self):
        pass

    def _init_groups(self):
        # self.root = os.path.join(self.root)
        file_list = os.path.join(self.root, self.list_file)
        self.file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        group_lsts = {}
        for idx, (file, label, group) in enumerate(self.file_list):
            if not group in group_lsts:
                group_lsts[group] = [[file, label, group]]
            else:
                group_lsts[group].append([file, label, group])
        self.groups = list(group_lsts.values())

    def _reset_list(self, sf_seed=None):
        # shuffle groups
        seed = sf_seed if sf_seed is not None else 1
        if self.split in ['val', 'infer'] or sf_seed is not None:
            random.seed(seed)
        for idx, group in enumerate(self.groups):
            random.shuffle(group)
        random.shuffle(self.groups)

        # generate batches of grouped dataset
        batches = []
        pos1, pos2 = 0, 0
        while pos1 < self.num_groups:
            bt, nxt_gp = [], False
            for idx in range(self.batch_sz):  # one batch
                file = self.groups[pos1][pos2]
                bt.append(file)
                pos2 += 1
                if pos2 >= len(self.groups[pos1]):
                    pos2 = 0  # pos2 out of index, go to head of group
                    nxt_gp = True
            batches.append(bt)
            if nxt_gp:  # go to next group
                pos1, pos2 = pos1 + 1, 0
        
        # generate list from batches for pytorch's default dataloader
        self.load_list = []
        random.shuffle(batches)
        for bt in batches:
            self.load_list += bt

    def __len__(self):
        return len(self.load_list)

    def _load_data(self, file):
        image_path = os.path.join(self.root, file[0][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if image.shape[-1] > 3:  # remove 4th layer of image, keep RGB
            image = image[:,:,:3]
        image_id = file[0].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, file[1][1:])
        if os.path.isfile(label_path):
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            if label.max() > 21 and not label.max() == 255:
                print('bad label: {}'.format(label_path))
        else:  # unlabeled data(only valset are labeled in personal dataset)
            if self.split != 'train_supervised':
                label = np.zeros(image.shape[:2], dtype=np.int32)
        return image, label, image_id, file[2], list(image.shape[:2])
    
    def infer_aug(self, image, label):
        # if self.base_size is not None:
        #     image, label = self._resize(image, label)
        #     image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        #     return image, label
        # image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        return image, label

    def __getitem__(self, index):
        image, label, image_id, group_id, imsz =  self._load_data(self.load_list[index])
        if self.split == 'val':
            image, label = self._val_augmentation(image, label)
        elif self.split == 'infer':
            image, label = self.infer_aug(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image, label, image_id, group_id, imsz
