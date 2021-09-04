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
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
        # if self.split == "val":
        #     file_list = os.path.join("dataloaders/voc_splits", f"{self.split}" + ".txt")
        # elif self.split in ["train_supervised", "train_unsupervised"]:
        #     file_list = os.path.join("dataloaders/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        # else:
        #     raise ValueError(f"Invalid split name {self.split}")

        self.root = os.path.join(self.root)
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
        # if label.shape[0] == image.shape[1] and not label.shape[0] == label.shape[1]:
        #     print('bad label: {}'.format(label_path))
        #     label = label.transpose(1, 0)
        #     lb_save = Image.fromarray(label)
        #     lb_save.save(label_path)
        return image, label, image_id

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
        # self.val_list = get_ids_from_lst(os.path.join(kwargs['data_dir'], 'val.txt'))

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)


class GroupLoader(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255
        
        num_workers = kwargs.pop('num_workers')

        self.dataset = GroupDataset(**kwargs)
        self.val_list = get_ids_from_lst(os.path.join(kwargs['data_dir'], 'val.txt'))

        super(GroupLoader, self).__init__(self.dataset, batch_size=1, shuffle=False,
                                num_workers=num_workers, val_split=None)


class GroupDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21
        self.batch_sz = kwargs['batch_size']
        
        if 'shuffle_seed' in kwargs:  # set random seed, designed for shuffle val set in a fixed manner
            random.seed(kwargs.pop('shuffle_seed'))
        
        try:
            self.shuffle = kwargs.pop('shuffle')
        except:
            self.shuffle = False
        # self.split = kwargs['split']

        self.palette = pallete.get_voc_pallete(self.num_classes)

        super(GroupDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root)
        file_list = os.path.join(self.root, self.list_file)

        self.file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]

        self.data_len = len(self.file_list)
        self._init_groups()
        if self.split == 'val':  # shuffle val only on init
            # TODO:need to fix data at the tail
            self._reset_groups(shuffle=True)

    def _init_groups(self):  # init groups from list
        self.pos1, self.pos2 = 0, 0
        group_lsts = {}
        for idx, (file, label, cls) in enumerate(self.file_list):
            # cls = int(cls)
            if not cls in group_lsts:
                group_lsts[cls] = [[file, label, cls]]
            else:
                group_lsts[cls].append([file, label, cls])
        self.groups = list(group_lsts.values())

    def _reset_groups(self, shuffle):  # reset groups every epoch, should be called manully
        if shuffle:  #  shuffle groups
            for idx, group in enumerate(self.groups):
                random.shuffle(group)
            random.shuffle(self.groups)
        self.pos1, self.pos2 = 0, 0
    
    def __len__(self):
        return int((self.data_len / self.batch_sz) * 1.5)  # 1.5 to cover all the data

    def __getitem__(self, index):
        images, labels, imids, group_ids = [], [], [], []
        new_group = False
        for i in range(self.batch_sz):
            if self.pos1 >= len(self.groups):  # go to head of all samples
                self.pos1, self.pos2 = 0, 0
            try:
                file = self.groups[self.pos1][self.pos2]
                self.pos2 += 1
            except IndexError:  # self.pos2 out of index, go to next group
                file = random.sample(self.groups[self.pos1], 1)[0]
                new_group = True
            try:
                image, label, image_id = self._prepare_sample(file)
            except FileNotFoundError:
                print(file)
                continue
            images.append(image)
            labels.append(label)
            imids.append(image_id)
            group_ids.append(file[-1])
        # print(self.pos1, self.pos2)
        if new_group:
            self.pos1, self.pos2 = self.pos1 + 1, 0
        return torch.stack(images), torch.stack(labels), imids, group_ids            

    def _load_data(self, file):
        try:
            image_path = os.path.join(self.root, file[0][1:])
        except:
            print(file)
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if image.shape[-1] > 3:  # remove 4th layer of image, keep RGB
            image = image[:,:,:3]
        image_id = file[0].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, file[1][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        if label.max() > 21 and not label.max() == 255:
            print('bad label: {}'.format(label_path))
        return image, label, image_id
    
    def infer_aug(self, image, label):
        if self.base_size is not None:
            image, label = self._resize(image, label)
            image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
            return image, label

        image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        return image, label

    def _prepare_sample(self, file):
        image, label, image_id =  self._load_data(file)
        if self.split == 'val':
            image, label = self._val_augmentation(image, label)
        elif self.split == 'infer':
            image, label = self.infer_aug(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image, label, image_id
