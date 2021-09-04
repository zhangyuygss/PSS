import os
import cv2
from tqdm import tqdm
from multiprocessing import Process
from PIL import Image
import numpy as np

bad_list = ['IMG20180815205608', 'IMG20180920231235', 'IMG_20200426_222108', 'IMG20181027120550', ]

root_dir = '/disk1/datasets/big_id1'
save_dir = '/disk1/datasets/personal_rsz'
base_size = 1024
persons = os.listdir(root_dir)
for person in persons:
    img_dir = os.path.join(root_dir, person, 'JPEGImages')
    sgcls_dir = os.path.join(root_dir, person, 'SegmentationClass')
    sgobj_dir = os.path.join(root_dir, person, 'SegmentationObject')
    # os.makedirs(os.path.join(save_dir, person, 'JPEGImages'))
    # os.makedirs(os.path.join(save_dir, person, 'SegmentationClass'))
    # os.makedirs(os.path.join(save_dir, person, 'SegmentationObject'))
    img_lst = os.listdir(img_dir)
    # for im in tqdm(img_lst):
    for im_name in bad_list:
        # im_name = im.split('.jpg')[0]
        img = cv2.imread(os.path.join(img_dir, im_name + '.jpg'))
        # resize
        h_pre, w_pre = img.shape[:2]
        if max(h_pre, w_pre) > base_size:
            h, w = (base_size, int(1.0 * base_size * w_pre / h_pre + 0.5)) if h_pre > w_pre else (int(1.0 * base_size * h_pre / w_pre + 0.5), base_size)
        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # if img.shape[-1] > 3:
        #     img = img[:,:,:3]
        # cv2.imwrite(os.path.join(save_dir, person, 'JPEGImages', im_name + '.jpg'), img)

        if os.path.isfile(os.path.join(sgcls_dir, im_name + '.png')):
            sgcls = np.asarray(Image.open(os.path.join(sgcls_dir, im_name + '.png')), dtype=np.int32)
            sgobj = np.asarray(Image.open(os.path.join(sgobj_dir, im_name + '.png')), dtype=np.int32)
            if sgcls.shape[0] != h_pre:  # wrong label orentation
                sgcls = sgcls.transpose(1, 0).transpose(1, 0).transpose(1, 0)[::-1,:]
                sgobj = sgobj.transpose(1, 0).transpose(1, 0).transpose(1, 0)[::-1,:]
                print(im_name)
                sgcls = cv2.resize(sgcls, (w, h), interpolation=cv2.INTER_NEAREST)
                sgobj = cv2.resize(sgobj, (w, h), interpolation=cv2.INTER_NEAREST)
                Image.fromarray(sgcls).save(os.path.join(save_dir, person, 'SegmentationClass', im_name + '.png'))
                Image.fromarray(sgobj).save(os.path.join(save_dir, person, 'SegmentationObject', im_name + '.png'))

