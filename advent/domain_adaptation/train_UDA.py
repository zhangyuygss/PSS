# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path
import pickle
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from advent.utils.func import per_class_iu, fast_hist
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.domain_adaptation.eval_UDA import eval_net, display_stats
from advent.utils.viz_segmask import colorize_mask


def train_advent(model, trainloader, targetloader, val_loader, cfg, group=False,
                fk_loader=None):
    ''' UDA training with advent, psp resnet50 backbone
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    # device = cfg.GPU_ID
    device = torch.device("cuda")
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # print("init model")
    # SEGMNETATION NETWORK
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('experiments/snapshots/id5_320-imgpocr-gpscale8_gponpsp/model_70000.pth'))
    model.train()
    model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_aux = torch.nn.DataParallel(d_aux)
    d_main = torch.nn.DataParallel(d_main)

    # d_aux.load_state_dict(torch.load('experiments/snapshots/id5_320-imgpocr-gpscale8_gponpsp/model_70000_D_aux.pth'))
    # d_main.load_state_dict(torch.load('experiments/snapshots/id5_320-imgpocr-gpscale8_gponpsp/model_70000_D_main.pth'))

    d_aux.train()
    d_aux.to(device)
    # seg maps, i.e. output, level
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    if fk_loader is not None:
        fk_loader_iter = enumerate(fk_loader)

    # for eval
    cache_path = osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'all_res.pkl')
    all_res = {}
    cur_best_miou, best_fbiou = -1, 0
    cur_best_model = ''
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        try:
            _, (images_source, labels, src_names, voc_ids, _) = trainloader_iter.__next__()
        except StopIteration:
            try:  
                trainloader.dataset._reset_list()
            except:  # voc dataset
                pass
            trainloader_iter = enumerate(trainloader)
            _, (images_source, labels, src_names, voc_ids, _) = trainloader_iter.__next__()
            print("~~~~End of epoch for voc, iteration {}".format(i_iter))
        # images_source, labels = images_source.squeeze(0), labels.squeeze(0)
        pred_src_main, pred_src_aux = model(images_source.cuda(device), group=group)
        # print('src forward done')
        if cfg.TRAIN.MULTI_LEVEL:
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        # if not type(loss) == float:  # for all 255 pseduo labels
        loss.backward()

        if fk_loader is not None and i_iter % 2 == 0:  # train segnet with easy target pesudo labels
            try:
                _, (images_psedo, pslbs, ps_names, psids, _) = fk_loader_iter.__next__()
            except StopIteration:
                try:  
                    trainloader.dataset._reset_list()
                except:  # voc dataset
                    pass
                fk_loader_iter = enumerate(trainloader)
                _, (images_psedo, pslbs, ps_names, psids, _) = fk_loader_iter.__next__()
                print("~~~~End of epoch for voc, iteration {}".format(i_iter))
            pred_ps_main, pred_ps_aux = model(images_psedo.cuda(device), group=group)
            if cfg.TRAIN.MULTI_LEVEL:
                loss_seg_ps_aux = loss_calc(pred_ps_aux, pslbs, device)
            else:
                loss_seg_ps_aux = 0
            loss_seg_ps_main = loss_calc(pred_ps_main, pslbs, device)
            loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_ps_main
                    + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_ps_aux)
            # if not type(loss) == float:  # for all 255 pseduo labels
            loss.backward()
        
        trg_aux_sup = False
        
        # adversarial training ot fool the discriminator
        try:
            _, (images, person_labels, img_names, gp_ids, _) = targetloader_iter.__next__()
        except StopIteration:
            print("----End of epoch for personal, iteration {}".format(i_iter))
            targetloader.dataset._reset_list()
            targetloader_iter = enumerate(targetloader)
            _, (images, person_labels, img_names, gp_ids, _) = targetloader_iter.__next__()
        # images, person_labels = images.squeeze(0), person_labels.squeeze(0)
        pred_trg_main, pred_trg_aux = model(images.cuda(device), group=group)
        if cfg.TRAIN.MULTI_LEVEL and cfg.TRAIN.AUXFORADV:
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
            # supervise target aux with pred_trg_main
            if trg_aux_sup:
                loss_trg_aux_seg = loss_calc(pred_trg_aux, pred_trg_main.max(1)[1], device)
                loss_trg_aux_seg *= cfg.TRAIN.LAMBDA_SEG_AUX
        else:
            loss_adv_trg_aux = 0
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        if trg_aux_sup: loss += loss_trg_aux_seg
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL and cfg.TRAIN.AUXFORADV:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL and cfg.TRAIN.AUXFORADV:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        # optimizer_decoder.step()
        if cfg.TRAIN.MULTI_LEVEL and cfg.TRAIN.AUXFORADV:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {
                          'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main
                          }
        print_losses(current_losses, i_iter, cfg)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            # eval model on valset
            model.eval()
            if cfg.TEST.SAVE_PRED:
                save_dir = osp.join(snapshot_dir, 'model_{}{}'.format(i_iter, cfg.TEST.EXT))
            else: save_dir = None
            iou, fbiou = eval_net(model, val_loader, cfg, group=group, save_dir=save_dir,
                        per_img_iou=True)
            if trg_aux_sup:
                aux_iou, _ = eval_net(model, val_loader, cfg, group=group,
                                save_dir=save_dir+'aux', val_aux=True)
                print(aux_iou, aux_iou.mean())
            model.train()
            computed_miou = round(np.nanmean(iou) * 100, 2)

            # torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            # torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            # torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')

            if cur_best_miou < computed_miou or best_fbiou < fbiou:
                print('taking snapshot ...')
                print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
                torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
                torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
                torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')

                if cur_best_miou < computed_miou:
                    cur_best_miou = computed_miou
                if best_fbiou < fbiou:
                    best_fbiou = fbiou
                best_iter = i_iter
                all_res['best'] = [(best_iter, cur_best_miou, best_fbiou)]
            all_res[i_iter] = iou
            with open(cache_path, 'wb') as f:
                pickle.dump(all_res, f)
            print('\tCurrent mIoU:', computed_miou)
            # print('\tCurrent IoU:', iou)
            display_stats(cfg, iou)
            print('\tCurrent best model:', best_iter)
            print('\tCurrent best mIoU:', cur_best_miou)

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')

def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter, cfg):
    list_strings = ['EXP:{}'.format(cfg.EXP_NAME)]
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, val_loader, cfg, group=False,
                            fk_loader=None):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, val_loader, cfg, group=group, fk_loader=fk_loader)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

