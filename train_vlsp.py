import argparse

from config import ConfigArgs as args
import os, sys, shutil
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import glob

import numpy as np
import pandas as pd
from collections import deque
from model import TPGST
# from data import SpeechDataset, collate_fn, load_vocab
from data_vlsp import VLSPSpeechDataset, collate_fn, load_vocab
from utils import att2img, plot_att, lr_policy


# torch.autograd.set_detect_anomaly = True


def train(model, data_loader, valid_loader, optimizer, scheduler, batch_size=32, ckpt_dir=None, writer=None,
          DEVICE=None):
    """
    train function

    :param model: nn module object
    :param data_loader: data loader for training set
    :param valid_loader: data loader for validation set
    :param optimizer: optimizer
    :param scheculer: for scheduling learning rate
    :param batch_size: Scalar
    :param ckpt_dir: String. checkpoint directory
    :param writer: Tensorboard writer
    :param DEVICE: 'cpu' or 'gpu'

    """
    epochs = 0
    global_step = args.global_step
    criterion = nn.L1Loss()  # default average
    bce_loss = nn.BCELoss()
    xe_loss = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    GO_frames = torch.zeros([batch_size, 1, args.n_mels * args.r]).to(DEVICE)  # (N, Ty/r, n_mels)
    idx2char = load_vocab()[-1]
    print("Start training...")
    while global_step < args.max_step:
        epoch_loss_mel, epoch_loss_fmel, epoch_loss_ff = 0., 0., 0.
        for step, (texts, mels, ff) in tqdm(enumerate(data_loader), total=len(data_loader), unit='B', ncols=70,
                                            disable=True,
                                            leave=False):
            optimizer.zero_grad()
            texts, mels, ff = texts.to(DEVICE), mels.to(DEVICE), ff.to(DEVICE)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)
            refs = mels.view(mels.size(0), -1, args.n_mels).unsqueeze(1)  # (N, 1, Ty, n_mels)
            if type(model).__name__ == 'TPGST':
                if model.type == "TPSE":
                    mels_hat, fmels_hat, A, style_attentions, ff_hat, se, tpse = model(texts, prev_mels, refs)
                    loss_tp__ = criterion(tpse, se.detach())
                else:  # TPCW
                    mels_hat, fmels_hat, A, style_attentions, ff_hat, se, tpcw = model(texts, prev_mels, refs)
                    sa = style_attentions.detach()
                    if sa.ndim == 1:
                        sa = sa.unsqueeze(0)  # REVIEW style_attentions dim = 1 if batch size = 1
                    loss_tp__ = xe_loss(tpcw, sa)  # Same size (N, n_tokens)
                    # loss_tp__ = xe_loss(tpcw.view(-1), sa.view(-1))  # Same size (N, n_tokens)
            else:
                mels_hat, fmels_hat, A, ff_hat = model(texts, prev_mels)

            loss_mel = criterion(mels_hat, mels)
            fmels = mels.view(mels.size(0), -1, args.n_mels)
            loss_fmel = criterion(fmels_hat, fmels)
            loss_ff = bce_loss(ff_hat, ff)

            if global_step > args.tp_start and type(model).__name__ == 'TPGST':
                loss = loss_mel + 0.01 * loss_ff + 0.01 * loss_tp__
                loss = loss_mel + 0.2 * loss_ff + 0.2 * loss_tp__
            else:
                loss = loss_mel + 0.01 * loss_ff

            # scaler.scale(loss).backward()
            # # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)
            # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            # scaler.step(optimizer)
            # # Updates the scale for next iteration.
            # scaler.update()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()

            epoch_loss_mel += loss_mel.item()
            epoch_loss_fmel += loss_fmel.item()
            epoch_loss_ff += loss_ff.item()

            if global_step % args.log_term == 0:
                writer.add_scalar('batch/loss_mel', loss_mel.item(), global_step)
                if type(model).__name__ == 'TPGST':
                    if model.type == "TPSE":
                        writer.add_scalar('batch/loss_se', loss_tp__.item(), global_step)
                    else:  # tpcw
                        writer.add_scalar('batch/loss_cw', loss_tp__.item(), global_step)
                        print(f"batch/cw[{global_step}]: {loss_tp__.item()}")
                writer.add_scalar('batch/loss_ff', loss_ff.item(), global_step)
                # writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr(), global_step)

            if global_step % args.eval_term == 0:
                model.eval()  #
                val_loss = evaluate(model, valid_loader, criterion, writer, global_step, DEVICE=DEVICE)
                model.train()

            if global_step % args.save_term == 0:
                try:
                    val_loss
                except Exception:
                    val_loss = -1
                save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir)  # save best 5 models
            global_step += 1

        if args.log_mode:
            # Summary
            avg_loss_mel = epoch_loss_mel / (len(data_loader))
            avg_loss_fmel = epoch_loss_fmel / (len(data_loader))
            avg_loss_ff = epoch_loss_ff / (len(data_loader))

            writer.add_scalar('train/loss_mel', avg_loss_mel, global_step)
            writer.add_scalar('train/loss_fmel', avg_loss_fmel, global_step)
            writer.add_scalar('train/loss_ff', avg_loss_ff, global_step)
            # writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr(), global_step)

            alignment = A[0:1].clone().cpu().detach().numpy()
            writer.add_image('train/alignments', att2img(alignment), global_step)  # (Tx, Ty)
            text = texts[0].cpu().detach().numpy()
            text = [idx2char[ch] for ch in text]
            plot_att(alignment[0], text, global_step,
                     path=os.path.join(args.logdir, type(model).__name__, 'A', 'train'))

            mel_hat = mels_hat[0:1].transpose(1, 2)
            fmel_hat = fmels_hat[0:1].transpose(1, 2)
            mel = mels[0:1].transpose(1, 2)
            writer.add_image('train/mel_hat', mel_hat, global_step)
            writer.add_image('train/fmel_hat', fmel_hat, global_step)
            writer.add_image('train/mel', mel, global_step)

            if type(model).__name__ == 'TPGST':
                styleA = style_attentions.unsqueeze(0) * 255.
                writer.add_image('train/styleA', styleA, global_step)
            # print('Training Loss: {}'.format(avg_loss))
        epochs += 1
        print(f"Epoch {epochs} - Step {global_step}: epoch_loss_mel={epoch_loss_mel} epoch_loss_fmel={epoch_loss_fmel} epoch_loss_ff={epoch_loss_ff}")

    print('Training complete')


def evaluate(model, data_loader, criterion, writer, global_step, DEVICE=None):
    """
    To evaluate with validation set

    :param model: nn module object
    :param data_loader: data loader
    :param criterion: criterion for spectorgrams
    :param writer: Tensorboard writer
    :param global_step: Scalar. global step
    :param DEVICE: 'cpu' or 'gpu'

    """
    bce_loss = nn.BCELoss()
    xe_loss = nn.CrossEntropyLoss()
    valid_loss_mel, valid_loss_fmel, valid_loss_ff, valid_loss_tp__ = 0., 0., 0., 0.
    A = None
    with torch.no_grad():
        for step, (texts, mels, ff) in enumerate(data_loader):
            texts, mels, ff = texts.to(DEVICE), mels.to(DEVICE), ff.to(DEVICE)
            GO_frames = torch.zeros([mels.shape[0], 1, args.n_mels * args.r]).to(DEVICE)  # (N, Ty/r, n_mels)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)
            refs = mels.view(mels.size(0), -1, args.n_mels).unsqueeze(1)  # (N, 1, Ty, n_mels)
            if type(model).__name__ == 'TPGST':
                if model.type == "TPSE":
                    mels_hat, fmels_hat, A, style_attentions, ff_hat, se, tpse = model(texts, prev_mels, refs)
                    loss_tp__ = criterion(tpse, se.detach())
                else:  # TPCW
                    mels_hat, fmels_hat, A, style_attentions, ff_hat, se, tpcw = model(texts, prev_mels, refs)
                    sa = style_attentions.detach()
                    if sa.ndim == 1:
                        sa = sa.unsqueeze(0)  # REVIEW style_attentions dim = 1 if batch size = 1
                    # loss_tp__ = xe_loss(tpcw, sa)  # Same size (N, n_tokens)
                    loss_tp__ = xe_loss(tpcw.view(-1), sa.view(-1))  # Same size (N, n_tokens)
                    valid_loss_tp__ += loss_tp__.item()
            else:
                mels_hat, fmels_hat, A, ff_hat = model(texts, prev_mels)

            loss_mel = criterion(mels_hat, mels)
            fmels = mels.view(mels.size(0), -1, args.n_mels)
            loss_fmel = criterion(fmels_hat, fmels)
            loss_ff = bce_loss(ff_hat, ff)

            valid_loss_mel += loss_mel.item()
            valid_loss_fmel += loss_fmel.item()
            valid_loss_ff += loss_ff.item()

        avg_loss_mel = valid_loss_mel / (len(data_loader))
        avg_loss_fmel = valid_loss_fmel / (len(data_loader))
        avg_loss_ff = valid_loss_ff / (len(data_loader))

        writer.add_scalar('eval/loss_mel', avg_loss_mel, global_step)
        writer.add_scalar('eval/loss_fmel', avg_loss_fmel, global_step)
        writer.add_scalar('eval/loss_ff', avg_loss_ff, global_step)

        alignment = A[0:1].clone().cpu().detach().numpy()
        writer.add_image('eval/alignments', att2img(alignment), global_step)  # (Tx, Ty)
        text = texts[0].cpu().detach().numpy()
        text = [load_vocab()[-1][ch] for ch in text]
        plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, type(model).__name__, 'A'))

        mel_hat = mels_hat[0:1].transpose(1, 2)
        fmel_hat = fmels_hat[0:1].transpose(1, 2)
        mel = mels[0:1].transpose(1, 2)

        writer.add_image('eval/mel_hat', mel_hat, global_step)
        writer.add_image('eval/fmel_hat', fmel_hat, global_step)
        writer.add_image('eval/mel', mel, global_step)
        if type(model).__name__ == 'TPGST':
            avg_loss_tp__ = valid_loss_tp__ / (len(data_loader))
            writer.add_scalar(
                'eval/loss_se' if model.type == "TPSE" else 'eval/loss_cw',
                avg_loss_tp__, global_step
            )
            styleA = style_attentions.view(1, mels.size(0), args.n_tokens) * 255.
            writer.add_image('eval/styleA', styleA, global_step)

    return avg_loss_mel


def save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir):
    """
    To save best models

    :param model: nn module object
    :param model_infos: top 5 models which have best losses [('step', loss)]*5
    :param optimizer: optimizer
    :param scheduler: for learning rate update
    :param val_loss: Scalar. validation loss
    :param global_step: Scalar.
    :param ckpt_dir: String. checkpoint directory

    Returns:
        model_infos: top 5 models

    """
    cur_ckpt = 'model-{:03d}k.pth.tar'.format(global_step // 1000)
    state = {
        'global_step': global_step,
        'name': type(model).__name__,
        'model': model.state_dict(),
        'loss': val_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, cur_ckpt))


def main(DEVICE, model_type):
    """
    main function

    :param DEVICE: 'cpu' or 'gpu'

    """
    model = TPGST(type=model_type).to(DEVICE)

    print('Model {} is working...'.format(type(model).__name__))
    ckpt_dir = os.path.join(args.logdir, type(model).__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_policy)

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
    else:
        print('Already exists. Retrain the model.')
        try:
            model_path = sorted(glob.glob(os.path.join(ckpt_dir, 'model-*.tar')))[-1]  # latest model
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state['model'])
            args.global_step = state['global_step']
            optimizer.load_state_dict(state['optimizer'])
            scheduler.last_epoch = state['scheduler']['last_epoch']
            scheduler.base_lrs = state['scheduler']['base_lrs']
        except Exception as e:
            print(e)
            print("Model file not exists. Train from start")

    dataset = VLSPSpeechDataset(args.data_path, args.meta, mem_mode=args.mem_mode, training=True)
    validset = VLSPSpeechDataset(args.data_path, args.meta, mem_mode=args.mem_mode, training=False)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn,
                             drop_last=True, pin_memory=True,
                             # num_workers=args.n_workers
    )
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=collate_fn, pin_memory=True)
    # torch.set_num_threads(4)
    print('{} threads are used...'.format(torch.get_num_threads()))

    writer = SummaryWriter(ckpt_dir)
    train(model, data_loader, valid_loader, optimizer, scheduler,
          batch_size=args.batch_size, ckpt_dir=ckpt_dir, writer=writer, DEVICE=DEVICE)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument(
        "--gpu_id",
        default=0
    )
    parser.add_argument(
        "--seed",
        default=999
    )
    parser.add_argument(
        "--model_type",
        default="TPSE"
    )
    _args = parser.parse_args()

    if _args.device=="cuda":
        gpu_id = int(_args.gpu_id)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = _args.device
    seed = _args.seed # Set random seem for reproducibility
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main(DEVICE, _args.model_type)
