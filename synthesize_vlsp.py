import os
import torch
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import glob

from model import TPGST
from data_vlsp import text_collate_fn, load_vocab, VLSPTextDataset
from config import ConfigArgs as args


DEVICE = None


def synthesize(model, data_loader, batch_size=100):
    """
    To synthesize with text samples

    :param model: nn module object
    :param data_loader: data loader
    :param batch_size: Scalar

    """
    idx2char = load_vocab()[-1]
    with torch.no_grad():
        print('*'*15, ' Synthesize ', '*'*15)
        for step, (texts, _, _) in enumerate(data_loader):
            texts = texts.to(DEVICE)
            GO_frames = torch.zeros([texts.shape[0], 1, args.n_mels*args.r]).to(DEVICE)
            mels_hat, mags_hat, A, _, _, se, _ = model(texts, GO_frames, synth=True)
            mels_hat = mels_hat.cpu().numpy()
            # alignments = A.cpu().detach().numpy()
            # visual_texts = texts.cpu().detach().numpy()
            # mags = mags_hat.cpu().detach().numpy() # mag: (N, Ty, n_mags)
            print('='*10, ' Vocoder ', '='*10)
            for idx in range(len(texts)):
                np.save(os.path.join(args.sampledir, 'mel-{:04d}.npy'.format(idx+step*batch_size+1)), mels_hat[idx])
                # text = [idx2char[ch] for ch in visual_texts[idx]]
                # utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{:03d}.png'.format(idx+step*batch_size+1))
    return None

def main(load_model='latest', synth_mode='synthesize'):
    """
    main function

    :param load_model: String. {best, latest, <model_path>}
    :param synth_mode: {'test', 'synthesize'}

    """
    assert os.path.exists(args.testset), f'Test sentence path is wrong: {args.testset}'

    model = TPGST().to(DEVICE)

    testset = VLSPTextDataset(args.testset)

    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                            shuffle=False, collate_fn=text_collate_fn, pin_memory=True)

    if load_model.lower() == 'best':
        ckpt = pd.read_csv(os.path.join(args.logdir, model.name, 'ckpt.csv'), sep=',', header=None)
        ckpt.columns = ['models', 'loss']
        model_path = ckpt.sort_values(by='loss', ascending=True).models.loc[0]
        model_path = os.path.join(args.logdir, model.name, model_path)
    elif 'pth.tar' in load_model:
        model_path = load_model
    else:
        model_path = sorted(glob.glob(os.path.join(args.logdir, model.name, 'model-*.tar')))[-1] # latest model
    state = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state['model'])
    args.global_step = state['global_step']

    print('The model is loaded. Step: {}'.format(args.global_step))

    model.eval()

    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))

    if synth_mode == 'test':
        ref_synthesize(model, test_loader, args.test_batch)
    elif synth_mode == 'style':
        style_synthesize(model, test_loader, args.test_batch)
    elif synth_mode == 'tp':
        tp_synthesize(model, test_loader, args.test_batch)
    elif synth_mode == 'fix':
        fixed_synthesize(model, test_loader, args.test_batch)
    else:
        synthesize(model, test_loader, args.test_batch)

if __name__ == '__main__':
    import argparse
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
        "--load_model",
        default=""
    )
    parser.add_argument(
        "--synth_mode",
        default="synthesize"
    )
    _args = parser.parse_args()

    if _args.device=="cuda":
        gpu_id = int(_args.gpu_id)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = _args.device
    # print(DEVICE)
    # torch.device(DEVICE)
    load_model = _args.load_model
    synth_mode = _args.synth_mode

    # gpu_id = int(sys.argv[1])
    # load_model = sys.argv[2] if len(sys.argv) > 2 else None
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(load_model=load_model, synth_mode=synth_mode)
