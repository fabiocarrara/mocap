import argparse
import glob

import os
import pandas as pd
import re
import shutil
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from tqdm import trange, tqdm

from dataset import MotionDataset
from model import MotionModel


def compute_loss(output, target, args):
    n_classes = output.shape[1]

    if args.label_smoothing:
        target = target * (1 - args.label_smoothing) + \
                 (1 - target) * args.label_smoothing / (n_classes - 1)

    return F.binary_cross_entropy_with_logits(output, target)


def f1_score(targets, predictions, multiple_thr=False):
    if multiple_thr:
        category_f1s = []
        for i in range(targets.shape[1]):
            p, r, _ = precision_recall_curve(targets[:, i], predictions[:, i])
            f1 = 2 * (p * r) / (p + r)
            category_f1s.append(max(f1))

        category_f1s = torch.FloatTensor(category_f1s)
        support = targets.sum(dim=0)
        microF1 = (category_f1s * support).sum() / support.sum()
        macroF1 = category_f1s.mean()
        return microF1, macroF1

    else:
        p, r, _ = precision_recall_curve(targets.view(-1), predictions.view(-1))
        f1 = 2 * (p * r) / (p + r)
        return max(f1)


def evaluate(loader, model, args):
    model.eval()

    avg_loss = 0.0

    targets = []
    predictions = []
    progress_bar = tqdm(loader, disable=args.no_progress)
    for i, (x, y) in enumerate(progress_bar):
        if args.cuda:
            x = x.cuda()
            y = y.cuda(async=True)

        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)

        y_hat = model.segment(x)

        y = y.squeeze()
        y_hat = y_hat.squeeze()

        loss = compute_loss(y_hat, y, args)
        avg_loss += loss.data[0]

        y = y.data.cpu()
        y_hat = y_hat.data.cpu()

        targets.append(y)
        predictions.append(y_hat)

        run_loss = avg_loss / (i + 1)
        progress_bar.set_postfix({
            'loss': '{:6.4f}'.format(run_loss),
            # 'mAP': '{:4.3f}'.format(run_ap)
        })

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    micro_ap = average_precision_score(targets, predictions, average='micro')
    macro_ap = average_precision_score(targets, predictions, average='macro')

    single_thr_f1 = f1_score(targets, predictions)
    multi_thr_micro_f1, multi_thr_macro_f1 = f1_score(targets, predictions, multiple_thr=True)

    return run_loss, micro_ap, macro_ap, single_thr_f1, multi_thr_micro_f1, multi_thr_macro_f1


def train(loader, model, optimizer, epoch, args):
    model.train()
    optimizer.zero_grad()

    avg_loss = 0.0
    n_samples = len(loader.dataset)
    progress_bar = tqdm(loader, disable=args.no_progress)
    for i, (x, y) in enumerate(progress_bar):
        if args.cuda:
            x = x.cuda()
            y = y.cuda(async=True)

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)

        y_hat = model.segment(x)

        y = y.squeeze()
        y_hat = y_hat.squeeze()

        loss = compute_loss(y_hat, y, args)
        loss.backward()

        avg_loss += loss.data[0]

        if (i + 1) % args.accumulate == 0 or (i + 1) == n_samples:
            if args.clip_norm:
                clip_grad_norm(model.parameters(), args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            avg_loss /= args.accumulate

            progress_bar.set_postfix({
                'loss': '{:6.4f}'.format(avg_loss),
            })

            print('Train Epoch {} [{}/{}]: Loss = {:6.4f}'.format(
                epoch, i + 1, n_samples, avg_loss), file=args.log, flush=True)

            avg_loss = 0


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        base_dir = os.path.dirname(filename)
        best_filename = os.path.join(base_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def get_last_checkpoint(run_dir):
    def get_epoch(fname):
        epoch_regex = r'.*epoch_(\d+).pth'
        matches = re.match(epoch_regex, fname)
        return int(matches.groups()[0]) if matches else None

    checkpoints = glob.glob(os.path.join(run_dir, 'epoch_*.pth'))
    checkpoints = [(get_epoch(i), i) for i in checkpoints]
    last_checkpoint = max(checkpoints)[1]
    return last_checkpoint


def main(args):
    # Use CUDA?
    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load datasets and build data loaders
    val_dataset = MotionDataset(args.val_data, fps=args.fps, mapper=args.mapper)
    val_actions = val_dataset.actions.keys()

    train_dataset = MotionDataset(args.train_data, keep_actions=val_actions, fps=args.fps, offset=args.offset, mapper=args.mapper)
    train_actions = train_dataset.actions.keys()

    # with open('a.txt', 'w') as f1, open('b.txt', 'w') as f2:
    #     f1.write('\n'.join(map(str, train_dataset.actions.keys())))
    #     f2.write('\n'.join(map(str, val_dataset.actions.keys())))

    assert len(train_actions) == len(val_actions), \
        "Train and val sets should have same number of actions ({} vs {})".format(
            len(train_actions), len(val_actions))

    in_size, out_size = train_dataset.get_data_size()

    if args.balance == 'none':
        sampler = RandomSampler(train_dataset)
    else:
        weights = train_dataset.get_weights()
        sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=1, pin_memory=args.cuda)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=args.cuda)

    # Build the model
    model = MotionModel(in_size, out_size,
                        hidden=args.hd,
                        dropout=args.dropout,
                        bidirectional=args.bidirectional,
                        stack=args.stack,
                        layers=args.layers,
                        embed=args.embed)
    if args.cuda:
        model.cuda()

    # Create the optimizer and start training-eval loop
    if args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Resume training?
    if args.resume:
        run_dir = args.resume
        last_checkpoint = get_last_checkpoint(run_dir)
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_ap = checkpoint['best_micro_ap']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_ap = 0
        start_epoch = 1
        parameters = vars(args)

        train_fname = os.path.splitext(os.path.basename(args.train_data))[0]
        val_fname = os.path.splitext(os.path.basename(args.val_data))[0]

        # Create the run directory and log file
        run_name = 'segment_tr-{1[train]}_vl-{1[val]}_' \
                   'bi{0[bidirectional]}_' \
                   'emb{0[embed]}_' \
                   'h{0[hd]}_' \
                   's{0[stack]}_' \
                   'l{0[layers]}_' \
                   '{0[head]}_' \
                   'a{0[accumulate]}_' \
                   'c{0[clip_norm]}_' \
                   'd{0[dropout]}_' \
                   'lr{0[lr]}_' \
                   'wd{0[wd]}_' \
                   'e{0[epochs]}_' \
                   'f{0[fps]:g}_' \
                   'o-{0[offset]}_' \
                   'opt-{0[optim]}_' \
                   'ls{0[label_smoothing]}_' \
                   'bal-{0[balance]}'.format(parameters, dict(train=train_fname, val=val_fname))

        runs_parent_dir = 'debug' if args.debug else args.run_dir
        run_dir = os.path.join(runs_parent_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        elif not args.debug:
            return

        params = pd.DataFrame(parameters, index=[0])  # an index is mandatory for a single line
        params_fname = os.path.join(run_dir, 'params.csv')
        params.to_csv(params_fname, index=False)

        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            print(params)

    log_file = os.path.join(run_dir, 'log.txt')
    args.log = open(log_file, 'a+')

    progress_bar = trange(start_epoch, args.epochs + 1, initial=start_epoch, disable=args.no_progress)
    for epoch in progress_bar:
        progress_bar.set_description('TRAIN [CurBestAP={:4.3f}]'.format(best_ap))
        train(train_loader, model, optimizer, epoch, args)

        progress_bar.set_description('EVAL [CurBestAP={:4.3f}]'.format(best_ap))
        metrics = evaluate(val_loader, model, args)

        print('Eval Epoch {}: '
              'Loss={:6.4f} '
              'microAP={:4.3f} '
              'macroAP={:4.3f} '
              'F1={:4.3f} '
              'microMultiF1={:4.3f} '
              'macroMultiF1={:4.3f}'.format(epoch, *metrics),
              file=args.log, flush=True)

        current_micro_ap = metrics[1]
        is_best = current_micro_ap > best_ap
        best_ap = max(best_ap, current_micro_ap)

        # SAVE MODEL
        if args.keep:
            fname = 'epoch_{:02d}.pth'.format(epoch)
        else:
            fname = 'last_checkpoint.pth'

        fname = os.path.join(run_dir, fname)
        save_checkpoint({
            'epoch': epoch,
            'best_micro_ap': best_ap,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on motion data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # DATA PARAMS
    parser.add_argument('train_data', help='path to train data file (Pickle file)')
    parser.add_argument('val_data', help='path to val data file (Pickle file)')
    parser.add_argument('--mapper', help='class mapper csv file')
    parser.add_argument('-f', '--fps', type=float, default=120, help='resampling FPS')
    parser.add_argument('-o', '--offset', choices=['none', 'random'], default='random', help='offset mode when resampling training data')
    parser.add_argument('--balance', choices=['none'], default='none', help='how to sample during training')
    parser.add_argument('--ls', '--label-smoothing', type=float, dest='label_smoothing', default=0.0, help='smooth one-hot labels by this factor')

    # NETWORK PARAMS
    parser.add_argument('--emb', '--embed', dest='embed', type=int, default=64, help='sequence embedding dimensionality (0 for none)')
    parser.add_argument('-b', '--bidirectional', action='store_true', dest='bidirectional', help='use bidirectional LSTM')
    parser.add_argument('-u', '--unidirectional', action='store_false', dest='bidirectional', help='use unidirectional LSTM')
    parser.add_argument('--hd', '--hidden-dim', type=int, default=1024, help='LSTM hidden state dimension')
    parser.add_argument('-s', '--stack', type=int, default=1, help='how many LSTMs to stack')
    parser.add_argument('-l', '--layers', type=int, default=1, help='how many layers for fully connected classifier')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='dropout applied on hidden state')
    parser.add_argument('--head', choices=['sigmoid'], default='sigmoid', help='networks head')

    # OPTIMIZER PARAMS
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam', help='optimizer')
    # parser.add_argument('-m','--momentum', type=float, default=0.9, help='momentum (only for SGD)')
    parser.add_argument('-a', '--accumulate', type=int, default=1, help='batch accumulation')
    parser.add_argument('-c', '--clip-norm', type=float, default=10.0, help='max gradient norm (0 for no clipping)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('-r', '--resume', help='run dir to resume training from')

    # MISC PARAMS
    parser.add_argument('--keep', action='store_true', dest='keep', help='keep all checkpoints evaluated during training')
    parser.add_argument('--no-keep', action='store_false', dest='keep', help='keep only last and best checkpoints')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable CUDA acceleration')
    parser.add_argument('--no-progress', action='store_true', help='disable progress bars')
    parser.add_argument('--run-dir', default='runs', help='where to place this run')
    parser.add_argument('--seed', type=int, default=42, help='random seed to reproduce runs')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    parser.set_defaults(bidirectional=True)
    parser.set_defaults(cuda=True)
    parser.set_defaults(debug=False)
    parser.set_defaults(keep=False)
    args = parser.parse_args()
    main(args)
