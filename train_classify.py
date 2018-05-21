import argparse
import glob

import os
import pandas as pd
import re
import shutil
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from tqdm import trange, tqdm

from dataset import MotionDataset
from model import MotionModel


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.data if isinstance(output, Variable) else output

    vals, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k.mul_(100.0 / batch_size)
        res.append(correct_k[0])
    return res


def compute_loss(output, target, args):

    if args.head == 'softmax' and args.label_smoothing == 0:
        return F.cross_entropy(output, target)

    n_classes = output.shape[1]

    # build one-hot vector
    target = target.data.cpu()[0]
    y = torch.zeros(1, n_classes)
    y[0, target] = 1

    if args.cuda:
        y = y.cuda()
    y = Variable(y, requires_grad=False)

    if args.label_smoothing:
        y = y * (1 - args.label_smoothing) + \
            (1 - y) * args.label_smoothing / (n_classes - 1)

    if args.head == 'softmax':
        return torch.sum(- y * F.log_softmax(output, dim=1), 1).mean()

    if args.head == 'sigmoid':
        return F.binary_cross_entropy_with_logits(output, y)


def evaluate(loader, model, args):
    model.eval()

    avg_loss = 0.0
    avg_acc1 = 0.0
    avg_acc5 = 0.0

    n_classes = len(loader.dataset.actions)
    action_correct = torch.zeros(n_classes)
    action_count = torch.zeros(n_classes)

    progress_bar = tqdm(loader, disable=args.no_progress)
    for i, (x, y) in enumerate(progress_bar):
        _y = y
        if args.cuda:
            x = x.cuda()
            y = y.cuda(async=True)

        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)

        y_hat = model(x)

        loss = compute_loss(y_hat, y, args)
        avg_loss += loss.data[0]

        acc1, acc5 = accuracy(y_hat.cpu(), _y, topk=(1, 5))
        action_correct[_y] += (acc1 / 100.0) + (acc5 / 100.0)
        action_count[_y] += 1

        avg_acc1 += acc1
        avg_acc5 += acc5

        run_loss = avg_loss / (i + 1)
        run_acc1 = avg_acc1 / (i + 1)
        run_acc5 = avg_acc5 / (i + 1)
        progress_bar.set_postfix({
            'loss': '{:6.4f}'.format(run_loss),
            'acc1': '{:5.2f}%'.format(run_acc1),
            'acc5': '{:5.2f}%'.format(run_acc5),
        })

    accuracy_balance = torch.log1p(2 * action_count - action_correct)

    return (run_loss, run_acc1, run_acc5), accuracy_balance


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

        y_hat = model(x)

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
    last_checkpoint = os.path.join(run_dir, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint):
        return last_checkpoint

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
        best_acc = checkpoint['best_accuracy']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_acc = 0
        start_epoch = 1
        # Create the run directory and log file
        train_fname = os.path.splitext(os.path.basename(args.train_data))[0]
        val_fname = os.path.splitext(os.path.basename(args.val_data))[0]

        parameters = vars(args)
        parameters.update(dict(train=train_fname, val=val_fname))

        run_name = 'model_tr-{0[train]}_vl-{0[val]}_' \
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
                   'f{0[fps]}_' \
                   'o-{0[offset]}_' \
                   'opt-{0[optim]}_' \
                   'ls{0[label_smoothing]}_' \
                   'bal-{0[balance]}'.format(parameters)

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
        progress_bar.set_description('TRAIN')
        train(train_loader, model, optimizer, epoch, args)

        progress_bar.set_description('EVAL')
        metrics, accuracy_balance = evaluate(val_loader, model, args)
        print('Eval Epoch {}: Loss={:6.4f} Acc@1={:5.2f} Acc@5={:5.2f}'.format(epoch, *metrics),
              file=args.log, flush=True)

        current_acc1 = metrics[1]

        is_best = current_acc1 > best_acc
        best_acc = max(best_acc, current_acc1)

        # SAVE MODEL
        if args.keep:
            fname = 'epoch_{:02d}.pth'.format(epoch)
        else:
            fname = 'last_checkpoint.pth'

        fname = os.path.join(run_dir, fname)
        save_checkpoint({
            'epoch': epoch,
            'best_accuracy': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)

        if args.balance == 'adaptive':
            # print(accuracy_balance)
            weights = train_dataset.get_weights(accuracy_balance)
            sampler = WeightedRandomSampler(weights, len(weights))
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=1, pin_memory=args.cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on motion data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # DATA PARAMS
    parser.add_argument('train_data', help='path to train data file (Pickle file)')
    parser.add_argument('val_data', help='path to val data file (Pickle file)')
    parser.add_argument('--mapper', help='class mapper csv file')
    parser.add_argument('-f', '--fps', type=int, default=10, help='resampling FPS')
    parser.add_argument('-o', '--offset', choices=['none', 'random'], default='random', help='offset mode when resampling training data')
    parser.add_argument('--balance', choices=['none', 'frequency', 'adaptive'], default='none', help='how to sample during training')
    parser.add_argument('--ls', '--label-smoothing', type=float, dest='label_smoothing', default=0.1, help='smooth one-hot labels by this factor')

    # NETWORK PARAMS
    parser.add_argument('--emb', '--embed', dest='embed', type=int, default=48, help='sequence embedding dimensionality (0 for none)')
    parser.add_argument('-b', '--bidirectional', action='store_true', dest='bidirectional', help='use bidirectional LSTM')
    parser.add_argument('-u', '--unidirectional', action='store_false', dest='bidirectional', help='use unidirectional LSTM')
    parser.add_argument('--hd', '--hidden-dim', type=int, default=1024, help='LSTM hidden state dimension')
    parser.add_argument('-s', '--stack', type=int, default=1, help='how many LSTMs to stack')
    parser.add_argument('-l', '--layers', type=int, default=1, help='how many layers for fully connected classifier')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='dropout applied on hidden state')
    parser.add_argument('--head', choices=['softmax', 'sigmoid'], default='softmax', help='networks head')

    # OPTIMIZER PARAMS
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam', help='optimizer')
    # parser.add_argument('-m','--momentum', type=float, default=0.9, help='momentum (only for SGD)')
    parser.add_argument('-a', '--accumulate', type=int, default=40, help='batch accumulation')
    parser.add_argument('-c', '--clip-norm', type=float, default=0.0, help='max gradient norm (0 for no clipping)')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of training epochs')
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
