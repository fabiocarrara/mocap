import argparse
import glob
import re

import os
import shutil
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm

from dataset import MotionDataset
from model import MotionModel


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    vals, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k.mul_(100.0 / batch_size)
        res.append(correct_k.data[0])
    return res


def eval(loader, model, args):
    model.eval()

    avg_loss = 0.0
    avg_acc1 = 0.0
    avg_acc5 = 0.0

    progress_bar = tqdm(loader, disable=args.no_progress)
    for i, (x, y) in enumerate(progress_bar):
        if args.cuda:
            x = x.cuda()
            y = y.cuda(async=True)
        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)

        y_hat = model(x)

        loss = F.cross_entropy(y_hat, y)
        avg_loss += loss.data[0]

        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))

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

    return run_loss, run_acc1, run_acc5


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

        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        avg_loss += loss.data[0]

        if (i + 1) % args.accumulate == 0:
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

    checkpoints = [(get_epoch(i), i) for i in glob.glob('epoch_*.pth')]
    last_checkpoint = max(checkpoints)[1]
    return last_checkpoint


def main(args):
    # Use CUDA?
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    # Load datasets and build data loaders
    val_dataset = MotionDataset(args.val_data)
    val_actions = val_dataset.actions.keys()

    train_dataset = MotionDataset(args.train_data, keep_actions=val_actions)
    train_actions = train_dataset.actions.keys()

    # with open('a.txt', 'w') as f1, open('b.txt', 'w') as f2:
    #     f1.write('\n'.join(map(str, train_dataset.actions.keys())))
    #     f2.write('\n'.join(map(str, val_dataset.actions.keys())))

    assert len(train_actions) == len(val_actions), \
        "Train and val sets should have same number of actions ({} vs {})".format(
            len(train_actions), len(val_actions))

    in_size, out_size = train_dataset.get_data_size()
    weights = train_dataset.get_weights()

    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=1, pin_memory=args.cuda)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=args.cuda)

    # Build the model
    model = MotionModel(in_size, args.hidden_dim, out_size)
    if args.cuda:
        model.cuda()

    # Create the optimizer and start training-eval loop
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

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
        run_name = 'model_tr-{1}_vl-{2}_lr{0[lr]}_a{0[accumulate]}_wd{0[wd]}_c{0[clip_norm]}_e{0[epochs]}'.format(
            vars(args), train_fname, val_fname)
        run_dir = os.path.join('runs/', run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

    log_file = os.path.join(run_dir, 'log.txt')
    args.log = open(log_file, 'a+')

    progress_bar = trange(start_epoch, args.epochs + 1, initial=start_epoch, disable=args.no_progress)
    for epoch in progress_bar:
        progress_bar.set_description('TRAIN')
        train(train_loader, model, optimizer, epoch, args)

        progress_bar.set_description('EVAL')
        metrics = eval(val_loader, model, args)
        print('Eval Epoch {}: Loss={:6.4f} Acc@1={:5.2f} Acc@5={:5.2f}'.format(epoch, *metrics),
              file=args.log, flush=True)

        current_acc1 = metrics[1]

        is_best = current_acc1 > best_acc
        best_acc = max(best_acc, current_acc1)

        # SAVE MODEL
        fname = 'epoch_{:02d}.pth'.format(epoch)
        fname = os.path.join(run_dir, fname)
        save_checkpoint({
            'epoch': epoch,
            'best_accuracy': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on motion data')
    parser.add_argument('train_data', help='path to train data file (Pickle file)')
    parser.add_argument('val_data', help='path to val data file (Pickle file)')
    parser.add_argument('-d', '--hidden-dim', type=int, default=128, help='LSTM hidden state dimension')
    parser.add_argument('-a', '--accumulate', type=int, default=10, help='batch accumulation')
    parser.add_argument('-c', '--clip-norm', type=float, default=0.0, help='max gradient norm (0 for no clipping)')
    parser.add_argument('-e', '--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('-r', '--resume', help='run dir to resume training from')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA acceleration')
    parser.add_argument('--no-progress', action='store_true', help='disable progress bars')
    args = parser.parse_args()
    main(args)
