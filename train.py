import argparse
import pickle
from collections import Counter

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm

from model import MotionModel


class MotionDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as in_file:
            self.data = pickle.load(in_file)

        self.actions = Counter(s['action_id'] for s in self.data)

        unique_actions = sorted(self.actions.keys())
        self.action_id_to_ix = {a: i for i, a in enumerate(unique_actions)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x = torch.from_numpy(sample['data'])
        y = self.action_id_to_ix[sample['action_id']]
        return x, y

    def get_weights(self):
        n_samples = float(len(self.data))
        weights = [n_samples / self.actions[s['action_id']] for s in self.data]
        return weights

    def get_data_size(self):
        in_size = self.data[0]['data'][0].size
        out_size = len(self.actions)
        return in_size, out_size


def accuracy(output, target, topk=(1,)):
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


def train(loader, model, optimizer, args):
    model.train()
    optimizer.zero_grad()

    avg_loss = 0.0
    avg_acc1 = 0.0
    avg_acc5 = 0.0

    progress_bar = tqdm(loader)
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

        acc1, acc5 = accuracy(y_hat, y, topk=(1, 5))

        avg_acc1 += acc1
        avg_acc5 += acc5

        if (i+1) % args.accumulate == 0:
            optimizer.step()
            avg_loss /= args.accumulate
            avg_acc1 /= args.accumulate
            avg_acc5 /= args.accumulate
            progress_bar.set_postfix({
                'loss': '{:6.4f}'.format(avg_loss),
                'acc1': '{:5.2f}%'.format(avg_acc1),
                'acc5': '{:5.2f}%'.format(avg_acc5),
            })
            optimizer.zero_grad()
            avg_loss = 0
            avg_acc1 = 0
            avg_acc5 = 0


def main(args):
    # Use CUDA?
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    dataset = MotionDataset(args.data)
    in_size, out_size = dataset.get_data_size()
    weights = dataset.get_weights()

    sampler = WeightedRandomSampler(weights, len(weights))
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1, pin_memory=args.cuda)

    model = MotionModel(in_size, 128, out_size)  # use 128-dimension hidden state
    if args.cuda:
        model.cuda()

    optimizer = Adam(model.parameters())

    for epoch in trange(args.epochs):
        train(loader, model, optimizer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on motion data')
    parser.add_argument('data', help='path to data file (Pickle file)')
    parser.add_argument('-e', '--epochs', default=10, help='number of training epochs')
    parser.add_argument('-a', '--accumulate', default=10, help='batch accumulation')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA acceleration')
    args = parser.parse_args()
    main(args)
