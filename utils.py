import glob2
import os
import numpy as np
import pandas as pd
import re
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MotionDataset
from model import MotionModel


def find_runs(base_dir):
    logs = glob2.glob(os.path.join(base_dir, '**/log.txt'))
    runs = [os.path.dirname(l) for l in logs]
    return runs


def get_metrics(run):
    regexp = r'((\w+)=\s*(\d+\.?\d+))'
    log = os.path.join(run, "log.txt")
    metrics = dict()
    with open(log, 'r') as f:
        for line in f:
            if line.startswith('Eval'):
                for match in re.findall(regexp, line):
                    _, key, value = match
                    value = float(value)
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)

    return metrics


def get_annot_rate(run_dir, train=False, stream=False):
    targets, predictions, annot_time = get_predictions(run_dir, train=train, stream=stream)
    if stream:
        n_samples = targets.shape[0]
    else:
        n_samples = sum(t.shape[0] for t in targets)

    frame_rate = annot_time / n_samples
    return frame_rate


def get_run_info(run_dir):
    label = os.path.basename(run_dir)
    best_model = os.path.join(run_dir, 'model_best.pth.tar')
    params = os.path.join(run_dir, 'params.csv')
    params = pd.read_csv(params).fillna(False).to_dict(orient='records')[0]
    metrics = get_metrics(run_dir)
    return run_dir, metrics, label, best_model, params


def load_run(run, data=None, **data_kwargs):
    run_info = get_run_info(run)
    run_dir, _, _, best_model, params = run_info

    if data is None:
        if params['mapper']:
            data_kwargs['mapper'] = params['mapper']

        train_dataset = MotionDataset(params['train_data'], fps=params.get('fps', 120), **data_kwargs)
        val_dataset = MotionDataset(params['val_data'], fps=params.get('fps', 120), **data_kwargs)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        loader = (train_loader, val_loader)
        in_size, out_size = val_dataset.get_data_size()

    else:
        dataset = MotionDataset(data, fps=params.get('fps', 120), **data_kwargs)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        in_size, out_size = dataset.get_data_size()

    model = MotionModel(in_size, out_size,
                        hidden=params.get('hd', 512),
                        dropout=params.get('dropout', 0),
                        bidirectional=params.get('bidirectional', False),
                        stack=params.get('stack', 1),
                        layers=params.get('layers', 2),
                        embed=params.get('embed', 64)
                        )
    if params['cuda']:
        model.cuda()
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return run_info, model, loader


def get_predictions(run, train=False, stream=False, force=False):

    # Compute and cache predictions
    base_file = 'predictions_stream.npz' if stream else 'predictions.npz'
    predictions_file = '{}_{}'.format('train' if train else 'val', base_file)
    predictions_file = os.path.join(run, predictions_file)
    annot_time = None
    if os.path.exists(predictions_file) and not force:
        print('Loading cached predictions:', predictions_file)
        a = np.load(predictions_file)
        targets = a['targets']
        predictions = a['predictions']
        annot_time = a['annot_time'] if 'annot_time' in a else None
        del a
    else:
        run_info, model, loaders = load_run(run)
        params = run_info[-1]
        loader = loaders[0] if train else loaders[1]

        targets = []
        predictions = []

        if stream:  # unique stream processing
            inputs = []
            for i, (x, y) in enumerate(tqdm(loader)):
                inputs.append(x)
                y = y.numpy().squeeze()
                targets.append(y)

            start = time.time()
            inputs = torch.cat(inputs, 1)
            if params['cuda']:
                inputs = inputs.cuda()

            inputs = Variable(inputs, volatile=True)
            logits = model.segment(inputs)
            predictions = torch.nn.functional.sigmoid(logits)
            end = time.time()

            targets = np.concatenate(targets, axis=0)
            predictions = predictions.data.cpu().numpy().squeeze()

        else:  # sequence-based processing
            start = time.time()
            for i, (x, y) in enumerate(tqdm(loader)):
                y = y.numpy().squeeze()
                targets.append(y)
                if params['cuda']:
                    x = x.cuda()

                x = Variable(x, volatile=True)

                logits = model.segment(x)
                y_hat = torch.nn.functional.sigmoid(logits)

                y_hat = y_hat.data.cpu().numpy().squeeze()  # [:, y]
                predictions.append(y_hat)
            end = time.time()

        annot_time = end - start
        np.savez_compressed(predictions_file, targets=targets, predictions=predictions, annot_time=annot_time)

    return targets, predictions, annot_time


def get_run_summary(info, epoch='all', aggr=max, **kwargs):
    run_dir, metrics, label, best_model, params = info

    if epoch == 'all':
        best_loss = min(metrics['Loss'])
        params['best_loss'] = best_loss

        for m, v in metrics.items():
            if m == 'Loss':
                continue

            best_value, epoch = max((a, i) for i, a in enumerate(v))
            params.update({
                'best_{}'.format(m): best_value,
                'best_{}_epoch'.format(m): epoch,
            })

        params.update(kwargs)
        params = pd.DataFrame(params, index=[0])

    elif epoch in metrics:
        _, best_epoch = aggr((a, i) for i, a in enumerate(metrics[epoch]))

        for m, v in metrics.items():
            params['best_{}'.format(m)] = v[best_epoch]

        params = pd.DataFrame(params, index=[0])

    elif epoch == 'test':
        metrics_file = os.path.join(run_dir, 'metrics.csv')
        assert os.path.exists(metrics_file)
        metrics = pd.read_csv(metrics_file)

        params = pd.DataFrame(params, index=[0])
        params = (
            params.assign(key=1)
                  .merge(metrics.assign(key=1), on="key")
                  .drop("key", axis=1)
        )

    return params
