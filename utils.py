import os
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader

from dataset import MotionDataset
from model import MotionModel


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


def get_run_info(run_dir):
    label = os.path.basename(run_dir)
    best_model = os.path.join(run_dir, 'model_best.pth.tar')
    params = os.path.join(run_dir, 'params.csv')
    params = pd.read_csv(params).to_dict(orient='records')[0]
    metrics = get_metrics(run_dir)
    return run_dir, metrics, label, best_model, params


def load_run(run, data=None, **data_kwargs):
    run_info = get_run_info(run)
    run_dir, _, _, best_model, params = run_info

    if data is None:
        data = params['val_data']

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


def get_run_summary(info, **kwargs):
    run_dir, metrics, label, best_model, params = info

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
    return pd.DataFrame(params, index=[0])
