import os
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader

from dataset import MotionDataset
from model import MotionModel


def get_series(run):
    loss_series = []
    accuracy_series = []
    regexp = r'.*Loss=(\s*\d+\.\d+).*Acc@1=(\s*\d+\.\d+).*'
    log = os.path.join(run, "log.txt")
    with open(log, 'r') as f:
        for line in f:
            if line.startswith('Eval'):
                matches = re.match(regexp, line)
                loss, accuracy = matches.groups()
                loss_series.append(float(loss))
                accuracy_series.append(float(accuracy))
    return loss_series, accuracy_series


def get_run_info(run_dir):
    label = os.path.basename(run_dir).replace(r'model_tr-split1_vl-split2_', '')
    best_model = os.path.join(run_dir, 'model_best.pth.tar')
    params = os.path.join(run_dir, 'params.csv')
    params = pd.read_csv(params).to_dict(orient='records')[0]
    loss, accuracy = get_series(run_dir)
    return run_dir, loss, accuracy, label, best_model, params


def load_run(run, data=None, data_offset='none'):
    run_info = get_run_info(run)
    run_dir, _, _, _, best_model, params = run_info

    if data is None:
        data = params['val_data']

    dataset = MotionDataset(data, fps=params.get('fps', 120), offset=data_offset)
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
    run_dir, loss, accuracy, label, best_model, params = info
    best_loss = min(loss)
    best_acc, epoch = max((a, i) for i, a in enumerate(accuracy))
    params.update({
        'best_loss': best_loss,
        'best_acc': best_acc,
        'best_epoch': epoch,
    })
    params.update(kwargs)
    return pd.DataFrame(params, index=[0])