import argparse
import os
import numpy as np
import pandas as pd
import re
import glob2 as glob2
import torch.nn.functional as F

import matplotlib
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import MotionDataset
from model import MotionModel

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
# sns.set_context('paper')


def get_series(log):
    loss_series = []
    accuracy_series = []
    regexp = r'.*Loss=(\s*\d+\.\d+).*Acc@1=(\s*\d+\.\d+).*'
    with open(log, 'r') as f:
        for line in f:
            if line.startswith('Eval'):
                matches = re.match(regexp, line)
                loss, accuracy = matches.groups()
                loss_series.append(float(loss))
                accuracy_series.append(float(accuracy))
    return loss_series, accuracy_series


def get_run_info(log):
    run_dir = os.path.dirname(log)
    label = os.path.basename(run_dir).replace(r'model_tr-split1_vl-split2_', '')
    best_model = os.path.join(run_dir, 'model_best.pth.tar')
    params = os.path.join(run_dir, 'params.csv')
    params = pd.read_csv(params).to_dict(orient='records')[0]
    loss, accuracy = get_series(log)
    return run_dir, loss, accuracy, label, best_model, params


def train_plot(logs):
    run_infos = [get_run_info(log) for log in logs]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Evaluation Loss')
    ax2.set_title('Evaluation Accuracy')
    for run_dir, loss, accuracy, label, best_model, params in run_infos:
        ax1.plot(loss, label=label)
        ax2.plot(accuracy, label=label)
    ax2.set_ylim([0, 100])
    plt.legend(loc='best', prop={'size': 6})
    plt.tight_layout()
    plt.savefig('train_progress.pdf')


def load_run(log, data=None):
    run_info = get_run_info(log)
    run_dir, _, _, _, best_model, params = run_info

    if data is None:
        data = params['val_data']

    dataset = MotionDataset(data, fps=params.get('fps', 120))
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


def predict(model, loader, cuda=False):
    predictions = []
    targets = []
    confidences = []

    for x, y in tqdm(loader):
        if cuda:
            x = x.cuda()
            y = y.cuda(async=True)

        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)

        logits = model(x)
        confidence = F.softmax(logits, dim=1)
        _, y_hat = torch.max(logits, 1)

        if cuda:
            y = y.cpu()
            y_hat = y_hat.cpu()
            confidence = confidence.cpu()

        targets.append(y.data[0])
        prediction = y_hat.data[0]
        predictions.append(prediction)
        confidences.append(confidence.data.numpy())

    return predictions, targets, confidences


def confusion_plot(logs):
    for log in logs:
        run_info, model, loader = load_run(log, data=args.data)
        run_dir, _, _, label, _, params = run_info
        dataset = loader.dataset

        predictions, targets, _ = predict(model, loader, cuda=params['cuda'])
        overall_accuracy = accuracy_score(targets, predictions)
        confusion = confusion_matrix(targets, predictions)
        mask = confusion == 0
        # Normalize it
        # confusion = confusion.astype('float') / confusion.sum(axis=1)[:, None]
        # fig, ax = plt.subplots()
        # im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        # fig.colorbar(im)

        plt.figure(figsize=(30, 30))
        plt.title('{}: (Overall Accuracy: {:4.2%}'.format(label, overall_accuracy))
        ax = sns.heatmap(confusion, annot=True, fmt='d', mask=mask, cbar=False)
        classes = dataset.action_descriptions
        tick_marks = np.arange(len(classes))
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_ticks(tick_marks + 0.5, minor=True)
            axis.set(ticks=tick_marks, ticklabels=classes)

        labels = ax.get_xticklabels()
        for label in labels:
            label.set_rotation(90)
        plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.grid(True, which='minor')

        plot_fname = os.path.join(run_dir, 'confusion.pdf')
        plt.savefig(plot_fname, bbox_inches='tight')
        plt.close()

        del model, loader, predictions, targets


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


def display_status(logs):
    infos = [get_run_info(log) for log in logs]
    summaries = [get_run_summary(i) for i in infos]
    summary = pd.concat(summaries, ignore_index=True).sort_values('best_acc', ascending=False)

    if args.output:
        summary.to_csv(args.output, index=False)
    else:
        with pd.option_context('display.width', None), \
             pd.option_context('max_columns', None):
            print(summary)


def offset_eval(logs):
    summaries = []
    for log in logs:
        run_info, model, loader = load_run(log, data=args.data)
        params = run_info[-1]
        dataset = loader.dataset

        _, targets, confidences = predict(model, loader, cuda=params['cuda'])

        n_samples = len(dataset) // dataset.skip
        targets = targets[:n_samples]
        confidences = np.concatenate(confidences, axis=0)
        confidences = confidences.reshape(dataset.skip, n_samples, -1).mean(axis=0)
        predictions = np.argmax(confidences, axis=1)
        multi_offset_accuracy = accuracy_score(targets, predictions)
        summary = get_run_summary(run_info, multi_offset_acc=multi_offset_accuracy)
        summaries.append(summary)

    summary = pd.concat(summaries, ignore_index=True).sort_values('multi_offset_acc', ascending=False)
    if args.output:
        summary.to_csv(args.output, index=False)
    else:
        with pd.option_context('display.width', None), \
             pd.option_context('max_columns', None):
            print(summary)


def main(args):
    logs = glob2.glob(os.path.join(args.run_dir, '**/log.txt'))

    if args.type == 'train':
        train_plot(logs)

    if args.type == 'confusion':
        confusion_plot(logs)

    if args.type == 'status':
        display_status(logs)

    if args.type == 'multi-eval':
        offset_eval(logs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('type', choices=['train', 'confusion', 'status', 'multi-eval'], help='what to plot')
    parser.add_argument('run_dir', nargs='?', default='runs/', help='folder in which logs are searched')
    parser.add_argument('-d', '--data', help='eval data (for confusion)')
    parser.add_argument('-o', '--output', help='outfile (for status)')
    args = parser.parse_args()
    main(args)
