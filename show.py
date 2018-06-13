import argparse
import os
import numpy as np
import pandas as pd
import glob2 as glob2
import torch.nn.functional as F

import matplotlib
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.autograd import Variable

from tqdm import tqdm

from utils import get_run_info, load_run, get_run_summary, find_runs

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
# sns.set_context('paper')


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def train_plot(runs, s):
    run_infos = [get_run_info(run) for run in runs]
    metrics = run_infos[0][1].keys()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(5, 3*n_metrics))
    for metric, ax in zip(metrics, axes):
        ax.set_title('Evaluation {}'.format(metric))

    # get col index of non unique columns (params that changes between runs)
    all_params = pd.DataFrame([p[-1] for p in run_infos])
    non_unique_params = all_params.apply(pd.Series.nunique) != 1

    for run_dir, metrics_vals, label, best_model, params in run_infos:
        relevant_params = pd.DataFrame(params, index=[0]).loc[:, non_unique_params]
        cols = relevant_params.columns
        vals = map(str, relevant_params.iloc[0])
        label = '_'.join(map(''.join, zip(cols, vals)))
        # label = label.replace(r'model_tr-.*_vl-.*_bi', 'bi')
        for m, ax in zip(metrics, axes):
            if m in metrics_vals:
                ax.set_title(m)
                ax.plot(smooth(metrics_vals[m], s), label=label)

    plt.legend(loc='best', prop={'size': 6})
    plt.tight_layout()
    plt.savefig('train_progress.pdf')


def predict(model, loader, cuda=False):
    targets = []
    confidences = []
    predictions = []

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


def confusion_plot(runs):
    for run in runs:
        run_info, model, loader = load_run(run, data=args.data)
        run_dir, _, label, _, params = run_info
        loader = loader[1]
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


def display_status(runs):
    infos = [get_run_info(r) for r in runs]
    summaries = [get_run_summary(i) for i in infos]
    summary = pd.concat(summaries, ignore_index=True)  # .sort_values('best_acc', ascending=False)

    if args.output:
        summary.to_csv(args.output, index=False)
    else:
        with pd.option_context('display.width', None), \
             pd.option_context('max_columns', None):

            # get col index of non unique columns (params that changes between runs)
            unique_cols = summary.apply(pd.Series.nunique) == 1
            non_unique_cols = summary.apply(pd.Series.nunique) != 1
            print(summary.loc[:, non_unique_cols])
            print("Common params:")
            print(summary.loc[0, unique_cols])


def offset_eval(runs):
    summaries = []
    for run in runs:
        run_info, model, loader = load_run(run, data=args.data, offset='all')
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


def ablation(runs):
    summaries = [get_run_summary(get_run_info(r)) for r in runs]
    summary = pd.concat(summaries, ignore_index=True)

    # Drop cols with unique value everywhere
    # value_counts = summary.apply(pd.Series.nunique)
    # cols_to_drop = value_counts[value_counts < 2].index
    # summary = summary.drop(cols_to_drop, axis=1)

    params = ['bidirectional', 'embed', 'hd', 'layers']
    for p in params:
        rest = params[:]
        rest.remove(p)
        table = summary.pivot_table(values='best_acc', columns=p, index=rest)
        table = table.mean()
        print(table)


def main(args):
    runs = find_runs(args.run_dir)

    if args.type == 'confusion':
        confusion_plot(runs)

    if args.type == 'status':
        display_status(runs)
        train_plot(runs, args.smooth)

    if args.type == 'multi-eval':
        offset_eval(runs)

    if args.type == 'ablation':
        ablation(runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show misc info about runs')
    parser.add_argument('type', choices=['confusion', 'status', 'multi-eval', 'ablation'], help='what to plot')
    parser.add_argument('run_dir', nargs='?', default='runs/', help='folder in which logs are searched')
    parser.add_argument('-d', '--data', help='eval data (for confusion)')
    parser.add_argument('-o', '--output', help='outfile (for status)')
    parser.add_argument('-s', '--smooth', default=0.0, help='exponential smooth weight')
    args = parser.parse_args()
    main(args)
