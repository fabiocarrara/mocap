import argparse
import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from natsort import natsorted
from utils import get_run_summary, get_run_info, find_runs

sns.set_style('darkgrid')


def segm_summary(args):
    runs = find_runs('runs_segmentation_hdm05-122/') + \
           find_runs('runs_segmentation_hdm05-65/') + \
           find_runs('runs_segmentation_hdm05-15/')

    summaries = [get_run_summary(get_run_info(r), epoch='microAP') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)
    # remove best_ prefix
    summary.columns = map(lambda x: x.replace('best_', ''), summary.columns)

    summary['Dataset'] = summary['run_dir'].str.extract('.*(hdm05-\d+)', expand=False).str.upper()
    summary['Fold'] = summary['val_data'].str.extract('.*fold-(\d+)-of.*', expand=False).apply(lambda x: 'Fold ' + x)
    sorted_datasets = natsorted(summary['Dataset'].unique())

    # summary = summary[summary['fps'] == 120.0]

    model_labels = pd.np.array(['Uni-LSTM', 'Bi-LSTM'])
    summary['bidirectional'] = model_labels[summary['bidirectional'].astype(int)]

    metric_cols = ('microAP', 'macroAP', 'F1', 'microMultiF1', 'macroMultiF1')
    metric_names = ('micro-averaged AP', 'macro-averaged AP', '$F_1$ (optimal threshold)',
                    'micro-averaged $F_1$ (multiple optimal thresholds)',
                    'macro-averaged $F_1$ (multiple optimal thresholds)')

    aggfunc = lambda x: '{:3.2f} $\pm$ {:3.2f}'.format(pd.np.mean(x), pd.np.std(x))

    for metric, metric_name in zip(metric_cols, metric_names):
        pivot = pd.pivot_table(summary, index=['fps', 'bidirectional'], values=metric,
                               columns=['Dataset', 'Fold']) # , aggfunc=aggfunc)
        pivot = pivot.reindex(model_labels, axis=0, level='bidirectional')
        pivot = pivot.reindex(sorted_datasets, axis=1, level='Dataset')
        print('\\multicolumn{7}{c}{\\textit{%s}} \\\\' % metric_name)
        print(pivot.to_latex(column_format='lcccccc', multicolumn_format='c', na_rep='-', escape=False))
        print()


def single_model(args, bidir):
    runs = find_runs('runs_segmentation_hdm05-122/') + \
           find_runs('runs_segmentation_hdm05-65/') + \
           find_runs('runs_segmentation_hdm05-15/')

    summaries = [get_run_summary(get_run_info(r), epoch='test') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)

    summary['Dataset'] = summary['run_dir'].str.extract('.*(hdm05-\d+)', expand=False).str.upper()
    summary['Fold'] = summary['val_data'].str.extract('.*fold-(\d+)-of.*', expand=False).apply(lambda x: 'Fold ' + x)

    sorted_datasets = natsorted(summary['Dataset'].unique())

    summary = summary[summary['bidirectional'] == bidir]
    summary = summary[summary['fps'] == 120.0]
    summary = summary[summary['Fair'] == args.fair]
    summary = summary[summary['Stream'] == args.stream]

    # summary.columns = map(lambda x: x.replace('best_', ''), summary.columns)
    # model = 'Bi-LSTM' if bidir else 'Uni-LSTM'

    metric_cols = ('microAP', 'macroAP', 'microF1', 'macroF1', 'catMicroF1', 'catMacroF1')
    metric_names = ('micro-$AP$', 'macro-$AP$', 'micro-$F_1$', 'macro-$F_1$',  'cmicro-$F_1$', 'cmacro-$F_1$')

    aggfunc = lambda x: '{:4.2%} $\pm$ {:3.2%}'.format(pd.np.mean(x), pd.np.std(x))

    pivot = pd.pivot_table(summary, values=metric_cols, columns='Dataset', aggfunc=aggfunc)
    pivot = pivot.reindex(metric_cols, axis=0)
    pivot = pivot.reindex(sorted_datasets, axis=1)
    pivot.index = metric_names

    print(pivot.to_latex(column_format='rXXX', multicolumn_format='r', na_rep='-', escape=False))
    if args.output:
        pivot.to_csv(args.output)


def uni(args):
    single_model(args, False)


def bi(args):
    single_model(args, True)


def sota_hdm05(args):
    runs = find_runs('runs_segmentation_hdm05-15_20-80/')
    summaries = [get_run_summary(get_run_info(r), epoch='test') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)

    # summary = summary[summary['bidirectional'] == bidir]
    summary = summary[summary['fps'] == 120.0]
    summary = summary[summary['Fair'] == args.fair]
    summary = summary[summary['Stream'] == args.stream]

    summary['Dataset'] = 'HDM05-15 (20-80)'
    # summary.columns = map(lambda x: x.replace('best_', ''), summary.columns)
    model_names = np.array(['\\unimodel{}', '\\bimodel{}'])
    summary['bidirectional'] = model_names[summary['bidirectional'].astype(int)]

    metric_cols = ('microAP', 'macroAP', 'microF1', 'macroF1', 'catMicroF1', 'catMacroF1')
    metric_names = ('micro-$AP$', 'macro-$AP$', 'micro-$F_1$', 'macro-$F_1$',  'cmicro-$F_1$', 'cmacro-$F_1$')

    pivot = pd.pivot_table(summary, values=metric_cols, columns='bidirectional')
    pivot = pivot.reindex(metric_cols, axis=0)
    pivot = pivot.reindex(model_names, axis=1)
    pivot.index = metric_names

    print(pivot.to_latex(column_format='rXX', multicolumn_format='r', na_rep='-', escape=False, formatters=['{:4.2%}'.format, '{:4.2%}'.format]))
    if args.output:
        pivot.to_csv(args.output)


def all_fps(args):
    runs = find_runs('runs_segmentation_hdm05-122/') + \
           find_runs('runs_segmentation_hdm05-65/') + \
           find_runs('runs_segmentation_hdm05-15/')

    summaries = [get_run_summary(get_run_info(r), epoch='microAP') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)
    summary['Dataset'] = summary['run_dir'].str.extract('.*(hdm05-\d+)', expand=False).str.upper()
    summary['Fold'] = summary['val_data'].str.extract('.*fold-(\d+)-of.*', expand=False).apply(lambda x: 'Fold ' + x)

    summary = summary[~summary['fps'].isin((6.0, 10.0, 12.0, 20.0))]

    sorted_datasets = natsorted(summary['Dataset'].unique())

    summary.columns = list(map(lambda x: x.replace('best_', ''), summary.columns))

    metric_cols = ('microAP', 'macroAP', 'F1', 'microMultiF1', 'macroMultiF1')

    summary = summary.groupby(['bidirectional', 'Dataset', 'fps'], as_index=False)[metric_cols].aggregate(pd.np.mean)
    fps_values = summary['fps'].unique()

    id_cols = list(set(summary.columns) - set(metric_cols))
    summary = summary.melt(id_vars=id_cols, value_vars=metric_cols, var_name='Metric', value_name='Value')

    g = sns.FacetGrid(summary, col='Dataset', row='Metric', hue='bidirectional',
                      col_order=sorted_datasets, margin_titles=True,
                      size=2, aspect=1.5)
    g = g.map(plt.semilogx, 'fps', 'Value') \
        .set(xticks=fps_values) \
        .set_xticklabels(['{:g}'.format(f) for f in fps_values]) \
        .add_legend()

    plt.subplots_adjust(top=0.925)
    g.fig.suptitle('Performance vs FPS')
    g.savefig(args.output)
    
    
def fps(args):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.2)

    runs = find_runs('runs_segmentation_hdm05-122/') + \
           find_runs('runs_segmentation_hdm05-65/') + \
           find_runs('runs_segmentation_hdm05-15/')

    summaries = [get_run_summary(get_run_info(r), epoch='test') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)
    summary['Dataset'] = summary['run_dir'].str.extract('.*(hdm05-\d+)', expand=False).str.upper()
    summary['Fold'] = summary['val_data'].str.extract('.*fold-(\d+)-of.*', expand=False).apply(lambda x: 'Fold ' + x)

    summary = summary[summary['Fair'] == args.fair]
    summary = summary[summary['Stream'] == args.stream]
    summary = summary[~summary['fps'].isin((0.5, 6.0, 10.0, 12.0, 20.0, 24.0, 40.0))]

    metric = 'microF1'
    summary = summary.groupby(['bidirectional', 'Dataset', 'fps'], as_index=False)[metric].aggregate(pd.np.mean)

    fps_values = summary['fps'].unique()
    sorted_datasets = natsorted(summary['Dataset'].unique())

    h = 2.5
    fig, ax = plt.subplots(1, 3, figsize=(4*h, h))
    for i, dset in enumerate(sorted_datasets):
        # Online
        keep = (summary['Dataset'] == dset) & ~summary['bidirectional']
        xy = summary[keep].sort_values('fps')
        ax[i].semilogx(xy['fps'], xy[metric], color='b', marker='.', label=r'\textrm{Online-LSTM}')

        # Offline
        keep = (summary['Dataset'] == dset) & summary['bidirectional']
        xy = summary[keep].sort_values('fps')
        ax[i].semilogx(xy['fps'], xy[metric], color='r', marker='.', label=r'\textrm{Offline-LSTM}')

        ax[i].set_title('\\textrm{{{}}}'.format(dset))
        ax[i].set_xticks(fps_values)
        ax[i].set_xticklabels(['\\textrm{{{:g}}}'.format(f) for f in fps_values])

    ax[0].set_ylim([0.75, 0.825])
    ax[0].set_yticks([0.77, 0.79, 0.81], minor=True)
    ax[0].grid(b=True, axis='y', which='minor', linestyle='--')

    ax[1].set_ylim([0.50, 0.8])
    ax[1].set_yticks([0.55, 0.65, 0.75], minor=True)
    ax[1].grid(b=True, axis='y', which='minor', linestyle='--')

    ax[2].set_ylim([0.25, 0.7])
    ax[2].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax[2].set_yticks([0.35, 0.45, 0.55, 0.65], minor=True)
    ax[2].grid(b=True, axis='y', which='minor', linestyle='--')

    # ax[0].set_yticks

    ax[0].set_ylabel(r'\textrm{micro-$F_1$}')
    ax[1].set_xlabel(r'\textrm{FPS (logarithmic scale)}')
    ax[1].legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(args.output)


def pr_plot(args):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    sns.set_style('whitegrid')
    sns.set_context('notebook', font_scale=1.4)

    uni_run = "runs_segmentation_hdm05-15/segment_tr-HDM05-15-whole-seq+annot-fold-1-of-2_vl-HDM05-15-whole-seq+annot-fold-2-of-2_biFalse_emb64_h1024_s1_l1_sigmoid_a1_c10.0_d0.5_lr0.0005_wd0.0001_e200_f120_o-random_opt-adam_ls0.0_bal-none"
    bi_run = "runs_segmentation_hdm05-15/segment_tr-HDM05-15-whole-seq+annot-fold-1-of-2_vl-HDM05-15-whole-seq+annot-fold-2-of-2_biTrue_emb64_h1024_s1_l1_sigmoid_a1_c10.0_d0.5_lr0.0005_wd0.0001_e200_f120_o-random_opt-adam_ls0.0_bal-none"

    uni_pr = os.path.join(uni_run, 'pr.npz')
    uni_pr = np.load(uni_pr)
    uni_p, uni_r = uni_pr['p'], uni_pr['r']

    bi_pr = os.path.join(bi_run, 'pr.npz')
    bi_pr = np.load(bi_pr)
    bi_p, bi_r = bi_pr['p'], bi_pr['r']

    plt.figure(figsize=(5, 3.5))
    plt.step(uni_r, uni_p, color='b', where='post', label=r'\textrm{Online-LSTM}')
    plt.step(bi_r, bi_p, color='r', where='post', label=r'\textrm{Offine-LSTM}')
    bi_p_on_uni_r = np.interp(uni_r, bi_r[::-1], bi_p[::-1])
    l = plt.fill_between(uni_r, uni_p, step='post', alpha=0.4, color='b')
    l.set_rasterized(True)
    l = plt.fill_between(uni_r, bi_p_on_uni_r, uni_p, step='post', alpha=0.4, color='r')
    l.set_rasterized(True)

    plt.xlabel(r'\textrm{Recall}')
    plt.ylabel(r'\textrm{Precision}')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='best', frameon=True)
    # plt.title('Micro-averaged Precision-Recall curve')
    plt.tight_layout()
    sns.despine()
    plt.savefig(args.output)
    plt.close()


def time(args):

    runs = find_runs('runs_segmentation_hdm05-122/') + \
           find_runs('runs_segmentation_hdm05-65/') + \
           find_runs('runs_segmentation_hdm05-15/')

    summaries = [get_run_summary(get_run_info(r), epoch='test') for r in runs]
    summary = pd.concat(summaries, ignore_index=True)

    summary = summary[summary['Fair'] & ~summary['Stream']]

    summary['Dataset'] = summary['run_dir'].str.extract('.*(hdm05-\d+)', expand=False).str.upper()
    summary['Fold'] = summary['val_data'].str.extract('.*fold-(\d+)-of.*', expand=False).apply(lambda x: 'Fold ' + x)

    sorted_datasets = natsorted(summary['Dataset'].unique())
    pivot = pd.pivot_table(summary, values='AnnotTime', index=['bidirectional', 'fps'], columns=['Dataset', 'Fold'])
    pivot = pivot.reindex(sorted_datasets, axis=1, level='Dataset')
    pivot.to_csv('annotation_times.csv')
    print(pivot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce Tables')
    parser.add_argument('table', help='which table to produce')
    parser.add_argument('-s', '--stream', action='store_true', help='consider stream predictions')
    parser.add_argument('-f', '--fair', action='store_true', help='consider train thresholds')
    parser.add_argument('-o', '--output', help='where to save the produced table')
    parser.set_defaults(fair=False, stream=False)
    args = parser.parse_args()

    table_fn = args.table.replace('-', '_')
    table_fn = globals()[table_fn]
    table_fn(args)
