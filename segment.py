import matplotlib
matplotlib.use('Agg')

import argparse
import itertools
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from adjustText import adjust_text

from utils import load_run, get_predictions

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.5)


def find_thresholds(targets, predictions, stream=False, pr=False):
    if not stream:
        targets = np.concatenate(targets, axis=0)
        predictions = np.concatenate(predictions, axis=0)

    print('Targets:', targets.shape)
    print('Predictions:', predictions.shape)

    p, r, t = precision_recall_curve(targets.ravel(), predictions.ravel())
    
    if pr:
        np.savez(pr, p=p, r=r, t=t)
    
    t = np.insert(t, 0, 0)

    f1 = 2 * (p * r) / (p + r)
    _, global_thr = max(zip(f1, t))

    category_thr = []
    for i in range(targets.shape[1]):
        p, r, t = precision_recall_curve(targets[:, i], predictions[:, i])
        f1 = 2 * (p * r) / (p + r)
        t = np.insert(t, 0, 0)
        _, thr = max(zip(f1, t))
        category_thr.append(thr)

    category_thr = np.array(category_thr)

    return global_thr, category_thr


def compute_metrics(targets, predictions, thrs, stream=False):
    if not stream:
        targets = np.concatenate(targets, axis=0)
        predictions = np.concatenate(predictions, axis=0)

    global_thr, category_thr = thrs

    microAP = average_precision_score(targets, predictions, average='micro')
    macroAP = average_precision_score(targets, predictions, average='macro')

    print('Micro-AP: {}'.format(microAP))
    print('Macro-AP: {}'.format(macroAP))

    microF1 = f1_score(targets, predictions > global_thr, average='micro')
    macroF1 = f1_score(targets, predictions > global_thr, average='macro')

    print('Global Thr Micro-F1: {} {}'.format(microF1, global_thr))
    print('Global Thr Macro-F1: {} {}'.format(macroF1, global_thr))

    catMicroF1 = f1_score(targets, predictions > category_thr, average='micro')
    catMacroF1 = f1_score(targets, predictions > category_thr, average='macro')

    print('Class-based Thr Micro-F1: {}'.format(catMicroF1))
    print('Class-based Thr Macro-F1: {}'.format(catMacroF1))

    # data = pd.DataFrame(dict(BestF1=cat_f1s, Threshold=cat_thr), index=labels)
    # print(data)

    return microAP, macroAP, microF1, macroF1, catMicroF1, catMacroF1


def plot_preditctions(targets, predictions, seq_ids, labels, thr, out):
    order = []
    with PdfPages('/tmp/app.pdf') as pdf:
        for i, (y, y_hat, cur_sequence_id) in tqdm(enumerate(zip(targets, predictions, seq_ids)), total=len(targets)):

            n_samples, n_classes = y_hat.shape
            time = np.arange(n_samples)

            order.append(cur_sequence_id)
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, n_classes))
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 4.5),
                                     gridspec_kw={'height_ratios': [3.5, 1, 1]})

            for ax in axes:
                ax.set_ylim([0, 1.1])
                ax.set_prop_cycle('color', colors)
                ax.grid(b=True, which='major', linewidth=1.0)
                ax.grid(b=True, which='minor', linewidth=0.5)

            (ax1, ax2, ax3) = axes

            # title = plt.suptitle('Sequence {}: (thr={:.2f})]'.format(cur_sequence_id, thr), y=0.95)

            pad = 21

            ax1.set_ylabel(r'$p(P_i, C_j)$')
            ax1.plot(time, y_hat, label=labels)
            ax1.axhline(thr, color='k', linestyle='solid', linewidth=1)
            ax1.get_yaxis().set_ticks([0, 0.5, 1])
            ax1.get_yaxis().set_ticks([0.25, 0.75], minor=True)
            ax1.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax1.tick_params(axis='y', which='major', labelsize=12, labelrotation=90)

            ax2.set_ylabel(r'\textrm{Pred.}', labelpad=pad)
            ax2.get_yaxis().set_ticks([])
            ax2.plot(time, y_hat > thr)
            ax2.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            
            ax3.set_ylabel(r'\textrm{GT}', labelpad=pad)
            lines = ax3.plot(time, y)
            ax3.get_yaxis().set_ticks([])
            ax3.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax3.set_xlabel(r'\textrm{Frame}')
            ax3.set_xlim([0, len(time) - 1])

            legends_ix = set(y.sum(axis=0).nonzero()[0].tolist() +
                             (y_hat > thr).sum(axis=0).nonzero()[0].tolist())
            legends_ix = np.array(list(legends_ix))
            lines = np.array(lines)

            lines = lines[legends_ix]
            legends = labels[legends_ix]
            legends = ['\\textrm{{{}}}'.format(l) for l in legends]

            sns.despine()
            n_legend_lines = len(lines) // 4
            lgd = axes[-1].legend(lines, legends, loc='center', ncol=4, fontsize='medium',
                                  bbox_to_anchor=(0.5, -1.5 - 0.14 * n_legend_lines))
            
            # pdf.savefig(bbox_extra_artists=(lgd, title), bbox_inches='tight')
            pdf.savefig(bbox_extra_artists=(lgd, ), bbox_inches='tight')
            plt.close()

    order = np.argsort(order) + 1
    order = " ".join(map(str, order))
    os.system('pdftk /tmp/app.pdf cat {} output {}'.format(order, out))
        

def delay_plot(args):
    n_points = 100

    targets, predictions, _ = get_predictions(args.run_dir)
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    
    t = np.linspace(0, 1, 500, endpoint=False)
    p, r, f1 = [], [], []
    for thr in tqdm(t):
        pp, rr, ff1, _ = precision_recall_fscore_support(targets.ravel(), predictions.ravel() > thr, average='binary')
        p.append(pp)
        r.append(rr)
        f1.append(ff1)
    
    p, r, f1 = map(np.array, (p, r, f1))
    
    # keep = p > 0.75
    # t = t[keep]
    # f1 = f1[keep]

    # offset = len(f1) // n_points
    # f1 = f1[::offset]
    # t = t[::offset]

    print('Num. Thresholds:', len(t))

    def find_annotations(curve):
        start = 0
        for value, sublist in itertools.groupby(curve):
            duration = len(list(sublist))
            if value == 1:  # skip 0s
                end = start + duration - 1
                yield (start, end, duration)
            start += duration

    def iou(annot, ground):
        # min of ends - max of starts + 1
        intersection = np.minimum(annot[:, 1], ground[:, 1]) - np.maximum(annot[:, 0], ground[:, 0]) + 1
        intersection = np.maximum(intersection, 0)
        # union = sum of durations - intersection
        union = annot[:, 2] + ground[:, 2] - intersection
        return intersection / union

    n_classes = targets.shape[1]

    # find all annotations of groundtruth
    ground = []
    n_annotations = 0
    for i in range(n_classes):
        annot = find_annotations(targets[:, i])
        annot = np.array(list(annot))
        ground.append(annot)
        n_annotations += annot.shape[0]

    # iterate over thresholds
    global_ious = []
    global_delays = []
    global_thr = []
    mean_delays = []
    accuracies = []    
    
    for thr in tqdm(t):
        hard_predictions = predictions > thr

        delays = []
        accuracy = 0
        for i in range(n_classes):
            annot = find_annotations(hard_predictions[:, i])
            annot = np.array(list(annot))
            if annot.size:
                # For each ground-truth start, search the nearest start of an annotation:
                # - compute start distances between all (prediction, gt) pair
                all_delays = annot[:, 0].reshape(1, -1) - ground[i][:, 0].reshape(-1, 1)

                # discard negative delays
                # all_delays = all_delays.astype(np.float32)
                # all_delays[all_delays < 0] = np.inf

                # - find the nearest annotations in terms of start frame
                nearest_annot_idx = np.argmin(np.absolute(all_delays), axis=1)
                nearest_annot = annot[nearest_annot_idx]
                nearest_delays = all_delays[np.arange(all_delays.shape[0]), nearest_annot_idx]
                # - keep only valid annotations (IoU > 0.5)
                annot_ious = iou(nearest_annot, ground[i])
                valid = annot_ious >= 0.5                
                valid_delays = nearest_delays[valid]
                
                global_ious.append(annot_ious[valid])
                global_delays.append(valid_delays)
                global_thr.append(np.ones_like(valid_delays) * thr)
                
                # save delays and number of valid annotations
                delays.append(valid_delays)
                accuracy += len(valid_delays)

        if delays:
            mean_delay = np.concatenate(delays).mean()
            accuracy /= n_annotations

            mean_delays.append(mean_delay / 120.0)
            accuracies.append(accuracy)
        else:
            print(thr, 'no valid predictions')

    metrics = (p, r, f1)
    names = ('Precision', 'Recall', 'F1')
    
#    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#    for i, (y, ylabel, ax) in enumerate(zip(metrics, names, axes.ravel()[:3])):
        
    for i, (y, ylabel) in enumerate(zip(metrics, names)):
        fig = plt.figure(figsize=(5, 4))
        ax = plt.gca()
    
        ax.plot(mean_delays, y, c='k', linewidth='0.5', zorder=1)
        im = ax.scatter(mean_delays, y, marker='.', c=t, zorder=2)
        fig.colorbar(im, ax=ax, use_gridspec=True, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        '''
        plt.minorticks_on()
        plt.grid(b=True, which='minor', linestyle='--', linewidth=0.5)
        
        xticks = plt.gca().get_xticks()
        labels = ['\\textrm{{{:g}}}\n\\textrm{{({:g})}}'.format(x, round(x*120)) for x in xticks]
        plt.gca().set_xticklabels(labels)
        '''
        
        ax.set_xlabel(r'\textrm{Average Delay [$s$]}')
        ax.set_ylabel('\\textrm{{{}}}'.format(ylabel))

        # n_thr_points = 10
        # skip = len(y) // n_thr_points
        #
        # show_d = mean_delays[::skip] if skip else mean_delays
        # show_y = y[::skip] if skip else y
        # show_t = t[::skip] if skip else t
        #
        # for d, _y, thr in zip(show_d, show_y, show_t):
        #     # if thr < 0.01: continue
        #     txt = 'T={:3.2f}'.format(thr)
        #     txt = r'\textrm{' + txt + '}'
        #     ax.annotate(txt, xy=(d,_y), fontsize=6)

        ax.set_title('\\textrm{{{} vs Average Delay}}'.format(ylabel))
        plt.tight_layout()
        plt.savefig('delay-{}.pdf'.format(ylabel.lower()))
        plt.close()
        
    # Last ax
    # ax = axes[1, 1]
    fig = plt.figure(dpi=600, figsize=(5, 4))
    ax = plt.gca()
    
    global_delays = np.concatenate(global_delays)
    global_ious = np.concatenate(global_ious)
    global_thr = np.concatenate(global_thr)
    
    ax.set_title(r'\textrm{Delay vs IoU}')
    ax.set_xlabel(r'\textrm{IoU}')
    ax.set_ylabel(r'\textrm{Delay (frames)}')
    ax.set_xlim([0.48, 1])
    ax.set_ylim([-500, 300])
    im = ax.scatter(global_ious, global_delays, 1, c=global_thr, rasterized=True)
    
    fig.colorbar(im, ax=ax, use_gridspec=True, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    fig.tight_layout()
    plt.savefig('delay-iou.pdf')
    plt.close()


def main(args):

    if args.delay:
        delay_plot(args)
        return

    if args.pr:
        targets, predictions, _ = get_predictions(args.run_dir, stream=False)
        pr_fname = os.path.join(args.run_dir, 'pr.npz')
        _ = find_thresholds(targets, predictions, stream=False, pr=pr_fname)
        return
        
    if args.compute_metrics or args.plot_predictions:
        run_info, model, loader = load_run(args.run_dir, data=args.data)
        params = run_info[-1]
        labels = np.array([a.replace('hdm05_', '') for a in loader[1].dataset.action_descriptions])

    if args.compute_metrics:
        rows = []
        thr_tab = pd.DataFrame(index=labels, columns=pd.MultiIndex.from_product([['fair', 'unfair'], ['stream', 'sequences']]))
        for stream, fair in itertools.product((False, True), repeat=2):

            targets, predictions, annot_time = get_predictions(args.run_dir, stream=stream, force=args.force)
            thr_targets, thr_predictions = targets, predictions
            if fair:
                thr_targets, thr_predictions, _ = get_predictions(args.run_dir, train=True, stream=stream, force=args.force)
                train_targets = thr_targets

            print('Stream: {} Fair: {}'.format(stream, fair))
            thrs = find_thresholds(thr_targets, thr_predictions, stream=stream)

            thr_tab[(('fair' if fair else 'unfair'), ('stream' if stream else 'sequences'))] = thrs[1]
            metrics = compute_metrics(targets, predictions, thrs, stream=stream)
            row = (stream, fair) + metrics + (annot_time,)
            rows.append(row)

        thr_tab['train_support'] = train_targets.sum(axis=0)
        thr_tab['test_support'] = targets.sum(axis=0)
        thresholds_file = os.path.join(args.run_dir, 'thresholds.csv')
        thr_tab.to_csv(thresholds_file)

        columns = ('Stream', 'Fair', 'microAP', 'macroAP', 'microF1', 'macroF1', 'catMicroF1', 'catMacroF1', 'AnnotTime')

        metrics = pd.DataFrame.from_records(rows, columns=columns)
        metrics_file = os.path.join(args.run_dir, 'metrics.csv')
        metrics.to_csv(metrics_file)
        print(metrics)

    if args.plot_predictions:
        stream, fair = False, False
        targets, predictions, annot_time = get_predictions(args.run_dir, stream=stream, force=args.force)
        thr_targets, thr_predictions = targets, predictions
        if fair:
            thr_targets, thr_predictions, _ = get_predictions(args.run_dir, train=True, stream=stream, force=args.force)
            train_targets = thr_targets
        
        thrs = find_thresholds(thr_targets, thr_predictions, stream=stream)
        
        global_thr, multiple_thrs = thrs
        out = os.path.join(args.run_dir, 'time-analysis.pdf')
        seq_ids = [int(loader[1].dataset.data[i]['seq_id']) for i in range(len(targets))]
        
        plot_preditctions(targets, predictions, seq_ids, labels, global_thr, out)
        return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Analysis')
    parser.add_argument('run_dir', help='folder of the model to use')
    parser.add_argument('-d', '--data', help='data to segment (if different from test data of the run)')
    parser.add_argument('-c', '--compute-metrics', action='store_true', help='compute eval metrics')
    parser.add_argument('-p', '--plot-predictions', action='store_true', help='draw segmentation plots and show eval metrics')
    parser.add_argument('--pr', action='store_true', help='draw precision-recall curve')
    parser.add_argument('--delay', action='store_true', help='draw delay plot')
    parser.add_argument('-f', '--force', action='store_true', help='force to recompute predictions')
    
    parser.set_defaults(delay=False)
    parser.set_defaults(compute_metrics=False)
    parser.set_defaults(plot_predictions=False)
    parser.set_defaults(pr=False)
    parser.set_defaults(force=False)
    args = parser.parse_args()
    main(args)
