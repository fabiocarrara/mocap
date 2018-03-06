import os
import argparse
import matplotlib
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from torch.autograd import Variable
from tqdm import tqdm

from dataset import MotionDataset
from utils import load_run

sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=0.8)


def main_old(args):
    dataset = MotionDataset('data/split1.pkl', fps=10)
    actions = dataset.actions.keys()
    run_info, model, loader = load_run(args.run_dir, args.data, keep_actions=actions)
    params = run_info[-1]

    out = os.path.join(args.run_dir, 'time-analysis.pdf')
    with PdfPages(out) as pdf:
        for i in tqdm(range(len(loader.dataset))):
            x, annotations = loader.dataset[i]
            if params['cuda']:
                x = x.cuda()
            x = Variable(x, volatile=True)

            outs = model.steps_forward(x)
            head = params.get('head', 'softmax')
            if head == 'softmax':
                outs = torch.nn.functional.softmax(outs, dim=1)
            elif head == 'sigmoid':
                outs = torch.nn.functional.sigmoid(outs)

            confidences = outs.data.cpu().numpy()  # [:, y]
            n_samples, n_classes = outs.shape
            time = np.arange(n_samples)
            # time = torch.linspace(0, 1, seq_len).numpy()

            groundtruth = np.zeros_like(confidences, dtype=int)
            for a in annotations:
                class_id = loader.dataset.action_id_to_ix[a['action_id']]
                start = int(round(a['start_frame'] / loader.dataset.skip))
                end = int(round((a['start_frame'] + a['duration']) / loader.dataset.skip))
                groundtruth[start:end, class_id] = 1

            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, n_classes))
            fig, axes = plt.subplots(3, 1)

            for ax in axes:
                ax.set_ylim([0, 1])
                ax.set_color_cycle(colors)

            (ax1, ax2, ax3) = axes
            ax1.plot(time, confidences)
            ax2.plot(time, groundtruth)
            ax3.plot(time, confidences * groundtruth)
            pdf.savefig()
            # plt.savefig('time-analysis.pdf')
            plt.close()


def main(args):
    run_info, model, loader = load_run(args.run_dir, data=args.data)
    params = run_info[-1]

    out = os.path.join(args.run_dir, 'time-analysis.pdf')
    labels = np.array([a.replace('hdm05_', '') for a in loader.dataset.action_descriptions])

    best_f1s = []
    targets = []
    predictions = []
    with PdfPages('/tmp/app.pdf') as pdf:
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

            n_samples, n_classes = y_hat.shape
            time = np.arange(n_samples)
            # time = torch.linspace(0, 1, seq_len).numpy()

            # np.savez('segmentation_outs_n_preds.npz', y=y, y_hat=y_hat)
            # break

            ap = average_precision_score(y, y_hat, average='micro')
            p, r, t = precision_recall_curve(y.ravel(), y_hat.ravel())
            t = np.insert(t, 0, 0)

            f1 = 2 * (p * r) / (p + r)
            best_f1, best_thr = max(zip(f1, t))

            best_f1s.append(best_f1)
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, n_classes))
            fig, axes = plt.subplots(3, 1)

            for ax in axes:
                ax.set_ylim([0, 1.1])
                ax.set_prop_cycle('color', colors)

            (ax1, ax2, ax3) = axes
            # (ax1, ax2) = axes
            ax1.set_title('Prediction [AP={:.1%}, F1={:.1%} (thr={})]'.format(ap, best_f1, best_thr))
            ax1.plot(time, y_hat, label=labels)
            ax2.set_title('Groundtruth ({})'.format(loader.dataset.data[i]['seq_id']))
            lines = ax2.plot(time, y)
            ax3.set_title('Masked Prediction')
            lines = ax3.plot(time, y_hat * y)

            legends_ix = set(y.sum(axis=0).nonzero()[0].tolist() +
                             (y_hat > 0.2).sum(axis=0).nonzero()[0].tolist())

            legends_ix = np.array(list(legends_ix))
            lines = np.array(lines)

            lines = lines[legends_ix]
            legends = labels[legends_ix]

            sns.despine()
            lgd = ax2.legend(lines, legends, loc='center', ncol=6, bbox_to_anchor=(0.5, -0.42))
            plt.tight_layout()
            pdf.savefig(bbox_extra_artists=(lgd,), bbox_inches='tight')
            # plt.savefig('time-analysis.pdf')
            plt.close()

    best_f1s = np.array(best_f1s)
    order = np.argsort(best_f1s)[::-1] + 1
    order = " ".join(map(str, order))
    os.system('pdftk /tmp/app.pdf cat {} output {}'.format(order, out))

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    p, r, t = precision_recall_curve(targets.ravel(), predictions.ravel())
    t = np.insert(t, 0, 0)

    f1 = 2 * (p * r) / (p + r)
    best_f1, best_thr = max(zip(f1, t))

    print('Single Thr F1: {} {}'.format(best_f1, best_thr))

    cat_f1s = []
    cat_thr = []
    for i in range(n_classes):
        p, r, t = precision_recall_curve(targets[:, i], predictions[:, i])
        f1 = 2 * (p * r) / (p + r)
        t = np.insert(t, 0, 0)
        b_f1, b_thr = max(zip(f1, t))
        cat_f1s.append(b_f1)
        cat_thr.append(b_thr)

    data = pd.DataFrame(dict(BestF1=cat_f1s, Threshold=cat_thr), index=labels)
    print(data)
    support = targets.sum(axis=0)
    avgF1 = (data['BestF1'].values * support).sum() / support.sum()
    print(avgF1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Analysis')
    parser.add_argument('run_dir', help='folder of the model to use')
    parser.add_argument('-d', '--data', help='data to analyze')
    args = parser.parse_args()
    main(args)
