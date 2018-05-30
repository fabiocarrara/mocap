import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import get_classifications, load_run


def main(args):
    _, _, loaders = load_run(args.run_dir)
    predictions, targets, annot_time = get_classifications(args.run_dir, train=False)

    labels = loaders[1].dataset.action_descriptions

    idx = np.nonzero(predictions != targets)
    # acc = (predictions == targets).sum() / targets.shape[0]
    c = confusion_matrix(targets, predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(c, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, annot_kws={'fontsize': 8})

    plt.tight_layout()
    plt.savefig('classify_confusion.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify data using trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_dir', help='folder of the model to use')
    parser.add_argument('-d', '--data', help='data to segment (if different from test data of the run)')
    args = parser.parse_args()

    main(args)
