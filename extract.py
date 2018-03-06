import argparse

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from utils import load_run


def main(args):
    run_info, model, loader = load_run(args.run_dir, data=args.data)
    params = run_info[-1]

    features = []
    for x, _ in tqdm(loader):
        if params['cuda']:
            x = x.cuda()
        x = Variable(x, volatile=True)
        f = model.extract(x).data
        features.append(f.cpu().numpy().squeeze())

    if args.format == 'jan':
        with open(args.output, 'w') as f:
            for sample, feature in tqdm(zip(loader.dataset.data, features), total=len(features)):
                f.write('#objectKey messif.objects.keys.AbstractObjectKey {}\n'.format(sample['id']))
                feature.tofile(f, sep=',')
                f.write('\n')
        return

    if args.format == 'numpy':
        features = np.stack(features)
        np.save(args.output, features)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Analysis')
    parser.add_argument('run_dir', help='folder of the model to use')
    parser.add_argument('data', help='data from which extract features')
    parser.add_argument('output', help='features output file')
    parser.add_argument('-f', '--format', choices=['jan', 'numpy'], default='jan', help='features output file')
    args = parser.parse_args()
    main(args)
