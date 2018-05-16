import argparse
import os
import pickle
import re
import zipfile
from itertools import groupby, chain, tee, filterfalse

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_sequences(fle):
    if fle.endswith('.zip'):
        z = zipfile.ZipFile(fle, 'r')
        f = z.open(os.path.basename(fle).replace('.zip', '.data'), 'r')
    else:
        f = open(fle)

    grps = groupby(f, key=lambda x: x.lstrip().startswith(b'#objectKey'))
    for k, v in grps:
        if k:
            yield list(chain([next(v)], (next(grps)[1])))  # all lines up to next #objectKey

    f.close()
    if fle.endswith('.zip'):
        z.close()


def parse_data_line(line):
    values = re.split(b'; |, | ', line)
    values = map(float, values)
    values = np.fromiter(values, dtype=np.float32)
    values = values.reshape(-1, 3)
    return values


def parse_sequence(lines):
    header = lines[0]
    header_regexp = r'.*\s(S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})_(\d+)_(\d+)_(\d+)_(\d+)).*'
    matches = re.match(header_regexp, header.decode('utf-8'))
    assert matches is not None, "Error parsing data header: %s" % header
    attributes = matches.groups()
    sample_id = attributes[0]
    setup, camera, performer, replication, action, action_again, start_frame, duration, _ = map(int, attributes[1:])
    assert action == action_again, "Action mismatch: %s" % header
    lines = lines[2:]  # discard header

    data = [parse_data_line(line) for line in lines]
    data = np.stack(data)  # time, joint, xyz
    data = np.delete(data, [5, 10, 12, 13, 19, 26], axis=1)  # remove duplicates joints
    assert data.shape[0] == duration, "Duration mismatch: %s (%d vs %d)" % (header, data.shape[0], duration)
    sequence = dict(
        id=sample_id,
        setup=setup,
        camera=camera,
        performer=performer,
        replication=replication,
        action=action,
        start_frame=start_frame,
        duration=duration,
        data=data
    )
    return sequence
    

def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filter(pred, t1), filterfalse(pred, t2)


def main(args):
    sequences = get_sequences(args.data)
    parsed = (parse_sequence(seq) for seq in sequences)

    actions_to_exclude = (50, 51, 52, 53, 54, 55, 56, 57, 59, 60)
    parsed = filter(lambda x: x['action'] not in actions_to_exclude, parsed)

    if args.scenario == 'cross-view':
        is_train = lambda x: x['camera'] >= 2
    elif args.scenario == 'cross-subject':
        is_train = lambda x: x['performer'] in (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38)

    train, test = partition(is_train, parsed)

    train, test = list(tqdm(train)), list(tqdm(test))

    out = '{}-train.pkl'.format(args.parsed_data)
    with open(out, 'wb') as outfile:
        pickle.dump(train, outfile)

    out = '{}-test.pkl'.format(args.parsed_data)
    with open(out, 'wb') as outfile:
        pickle.dump(test, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('data', help='path to data file (in textual format)')
    parser.add_argument('--scenario', choices=('cross-view', 'cross-subject'), help='scenario for splitting the dataset')
    parser.add_argument('parsed_data', help='output file with parsed data file (in Pickle format)')
    args = parser.parse_args()
    main(args)
