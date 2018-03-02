import argparse

import pandas as pd
import re
import pickle

import numpy as np

from itertools import groupby, chain

from tqdm import tqdm


def get_sequences(fle):
    with open(fle) as f:
        grps = groupby(f, key=lambda x: x.lstrip().startswith("#objectKey"))
        for k, v in grps:
            if k:
                yield list(chain([next(v)], (next(grps)[1])))  # all lines up to next #objectKey


def parse_data_line(line):
    values = re.split('; |, | ', line)
    values = map(float, values)
    values = np.fromiter(values, dtype=np.float32)
    values = values.reshape(-1, 3)
    return values


def parse_sequence(lines):
    header = lines[0]
    header_regexp = r'.*\s((\d+)_(\d+)_(\d+)_(\d+))'
    matches = re.match(header_regexp, header)
    assert matches is not None, "Error parsing data header: %s" % header
    attributes = matches.groups()
    sample_id = attributes[0]
    seq_id, action_id, start_frame, duration = map(int, attributes[1:])
    lines = lines[2:]  # discard header

    data = [parse_data_line(line) for line in lines]
    data = np.stack(data)
    sequence = dict(
        id=sample_id,
        seq_id=seq_id,
        action_id=action_id,
        start_frame=start_frame,
        duration=duration,
        data=data
    )
    return sequence


def get_ids_to_keep(split_file, format='list', train=True):

    if format == 'list':
        id_regexp = r'.*/(\d+_\d+_\d+_\d+).*'

        def get_id(line):
            matches = re.match(id_regexp, line)
            return matches.groups()[0] if matches else None

        with open(split_file, 'rt') as f:
            ids = {get_id(line) for line in f}

    elif format == 'csv':
        ids = set(pd.read_csv(split_file, header=None).iloc[0])

    elif format == 'petr':
        with open(split_file, 'rt') as f:
            lines = f.readlines()

        idx = 1 if train else 4
        ids = map(int, lines[idx].rstrip('\n ,').split(','))
        ids = set(ids)

    return ids


def parse_annotated_sequence(lines, annotations):
    seq_id = int(lines[0].split(' ')[-1])
    duration = int(lines[1].split(';')[0])
    annotations = [a for a in annotations if a['seq_id'] == seq_id]
    for a in annotations:
        if 'data' in a:
            del a['data']

    lines = lines[2:]  # discard header

    data = [parse_data_line(line) for line in lines]
    data = np.stack(data)
    sequence = dict(
        seq_id=seq_id,
        annotations=annotations,
        duration=duration,
        data=data
    )
    return sequence


def load_annotations(annot_file, format, train=True):
    if format == 'pkl':
        with open(annot_file, 'rb') as infile:
            annotations = pickle.load(infile)

    elif format == 'petr':
        with open(annot_file, 'rt') as infile:
            lines = infile.readlines()

        idx = 7 if train else 10
        ids = lines[idx].rstrip('\n ,').split(',')

        def parse_annotation(a):
            fields = a.strip().split('_')
            fields = map(int, fields)
            names = ('seq_id', 'action_id', 'start_frame', 'duration')
            return dict(zip(names, fields))

        annotations = [parse_annotation(i) for i in ids]

    return annotations


def main(args):
    sequences = get_sequences(args.data)

    if args.annotations:  # parse parent sequences containing multiple annotations
        annotations = load_annotations(args.annotations, args.af, args.train)
        parsed = (parse_annotated_sequence(seq, annotations) for seq in sequences)
    else:  # parse single annotated sequences
        parsed = (parse_sequence(seq) for seq in sequences)

    if args.split:
        key = 'seq_id' if args.annotations else 'id'
        ids_to_keep = get_ids_to_keep(args.split, args.sf, args.train)
        parsed = filter(lambda x: x[key] in ids_to_keep, parsed)

    parsed = list(tqdm(parsed))
    with open(args.parsed_data, 'wb') as outfile:
        pickle.dump(parsed, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('data', help='path to data file (in textual format)')
    parser.add_argument('-s', '--split', help='path to optional split file (in textual format)')
    parser.add_argument('-a', '--annotations', help='path to annotations file for parent sequences')
    parser.add_argument('--sf', '--split-format', choices=['list', 'csv', 'petr'], default='list', help='split format')
    parser.add_argument('--af', '--annot-format', choices=['pkl', 'petr'], default='pkl', help='annotation format')
    parser.add_argument('--test', action='store_false', dest='train', help='whether to save train or test annotations (for \'petr\' format only)')
    parser.add_argument('parsed_data', help='output file with parsed data file (in Pickle format)')
    parser.set_defaults(train=True)
    args = parser.parse_args()
    main(args)
