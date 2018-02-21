import argparse
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


def parse_sequence(lines):
    header = lines[0]
    header_regexp = r'.*\s((\d+)_(\d+)_(\d+)_(\d+))'
    matches = re.match(header_regexp, header)
    assert matches is not None, "Error parsing data header: %s" % header
    attributes = matches.groups()
    sample_id = attributes[0]
    seq_id, action_id, start_frame, duration = map(int, attributes[1:])
    lines = lines[2:]  # discard header

    def parse_line(line):
        values = re.split('; |, | ', line)
        values = map(float, values)
        values = np.fromiter(values, dtype=np.float32)
        values = values.reshape(-1, 3)
        return values

    data = [parse_line(line) for line in lines]
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


def get_ids_to_keep(split_file):
    id_regexp = r'.*/(\d+_\d+_\d+_\d+).*'

    def get_id(line):
        matches = re.match(id_regexp, line)
        return matches.groups()[0] if matches else None

    with open(split_file, 'rt') as f:
        ids = {get_id(line) for line in f}

    return ids


def main(args):
    sequences = get_sequences(args.data)
    parsed = (parse_sequence(seq) for seq in sequences)

    if args.split:
        ids_to_keep = get_ids_to_keep(args.split)
        parsed = filter(lambda x: x['id'] in ids_to_keep, parsed)

    parsed = list(tqdm(parsed))
    with open(args.parsed_data, 'wb') as outfile:
        pickle.dump(parsed, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('data', help='path to data file (in textual format)')
    parser.add_argument('-s', '--split', help='path to optional split file (in textual format)')
    parser.add_argument('parsed_data', help='output file with parsed data file (in Pickle format)')
    args = parser.parse_args()
    main(args)
