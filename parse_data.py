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
    header_regexp = r'.*\s(\d+)_(\d+)_(\d+)_(\d+)'
    matches = re.match(header_regexp, header)
    assert matches is not None, "Error parsing data header: %s" % header
    seq_id, action_id, start_frame, duration = map(int, matches.groups())
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
        seq_id=seq_id,
        action_id=action_id,
        start_frame=start_frame,
        duration=duration,
        data=data
    )
    return sequence


def main(args):
    sequences = get_sequences(args.data)
    parsed = [parse_sequence(seq) for seq in tqdm(sequences)]
    with open(args.parsed_data, 'wb') as outfile:
        pickle.dump(parsed, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('data', help='path to data file (in textual format)')
    parser.add_argument('parsed_data', help='output file with parsed data file (in Pickle format)')
    args = parser.parse_args()
    main(args)
