import pickle
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, data_file, fps=30, offset='none'):
        with open(data_file, 'rb') as in_file:
            self.data = pickle.load(in_file)

        descriptions = pd.read_csv('data/ntu_action_descriptions.txt', sep=';', index_col=0,
                                   names=['Label', 'Description'])

        assert offset in ('none', 'random', 'all'), "offset must be one of: none, random, all"

        self.skip = round(30.0 / fps)  # XXX which is NTU default frame rate?
        self.offset = offset

        self.actions = Counter(s['action'] for s in self.data)

        self.action_labels = sorted(self.actions.keys())
        self.action_descriptions = descriptions.loc[self.action_labels, 'Description'].values
        self.action_id_to_ix = {a: i for i, a in enumerate(self.action_labels)}
        self.n_actions = len(self.actions)

        print('Loaded: {} (Samples: {})'.format(data_file, len(self.data)))
        # pprint(self.actions)

    def __len__(self):
        if self.offset == 'all':
            return len(self.data) * self.skip
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index % len(self.data)]
        offset = 0
        if self.offset == 'random':
            offset = torch.IntTensor(1).random_(self.skip)[0]
        elif self.offset == 'all':
            offset = index // len(self.data)
        sequence = sample['data'][offset::self.skip, ...]
        x = torch.from_numpy(sequence)
        y = self.action_id_to_ix[sample['action']]
        return x, y

    def get_weights(self, action_balance=None):
        n_samples = float(len(self.data))
        if action_balance is None:
            weights = [n_samples / self.actions[s['action_id']] for s in self.data]
        else:
            weights = [action_balance[self.action_id_to_ix[s['action_id']]] for s in self.data]
        return weights

    def get_data_size(self):
        in_size = self.data[0]['data'][0].size
        out_size = len(self.actions)
        return in_size, out_size


if __name__ == '__main__':
    dataset = MotionDataset('data/NTU/NTU-CS-objects-annotations_filtered0.9GT-coords_normPOS-train.pkl')
    print(dataset.actions)
    print(len(dataset.actions))
