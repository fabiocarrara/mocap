import pickle
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, data_file, at_least=None, keep_actions=None, fps=120, offset='none'):
        with open(data_file, 'rb') as in_file:
            self.data = pickle.load(in_file)

        assert offset in ('none', 'random', 'all'), "offset must be one of: none, random, all"

        self.skip = round(120.0 / fps)
        self.offset = offset
        self.actions = Counter(s['action_id'] for s in self.data)

        if keep_actions:
            self.actions = filter(lambda x: x[0] in keep_actions, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least:
            self.actions = filter(lambda x: x[1] >= at_least, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least or keep_actions:
            self.data = list(filter(lambda x: x['action_id'] in self.actions, self.data))

        self.action_labels = sorted(self.actions.keys())
        descriptions = pd.read_csv('data/action_descriptions.txt', sep=';', index_col=0, names=['Label', 'Description'])
        descriptions = descriptions.replace({'hdm05_spec_': ''}, regex=True)
        self.action_descriptions = descriptions.loc[self.action_labels, 'Description'].values
        self.action_id_to_ix = {a: i for i, a in enumerate(self.action_labels)}

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
        y = self.action_id_to_ix[sample['action_id']]
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