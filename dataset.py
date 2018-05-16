import os
import pickle
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, data_file, at_least=None, keep_actions=None, fps=120, offset='none', mapper=None):
        with open(data_file, 'rb') as in_file:
            self.data = pickle.load(in_file)

        # check if we are dealing with annotated subsequences or multi-annotated sequences
        self.multi = 'annotations' in self.data[0]

        if mapper:
            descriptions = pd.read_csv(mapper, sep=';', index_col=0,
                                       names=['OldLabel', 'Label', 'Description'])
            for s in self.data:
                if self.multi:
                    for a in s['annotations']:
                        a['action_id'] = descriptions.loc[a['action_id'], 'Label']
                else:
                    s['action_id'] = descriptions.loc[s['action_id'], 'Label']

            descriptions = descriptions.groupby('Label').aggregate({'Description': lambda x: os.path.commonprefix(x.tolist())})

        else:
            descriptions = pd.read_csv('data/hdm05_action_descriptions.txt', sep=';', index_col=0,
                                       names=['Label', 'Description'])
            descriptions = descriptions.replace({'hdm05_spec_': ''}, regex=True)

        assert offset in ('none', 'random', 'all'), "offset must be one of: none, random, all"

        self.skip = round(120.0 / fps)
        self.offset = offset

        # for HDM05-15 we remove 'other' class (14)
        if self.multi:
            self.actions = Counter(a['action_id'] for s in self.data for a in s['annotations'] if mapper or a['action_id'] != 14)
        else:
            self.actions = Counter(s['action_id'] for s in self.data if mapper or s['action_id'] != 14)

        if keep_actions:
            self.actions = filter(lambda x: x[0] in keep_actions, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least:
            self.actions = filter(lambda x: x[1] >= at_least, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least or keep_actions:
            if self.multi:
                for s in self.data:
                    s['annotations'] = list(filter(lambda x: x['action_id'] in self.actions, s['annotations']))
            else:
                self.data = list(filter(lambda x: x['action_id'] in self.actions, self.data))

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
        if self.multi:
            n_samples = len(sample['data'])
            y = torch.zeros(n_samples, self.n_actions)
            for a in sample['annotations']:
                class_id = self.action_id_to_ix[a['action_id']]
                start = a['start_frame']
                end = a['start_frame'] + a['duration']
                y[start:end, class_id] = 1
            y = y[offset::self.skip, ...]
        else:
            y = self.action_id_to_ix[sample['action_id']]
        return x, y

    def get_weights(self, action_balance=None):
        assert not self.multi, "Cannot use get_weights() with multi-annotated sequences"
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



class NTUMotionDataset(Dataset):
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

    # dataset = MotionDataset('data/split2.pkl', fps=10, mapper='data/HDM05-category_mapper-130vs65.csv')
    # print(dataset.actions)
    # print(len(dataset.actions))

    dataset = NTUMotionDataset('data/NTU/NTU-CS-objects-annotations_filtered0.9GT-coords_normPOS-train.pkl')
    print(dataset.actions)
    print(len(dataset.actions))
