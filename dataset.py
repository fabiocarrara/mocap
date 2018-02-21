import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, data_file, at_least=None, keep_actions=None):
        with open(data_file, 'rb') as in_file:
            self.data = pickle.load(in_file)

        self.actions = Counter(s['action_id'] for s in self.data)

        if keep_actions:
            self.actions = filter(lambda x: x[0] in keep_actions, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least:
            self.actions = filter(lambda x: x[1] >= at_least, self.actions.items())
            self.actions = Counter(dict(self.actions))

        if at_least or keep_actions:
            self.data = list(filter(lambda x: x['action_id'] in self.actions, self.data))

        unique_actions = sorted(self.actions.keys())
        self.action_id_to_ix = {a: i for i, a in enumerate(unique_actions)}

        print('Loaded: {} (Samples: {})'.format(data_file, len(self.data)))
        # pprint(self.actions)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x = torch.from_numpy(sample['data'])
        y = self.action_id_to_ix[sample['action_id']]
        return x, y

    def get_weights(self):
        n_samples = float(len(self.data))
        weights = [n_samples / self.actions[s['action_id']] for s in self.data]
        return weights

    def get_data_size(self):
        in_size = self.data[0]['data'][0].size
        out_size = len(self.actions)
        return in_size, out_size