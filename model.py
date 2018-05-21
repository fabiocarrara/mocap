import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class MotionModel(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, dropout=0.5, bidirectional=True, stack=1, layers=1, embed=0):
        super(MotionModel, self).__init__()
        self.in_size = in_size
        self.bidirectional = bidirectional
        rnn_hidden = hidden // 2 if bidirectional else hidden

        self.embed = None
        if embed > 0:
            self.embed = nn.Sequential(
                nn.Linear(in_size, embed),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif embed < 0:
            embed = - embed
            self.embed = LimbRelations(embed)
            embed = in_size + 10 * embed  # 10 are the possible couples between limbs

        self.lstm = nn.LSTM(embed if embed > 0 else in_size, rnn_hidden,
                            num_layers=stack,
                            bidirectional=bidirectional,
                            dropout=dropout)
        classifier_layers = []
        for _ in range(layers - 1):
            classifier_layers.append(nn.Linear(hidden, hidden))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(hidden, out_size))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward_lstm(self, input):
        input = input.view(-1, self.in_size)  # seq, data
        input = self.embed(input) if self.embed is not None else input  # embed all data in the sequence
        input = input.unsqueeze(1)  # seq, batch, data
        return self.lstm(input)

    def forward(self, input):
        outputs, hidden = self.forward_lstm(input)
        last_out = outputs[-1]  # this is the last hidden state (last timestep) of the last stacked layer
        return self.classifier(last_out)

    def segment(self, input):
        outputs, hidden = self.forward_lstm(input)
        return self.classifier(outputs)

    def steps_forward(self, input):
        outputs, hidden = self.forward_lstm(input)

        ''' for bidirectional models, we have to reverse the hidden states
            of the second direction in order to have the combined hidden state
            for each time step
        '''
        if self.bidirectional:
            seq_len = input.shape[0]
            outputs = outputs.view(seq_len, 2, -1)
            idx = torch.LongTensor([i for i in range(seq_len - 1, -1, -1)])
            if outputs.is_cuda:
                idx = idx.cuda()
            idx = Variable(idx, requires_grad=False)
            outputs[:, 1] = outputs[:, 1].index_select(0, idx)
            outputs = outputs.view(seq_len, -1)

        return self.classifier(outputs)

    def extract(self, input):
        outputs, hidden = self.forward_lstm(input)
        last_out = hidden[1].view(1, -1)
        return last_out


class LimbRelations(nn.Module):
    def __init__(self, rel_size):
        super(LimbRelations, self).__init__()
        self.rel_size = rel_size
        self.idx1, self.idx2 = np.triu_indices(5, 1)
        self.fc = nn.Sequential(
            nn.Linear(2 * (4 * 3), rel_size),  # 2 * (4 joins per part * 3 coordinates)
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x: input shape should be (T, N), meaning Timestamp and rest (N = Flatten Joint + Coordinate)
        :return: something
        """
        # for NTU:
        # 0-3: central spine
        # 4-7: left arm
        # 8-11: right arm
        # 12-15: left leg
        # 16-19: right arm
        # 20+: fingers

        xx = x.view(-1, 25, 3)  # T x 25 x 3 (return to joints x coordinates)
        limbs = xx[:, :20, :].contiguous().view(-1, 5, 12)  # T x 5 x (4x3) = T x 5 x 12 (5 limbs, 4 3D point each)
        limb1 = limbs.unsqueeze(2).expand(-1, -1, 5, -1)  # T x 5 x 5 x 12
        limb2 = limbs.unsqueeze(1).expand(-1, 5, -1, -1)  # T x 5 x 5 x 12
        couples = torch.cat([limb1, limb2], -1)  # T x 5 x 5 x 24
        unique_couples = couples[:, self.idx1, self.idx2, :]  # T x 10 x 24
        x_rels = self.fc(unique_couples)  # T x 10 x rel_size
        x_rels = x_rels.view(-1, 10 * self.rel_size)  # T x (10 * rel_size)
        return torch.cat((x, x_rels), 1)  # T x (N + 10 * rel_size)
