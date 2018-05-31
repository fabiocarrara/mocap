import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class MotionModel(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, dropout=0.5, bidirectional=True, stack=1, layers=1,
                 embed_layers=0, rel_dim=0, **kwargs):
        super(MotionModel, self).__init__()
        self.in_size = in_size
        self.bidirectional = bidirectional
        rnn_hidden = hidden // 2 if bidirectional else hidden

        self.embed = SequenceEmbedding(in_size, embed_layers, rel_dim, dropout)
        in_size = self.embed.out_size

        self.lstm = nn.LSTM(in_size, rnn_hidden, num_layers=stack, bidirectional=bidirectional, dropout=dropout)

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


class SequenceEmbedding(nn.Module):
    def __init__(self, in_size, embed_layers, relation_dim, dropout):
        super(SequenceEmbedding, self).__init__()

        pose_embed = []
        for _ in range(embed_layers):
            pose_embed.append(nn.Linear(in_size, in_size, bias=False))
            pose_embed.append(nn.ReLU())
            pose_embed.append(nn.Dropout(dropout))

        self.embed = nn.Sequential(*pose_embed) if pose_embed else None

        self.limb_dim = 4 * 3  # (4 joins in a limb * 3 coordinates)
        self.n_limbs = 6
        self.n_limb_couples = self.n_limbs * (self.n_limbs - 1) // 2

        self.fc = nn.Sequential(
            nn.Linear(2 * self.limb_dim + self.n_limb_couples, relation_dim),  # 2 * 12 + 15 dims for one-hot
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if relation_dim else None

        self.idx1, self.idx2 = np.triu_indices(self.n_limbs, 1)
        self.register_buffer('onehot', torch.eye(self.n_limb_couples))
        self.out_size = ((in_size + self.n_limb_couples * relation_dim)  # 15 are the possible couples between limbs
                         if relation_dim else in_size)

    def forward(self, x):
        """
        :param x: input shape should be (T, N), meaning Timestamp and rest (N = Flatten Joint + Coordinate)
        :return: something
        """

        x = self.embed(x) + x if self.embed else x

        if self.fc:
            # for NTU:
            # 0-3: central spine
            # 4-7: left arm
            # 8-11: right arm
            # 12-15: left leg
            # 16-19: right arm
            # 20: chest
            # 21-24: fingers

            # all but chest
            limb_idx = [i for i in range(25) if i != 20]

            xx = x.view(-1, 25, 3)  # T x 25 x 3 (return to joints x coordinates)
            seq_len = xx.shape[0]
            # GET THE LIMB PARTS
            limbs = xx[:, limb_idx, :].contiguous().view(-1, self.n_limbs, self.limb_dim)  # T x 6 x limb_size (6 limbs, 4 3D point each)

            # MAKE ALL POSSIBLE COMBINATIONS
            limb1 = limbs.unsqueeze(2).expand(-1, -1, self.n_limbs, -1)  # T x 6 x 6 x limb_size
            limb2 = limbs.unsqueeze(1).expand(-1, self.n_limbs, -1, -1)  # T x 6 x 6 x limb_size
            couples = torch.cat([limb1, limb2], -1)  # T x 6 x 6 x (2 * limb_size)
            relations = couples[:, self.idx1, self.idx2, :]  # T x 10 x (2 * limb_size)

            # (OPTIONALLY) ADD ONE-HOT VECTORS TO IDENTIFY RELATIONS
            onehot = Variable(self.onehot, requires_grad=False)
            onehot = onehot.unsqueeze(0).expand(seq_len, -1, -1)
            relations = torch.cat((relations, onehot), 2)  # T x 10 x (2 * limb_size + 10)

            relations = self.fc(relations) if self.fc else relations  # T x 10 x rel_size
            relations = relations.view(seq_len, -1)  # T x (10 * rel_size)
            x = torch.cat((x, relations), 1)  # T x (N + 10 * rel_size)

        return x
