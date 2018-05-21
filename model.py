import torch
from torch import nn
from torch.autograd import Variable


class MotionModel(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, dropout=0.5, bidirectional=True, stack=1, layers=1, embed=0):
        super(MotionModel, self).__init__()
        self.in_size = in_size
        self.bidirectional = bidirectional
        rnn_hidden = hidden // 2 if bidirectional else hidden

        self.embed = nn.Sequential(
                        nn.Linear(in_size, embed),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                     ) if embed > 0 else None

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
