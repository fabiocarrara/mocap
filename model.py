from torch import nn


class MotionModel(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, dropout=0.5, bidirectional=True, stack=1, layers=1, embed=0):
        super(MotionModel, self).__init__()
        self.in_size = in_size
        rnn_hidden = hidden // 2 if bidirectional else hidden

        self.embed = nn.Sequential(
                        nn.Linear(in_size, embed),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                     ) if embed > 0 else None

        self.lstm = nn.LSTM(embed if embed > 0 else in_size, rnn_hidden, bidirectional=bidirectional)
        self.stack = [nn.LSTM(rnn_hidden, rnn_hidden, bidirectional=bidirectional) for _ in range(stack - 1)]
        classifier_layers = []
        for _ in range(layers - 1):
            classifier_layers.append(nn.Linear(hidden, hidden))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(hidden, out_size))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, input):
        input = input.view(-1, self.in_size)  # seq, data
        if self.embed is not None:
            input = self.embed(input)  # embed all data in the sequence

        input = input.unsqueeze(1)  # seq, batch, data
        outputs, hidden = self.lstm(input)
        for stack_lstm in self.stack:
            outputs, hidden = stack_lstm(outputs)
        last_out = hidden[1].view(1, -1)
        return self.classifier(last_out)

    def extract(self, input):
        input = input.view(-1, self.in_size)  # seq, data
        if self.embed is not None:
            input = self.embed(input)  # embed all data in the sequence

        input = input.unsqueeze(1)  # seq, batch, data
        outputs, hidden = self.lstm(input)
        for stack_lstm in self.stack:
            outputs, hidden = stack_lstm(outputs)
        last_out = hidden[1].view(1, -1)
        return last_out
