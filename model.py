from torch import nn


class MotionModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MotionModel, self).__init__()
        self.in_size = in_size
        self.lstm = nn.LSTM(in_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        input = input.view(-1, 1, self.in_size)  # seq, batch, data
        _, hidden = self.lstm(input)
        last_out = hidden[1].view(1, -1)
        return self.classifier(last_out)
