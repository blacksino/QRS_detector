import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, seq_length, n_hidden, n_layer):
        super(BiLSTM, self).__init__()
        self.len = seq_length
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.n_hidden, bidirectional=True, num_layers=self.n_layer,
                            batch_first=True)
        self.fc = nn.Linear((self.lstm.bidirectional + 1) * self.lstm.hidden_size, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        self.lstm.flatten_parameters()
        # input shape [B,Length,dim]
        batch_size = x.shape[0]

        hidden_state = torch.randn(self.n_layer *
                                   (self.lstm.bidirectional + 1), batch_size,
                                   self.n_hidden).cuda()  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(self.n_layer *
                                 (self.lstm.bidirectional + 1), batch_size,
                                 self.n_hidden).cuda()

        feat, (_, _) = self.lstm(x, (hidden_state, cell_state))
        feat = self.fc(feat)
        pred = self.activate(feat)
        return pred
