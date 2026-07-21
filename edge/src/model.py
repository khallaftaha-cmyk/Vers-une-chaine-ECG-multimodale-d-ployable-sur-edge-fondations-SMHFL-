"""CNN-1D + BiLSTM model for 12-lead ECG classification."""

import torch
import torch.nn as nn
from typing import List


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class CNN_BiLSTM(nn.Module):
    def __init__(self, num_leads=12, cnn_channels=[64, 128, 256], cnn_kernel_sizes=[7, 5, 3],
                 lstm_hidden_size=128, lstm_num_layers=2, num_classes=4, dropout=0.3):
        super(CNN_BiLSTM, self).__init__()

        # 1. Sequential CNN blocks
        blocks = []
        in_c = num_leads
        for out_c, k_size in zip(cnn_channels, cnn_kernel_sizes):
            blocks.append(CNNBlock(in_c, out_c, k_size, dropout))
            in_c = out_c
        self.cnn = nn.Sequential(*blocks)

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # 3. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 12, 5000)
        x = self.cnn(x)
        # Permute: (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)

        # BiLSTM
        # lstm_out: (batch, seq, 2*hidden) -- every timestep, both directions
        # hn:       (num_layers*2, batch, hidden) -- final hidden state per layer/direction
        lstm_out, (hn, cn) = self.lstm(x)

        # FIX: lstm_out[:, -1, :] mixes the forward direction's fully-accumulated
        # final state with the backward direction's state at the LAST timestep --
        # but the backward pass only just started there, so it's seen one input,
        # not the whole sequence in reverse. Use hn directly instead: it holds the
        # true final state for each direction after processing the full sequence.
        forward_final = hn[-2]   # last layer, forward direction: (batch, hidden)
        backward_final = hn[-1]  # last layer, backward direction: (batch, hidden)
        x = torch.cat([forward_final, backward_final], dim=1)  # (batch, 2*hidden)

        # Classifier
        x = self.classifier(x)
        return x


def build_model(config: dict) -> CNN_BiLSTM:
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    return CNN_BiLSTM(
        num_leads=data_cfg.get('num_leads', 12),
        cnn_channels=model_cfg.get('cnn_channels', [64, 128, 256]),
        cnn_kernel_sizes=model_cfg.get('cnn_kernel_sizes', [7, 5, 3]),
        lstm_hidden_size=model_cfg.get('lstm_hidden_size', 128),
        lstm_num_layers=model_cfg.get('lstm_num_layers', 2),
        num_classes=model_cfg.get('num_classes', 4),
        dropout=model_cfg.get('dropout', 0.3)
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Dummy config
    config = {
        'model': {
            'cnn_channels': [64, 128, 256],
            'cnn_kernel_sizes': [7, 5, 3],
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'dropout': 0.3,
            'num_classes': 4
        },
        'data': {
            'num_leads': 12
        }
    }

    model = build_model(config)
    print("Model Architecture:")
    print(model)
    print(f"Total Trainable Parameters: {count_parameters(model)}")

    x = torch.randn(1, 12, 5000)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")