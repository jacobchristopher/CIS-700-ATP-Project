import torch
import torch.nn as nn
import utils
from dataset import encode

"""

Siamese Models

1. Simple Transformer  (Submodule)
2. Siamese Transformer (Our Approach)
3. Siamese CNN + LSTM  (Baseline)

"""

class Transformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers):
        super(Transformer, self).__init__()

        self.positional_encoder = utils.trigonometric_positional_encoder(256, 256, 10000, "cuda")
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.trf = nn.TransformerEncoder(layer, num_encoder_layers)
        self.attn_mask = torch.triu(torch.full((256, 256), float('-inf'), device="cuda"), diagonal=1)

    def forward(self, x):
        # x = self.positional_encoder(x)
        output = self.trf(x)
        return output


class SiameseTransformer(nn.Module):
    def __init__(self, in_dim, nhead=16, num_encoder_layers=12):
        super(SiameseTransformer, self).__init__()
        
        # Transformer layers
        self.conj_transformer = Transformer(nhead, num_encoder_layers)
        self.step_transformer = Transformer(nhead, num_encoder_layers)
        
        # Fully connected layers
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(in_dim*2, 2)
        self.maxpool = nn.MaxPool1d(kernel_size=in_dim)


    def forward(self, conjecture, step):

        # Forward pass through transformer modules
        conj_out = self.conj_transformer(conjecture)
        step_out = self.step_transformer(step)
        x = torch.cat([conj_out, step_out], dim=1)

        # Feed into fully connected layer
        x = self.fc(x)
        x = self.activation(x)
        x = self.maxpool(x.transpose(0, 1)).squeeze()
        return x
    


class SiameseCNNLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SiameseCNNLSTM, self).__init__()

        # Convolutional layers
        self.conv1a = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1b = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1c = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)

        # Activation function
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=256, stride=256)

        #LSTM
        self.lstm = nn.LSTM(in_dim, hidden_dim, in_dim)

        # Fully connected layers
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(50, 2)


    def forward(self, conjecture, step):
        # Conjecture forward pass
        conjecture = self.conv1a(conjecture)
        conjecture = self.maxpool1(conjecture)
        conjecture = self.conv1b(conjecture)
        conjecture = self.maxpool1(conjecture)

        # Step forward pass
        step = self.conv1c(step)
        step = self.maxpool1(step)
        step = self.conv1d(step)
        step = self.maxpool1(step)

        x = torch.cat([conjecture, step], dim=1)

        # Feed into fully connected layer
        x = self.fc(x)
        x = self.activation(x).permute(1, 0)
        x = self.maxpool2(x.unsqueeze(0)).squeeze(0).squeeze(0).squeeze(1)
        return x
    

