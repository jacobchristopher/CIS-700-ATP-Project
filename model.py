import torch
import torch.nn as nn

"""

Siamese Models

1. Simple Transformer  (Submodule)
2. Siamese Transformer (Our Approach)
3. Siamese CNN + LSTM  (Baseline)

"""

class Transformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers):
        super(Transformer, self).__init__()
        self.module = nn.Transformer(d_model=256, nhead=nhead, num_encoder_layers=num_encoder_layers)

    def forward(self, x, y):
        # Simple forward pass implementaiton
        output = self.module(x, y)
        return output



class SiameseTransformer(nn.Module):
    def __init__(self, in_dim, nhead=16, num_encoder_layers=12):
        super(SiameseTransformer, self).__init__()
        
        # Transformer layers
        self.conj_transformer = Transformer(nhead, num_encoder_layers)
        self.step_transformer = Transformer(nhead, num_encoder_layers)
        
        # Fully connected layers
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(in_dim, 2)


    def forward(self, conjecture, step, con_label, step_label):
        # Forward pass through transformer modules
        conj_out = self.conj_transformer(conjecture, con_label)
        step_out = self.step_transformer(step, step_label)
        x = torch.cat([conj_out, step_out], dim=1)

        # Feed into fully connected layer
        x = self.fc(x)
        x = self.activation(x)
        return x
    


class SiameseCNNLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SiameseCNNLSTM, self).__init__()

        # Convolutional layers
        self.conv1a = torch.conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1b = torch.conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1c = torch.conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)
        self.conv1d = torch.conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=7)

        # Activation function
        self.maxpool = torch.max_pool1d(kernel_size=3, stride=3)

        #LSTM
        self.lstm = nn.LSTM(in_dim, hidden_dim, in_dim)

        # Fully connected layers
        self.activation = nn.Sigmoid()
        self.fc = nn.Linear(in_dim, 2)


    def forward(self, conjecture, step):
        # Conjecture forward pass
        conjecture = self.conv1a(conjecture)
        conjecture = self.maxpool(conjecture)
        conjecture = self.conv1b(conjecture)
        conjecture = self.maxpool(conjecture)

        # Step forward pass
        step = self.conv1c(step)
        step = self.maxpool(step)
        step = self.conv1d(step)
        step = self.maxpool(step)

        x = torch.concat((conjecture, step))

        # Feed into fully connected layer
        x = self.fc(x)
        x = self.activation(x)
        return x
