import torch
import torch.nn as nn

"""

Siamese Transformer Module

"""

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # TODO: Add customization here
        self.module = nn.Transformer(nhead=16, num_encoder_layers=12)

    def forward(self, x):
        # Simple forward pass implementaiton
        output = self.module(x)
        return output



class SiameseTransformer(nn.Module):
    def __init__(self, in_dim):
        super(SiameseTransformer, self).__init__()
        
        # Transformer layers
        self.conj_transformer = Transformer()
        self.step_transformer = Transformer()
        
        # Fully connected layers
        self.activation = nn.ReLU()
        self.fc = nn.Linear(in_dim, 2)


    def forward(self, conjecture, step):
        # Forward pass through transformer modules
        conj_out = self.conj_transformer(conjecture)
        step_out = self.step_transformer(step)
        x = torch.concat((conj_out, step_out))

        # Feed into fully connected layer
        x = self.activation(x)
        x = self.fc(x)
        return x
