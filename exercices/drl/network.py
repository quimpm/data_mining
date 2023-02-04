import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    """Actor (Policy) Network Model."""

    def __init__(self, state_size, action_size, seed=42, fc1_units=64, fc2_units=64):
        """
        Initialize parameters and build model.
        
        :param state_size: (int) # state
        :param action_size: (int) # action
        :param seed: (int) Random 
        :param fc1_units: (int) First hidden layer size
        :param fc2_units: (int) Second hidden layer size
        """
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvNN(nn.Module):

    def __init__(self, in_channels, out_shape):
        super(ConvNN, self).__init__()
        self.out_shape = out_shape
        self.fully_con = nn.Sequential(
            self.block(in_channels, 64),
            self.block(64, 128)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.out = nn.Sequential(
            nn.Linear(128, out_shape),
        )
    
    def block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3,3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 3, 96, 96)
        x = self.fully_con(x)
        x = self.avg_pool(x)
        x = x.reshape(batch_size, -1)
        x = self.out(x)
        return x
