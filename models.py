"""
Models class

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseline_ff(nn.Module):
    """
    Baseline model for the Long Covid Modeling

    Trains a feed forward network from scratch

    Args:
        hidden_size,
        drop_prob
    """
    
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(baseline_ff, self).__init__()

        self.fc1 = nn.Linear(23074, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        # forward through linear layers
        out = self.fc1(x)
        out = self.relu1(out)      
        out = self.relu2(self.fc2(out))  
        out = self.dropout(out)
        
        out = self.fc3(out)

        return out

class ff_pca(nn.Module):
    """
    Baseline model for the Long Covid Modeling

    Trains a feed forward network from scratch

    Args:
        hidden_size,
        drop_prob
    """
    
    def __init__(self, hidden_size, drop_prob = 0.1):
        super(ff_pca, self).__init__()

        self.fc1 = nn.Linear(38, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x):

        # forward through linear layers
        out = self.fc1(x)
        out = self.relu1(out)      
        out = self.dropout1(out)
        out = self.relu2(self.fc2(out))  
        out = self.dropout2(out)
        
        out = self.fc3(out)

        return out