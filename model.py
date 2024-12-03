import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class BasicBlock(nn.Module):
    """
    ----------
    basic block
    ----------
    """
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(12, momentum=0.99, eps=0.001)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)
        init.constant_(self.batch_norm.weight, 1)
        init.constant_(self.batch_norm.bias, 0)

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = self.conv3(y)
        z = y + x  # Residual connection
        z = self.batch_norm(z)
        z = F.elu(z)
        out = self.pool(z)
        return out

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.block1 = BasicBlock(1)  # Input channels = 1
        self.pool1 = nn.MaxPool2d(kernel_size=5)
        self.block2 = BasicBlock(12)  # Output of block1 = 12 channels
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(12 * (480 // 20) * (480 // 20), 10)  # Adjust dimensions after pooling
        self.output_layer = nn.Linear(10, 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        init.kaiming_normal_(self.dense1.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dense1.bias, 0)
        init.xavier_normal_(self.output_layer.weight)
        init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.elu(self.dense1(x))
        out = self.output_layer(x)
        return out
