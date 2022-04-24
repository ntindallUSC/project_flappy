from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        self.convLayer1 = nn.Conv2d(
            in_channels = 4, 
            out_channels = 32, 
            kernel_size = 8, 
            stride = 4
        )
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.convLayer2 = nn.Conv2d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 4, 
            stride = 2
        )
        self.batchNorm2 = nn.BatchNorm2d(64)

        self.convLayer3 = nn.Conv2d(
            in_channels = 64, 
            out_channels = 64, 
            kernel_size = 3, 
            stride = 1
        )
        self.batchNorm3 = nn.BatchNorm2d(64)

        self.feedForward1 = nn.Linear(7 * 7 * 64, 512)
        self.feedForward2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        val = F.relu(self.batchNorm1(self.convLayer1(x)))
        val = F.relu(self.batchNorm2(self.convLayer2(val)))
        val = F.relu(self.batchNorm3(self.convLayer3(val)))

        val = val.view(val.size(0), -1)

        val = F.relu(self.feedForward1(val))
        val = self.feedForward2(val)
        
        return val

