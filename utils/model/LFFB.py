import torch
from torch import nn
from torch.nn import functional as F


class LFFB(nn.Module):

    def __init__(self, cascade):
        super().__init__()
        self.cascade = cascade
        self.conv3 = ConvBlock(1,1,kernel_size=3)
        self.conv5 = ConvBlock(1,1,kernel_size=5)
        self.conv7 = ConvBlock(1,1,kernel_size=7)
        self.conv13 = nn.Conv2d(2, 1, kernel_size = 1, padding="same")
        self.conv35 = nn.Conv2d(2, 1, kernel_size = 1, padding="same")
        self.conv57 = nn.Conv2d(2, 1, kernel_size = 1, padding="same")
    def forward(self, input):
        output = input 
        for i in range(cascade):
            input = output
            output3 = self.conv3(input)
            output5 = self.conv5(input)
            output7 = self.conv7(input)
            concat1 = torch.stack([input, output3], axis = 1)
            coutput3 = self.conv13(concat1)
            concat2 = torch.stack([coutput3, output5], axis = 1)
            coutput5 = self.conv35(concat2)
            concat3 = torch.stack([coutput5, output7], axis = 1)
            coutput7 = self.conv35(concat3)
            output = (coutput7+input)/2
        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, kernel_size):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size, padding="same"),
        )

    def forward(self, x):
        return self.layers(x)
