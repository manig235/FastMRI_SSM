import torch
from torch import nn
from torch.nn import functional as F

class UnetCascade(nn.Module):
    def __init__(self, in_chans, out_chans, num_of_unet, consistency = 0.4):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_of_unet = num_of_unet
        self.unet_list = nn.ModuleList()
#         self.batchNorm = nn.BatchNorm2d(1)
        self.consistency = consistency
        for i in range(num_of_unet):
            self.unet_list.append(Unet(self.in_chans,self.out_chans))
    def forward(self, input, grappa):
        output = input
        for unet in self.unet_list:
            prev_output = output
            output = unet(output, grappa)
            output = self.consistency*(prev_output)+(1-self.consistency)*output
        return output

class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 16)
        """
        self.linear_block = nn.Sequential(
            nn.Linear(576, 576),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Linear(576, 576),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        """
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up4 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up1 = Up(32, 16)
        self.last_block = nn.Conv2d(16, out_chans, kernel_size=1)
#         self.linear_block = nn.Linear(576, 576)
    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input, grappa):
        input, mean, std = self.norm(input)
#         input = input.unsqueeze(1)
#         print(input.shape)
        grappa = (grappa-mean)/std
        input = torch.cat([input.unsqueeze(1), grappa.unsqueeze(1)], dim = 1)
        d1 = self.first_block(input)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
#         print(d4.shape)
        d5 = self.down4(d4)
#         d5 = d5.reshape(-1,256,1,576)
#         c5 = self.linear_block(d5)
#         c5 = c5.reshape(-1,256,24,24)
        u4 = self.up4(d5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        output = self.last_block(u1)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output



class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)

class Up(nn.Module):
    def __init__(self, in_chans, out_chans, mid_chans = 32):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.mid_chans = mid_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.att = Up_Attention(in_chans, in_chans // 2, mid_chans)
        self.conv =nn.Sequential(ConvBlock(in_chans, out_chans))

    def forward(self, x, concat_input):
        concat_input = self.att(x, concat_input)
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)

class Up_Attention(nn.Module):

    def __init__(self, in_chans, out_chans, mid_chans = None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        if mid_chans is None:
            mid_chans = out_chans
        self.mid_chans = mid_chans
        self.up_input = nn.Conv2d(in_chans, mid_chans, kernel_size = 1, stride = 1)
        self.up_concat = nn.Conv2d(out_chans, mid_chans, kernel_size = 1, stride = 2)
        self.flat = nn.Conv2d(mid_chans, 1, kernel_size = 1)
        self.ReLU = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.up_sample = nn.Upsample(scale_factor = 2)
        
    def forward(self, x, concat_input):
        up_input = self.up_input(x)
        up_concat = self.up_concat(concat_input)
        add = up_input + up_concat
        add_act = self.ReLU(add)
        flat = self.flat(add_act)
        flat_sig = self.sig(flat)
        up_sam = self.up_sample(flat_sig)
        return up_sam * concat_input
        
        