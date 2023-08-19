import random
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F

class UnetCascade(nn.Module):
    def __init__(self, in_chans, out_chans, num_of_unet):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_of_unet = num_of_unet
        self.unet_list = nn.ModuleList()
#         self.batchNorm = nn.BatchNorm2d(1)
#        self.consistency = consistency
        for i in range(num_of_unet):
            self.unet_list.append(AttentionGUnet(self.in_chans,self.out_chans))
    def forward(self, input, grappa):
        output = input
        for unet in self.unet_list:
            prev_output = output
            output = unet(output, grappa)
#            output = self.consistency*(prev_output)+(1-self.consistency)*output
        return output

class Unet(nn.Module): 
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 16,
        num_pool_layers: int = 5,
        drop_prob: float = 0.2,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        
        return output

class AttentionUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
    
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_attention = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_attention.append(Up_Attention(ch * 2 , ch, 64))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        self.up_attention.append(Up_Attention(ch * 2 , ch, 64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        stack = []
        output = image
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        # apply up-sampling layers
        for transpose_conv, conv, up_att in zip(self.up_transpose_conv, self.up_conv, self.up_attention):
            downsample_layer = stack.pop()
            downsample_layer = up_att(output, downsample_layer)
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output

    
class AttentionGUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.1,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.res_param = (nn.Parameter(torch.tensor(0.)))
#        self.avg_param = nn.Parameter(torch.tensor(0.))
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_attention = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_attention.append(Up_Attention(ch * 2 , ch, 64))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        self.up_attention.append(Up_Attention(ch * 2 , ch, 64))
    def norm(self, x):
        b, h, w = x.shape
        x = x.reshape(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, image_1: torch.Tensor, image_2, grappa) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        stack = []
        image_input = image_1
        image, mean, std = self.norm(image_1)
        grappa = (grappa-mean)/std
        image_2 = (image_2 - mean)/std
        output = torch.cat([image.unsqueeze(1), image_2.unsqueeze(1), grappa.unsqueeze(1)], dim = 1)
        #image transform
        randVal = random.random();
        if randVal < 1/6:
            output = transforms.functional.rotate(output, 90)
        elif randVal < 2/6:
            output = transforms.functional.rotate(output, 180)
        elif randVal < 3/6:
            output = transforms.functional.rotate(output, 270)
        elif randVal < 4/6:
            output = transforms.functional.hflip(output)
        elif randVal < 5/6:
            output = transforms.functional.vflip(output)
        
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        
        output = self.conv(output)
        # apply up-sampling layers
        for transpose_conv, conv, up_att in zip(self.up_transpose_conv, self.up_conv, self.up_attention):
            downsample_layer = stack.pop()
            downsample_layer = up_att(output, downsample_layer)
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
#         print(output.shape)
#        self.avg = nn.Sigmoid()(self.avg_param)
#        output = output[:,0,...]*self.avg+output[:,1,...]*(1-self.avg)
        
        #image invert
        if randVal < 1/6:
            output = transforms.functional.rotate(output, 270)
        elif randVal < 2/6:
            output = transforms.functional.rotate(output, 180)
        elif randVal < 3/6:
            output = transforms.functional.rotate(output, 90)
        elif randVal < 4/6:
            output = transforms.functional.hflip(output)
        elif randVal < 5/6:
            output = transforms.functional.vflip(output)
            
        output = self.unnorm(output, mean, std)
#         print(self.res_param)
        self.res = nn.Sigmoid()(self.res_param)
        output = self.res * output + image_input * (1-self.res)
        return output.squeeze(1)


"""
class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 16)
        self.linear_block = nn.Sequential(
            nn.Linear(576, 576),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Linear(576, 576),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
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

"""
"""
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
        
"""
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    
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