# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch

        
class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class SimpleConv(Module):
    def __init__(self, enc_channels=(3, 4, 8, 16, 32, 48,64),
                 dec_channels=(64, 48, 32, 24, 16, 8, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            x = block(x)

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    
class HourGlass(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        self.pool = MaxPool2d(2)

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.upconv_blocks = ModuleList(
            [ConvTranspose2d(dec_channels[i], dec_channels[i + 1], 2, 2)
             for i in range(len(dec_channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i + 1], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # prepare upconvolutions
    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            x = self.pool(x)

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconv_blocks[i](x)
            x = self.dec_blocks[i](x)
        #
        # return the final decoder output
        return x


class UNet(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # prepare pooling
        self.pool = MaxPool2d(2)

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.enc_channels = enc_channels
        self.upconv_blocks = ModuleList(
            [ConvTranspose2d(dec_channels[i], dec_channels[i + 1], 2, 2)
             for i in range(len(dec_channels) - 1)])
        # decoder blocks need to accept concatenated channels from skip connections
        # store.pop() gives us enc_channels in reverse: [-(i+1)]
        self.dec_blocks = ModuleList([
            Block(dec_channels[i + 1] + enc_channels[-(i + 1)], dec_channels[i + 1])
            for i in range(len(dec_channels) - 1)
        ])
        # prepare upconvolutions
    
    def forward(self, x):
        store = []
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            store.append(x)
            x = self.pool(x)
        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            up_conv_curr = self.upconv_blocks[i](x)
            curr_store = store.pop()
            x = torch.cat([up_conv_curr, curr_store], dim=1)
            x = self.dec_blocks[i](x)
        #
        # return the final decoder output
        return x