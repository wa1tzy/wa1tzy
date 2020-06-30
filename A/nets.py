import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from arc_net import Arc_Net


class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias=False):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    def forward(self,x):
        return self.cnn_layer(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvolutionalLayer(1,32,5,1,2),#28-5+4+1=28
            ConvolutionalLayer(32,32,5,1,2),#28
            nn.MaxPool2d(2,2),
            ConvolutionalLayer(32,64,5,1,2),#14
            ConvolutionalLayer(64,64,5,1,2),#14
            nn.MaxPool2d(2,2),
            ConvolutionalLayer(64,128,5,1,2),#7
            ConvolutionalLayer(128,128,5,1,2),#7
            nn.MaxPool2d(2,2)#3
        )
        self.feature = nn.Linear(128*3*3,2)
        self.arcsoftmax = Arc_Net()

    def forward(self,x):
        y_conv = self.conv_layer(x)
        y_conv = torch.reshape(y_conv,[-1,128*3*3])
        y_feature = self.feature(y_conv)
        y_output = torch.log(self.arcsoftmax(y_feature))

        return y_feature,y_output

    def visualize(self,feat,labels,epoh):
        # plt.ion()
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels==i,0],feat[labels==i,1],".",c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],loc="upper right")
        # plt.xlim(xmin=-5,xmax=5)
        # plt.ylim(ymin=-5,ymax=5)
        plt.title("epoh=%d" % epoh)
        plt.savefig("./images3/epoh=%d.jpg" % epoh)
        # plt.draw()
        # plt.pause(0.001)

