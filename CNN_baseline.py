# https://huggingface.co/datasets/mnist

from datasets import load_dataset

dataset = load_dataset("mnist")

import torch
from torch import nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvolutionalNeuralNetworkClass(nn.Module):
    """
        Convolutional Neural Network (CNN) Class
        __init__ 
        	xdim : 입력데이터의 차원 (채널, 행, 열)
            ksize : 커널의 사이즈 (행(열) or (행,열))
            cdims : 합성곱 후 outsize (1st convolution output size, 2st convolution output size, ...)
            hdims : 선형결합 후 outsize (1st layer output size, 2st layer output size, ...)
            ydim : 최종 output 클래스 개수
            USE_BATCHNORM : 배치 정규화 적용 여부
            
     	init_param - 파라미터 초기화
          	isinstance(x,y) # x가 y타입인지 확인함
          	nn.init.kaiming_normal_() # He 초기화를 수행하는 코드
          	nn.init.zeros_() # 스칼라0으로 채우는 코드
          	nn.init.constant_(tensor,value) # tensor를 value로 채우는 코드	
    """
    def __init__(self,name='cnn',
                 xdim=[1,28,28],
                 ksize=4,
                 cdims=[28,14,7],
                 hdims=[10],
                 ydim=10,
                 USE_BATCHNORM=False):
        super(ConvolutionalNeuralNetworkClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM

        self.layers = []
        prev_cdim = self.xdim[0]
        for cdim in self.cdims:
            self.layers.append(nn.Conv2d(in_channels  = prev_cdim, 
                                         out_channels = cdim,
                                         kernel_size  = self.ksize,
                                         stride=(1,1),
                                         padding = self.ksize//2)
                                         )
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            prev_cdim = cdim

        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim*(self.xdim[1]//(2**len(self.cdims)))*(self.xdim[2]//(2**len(self.cdims)))
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim,hdim,bias=True))
            self.layers.append(nn.ReLU(True))
            prev_hdim = hdim
        self.layers.append(nn.Linear(prev_hdim,self.ydim,bias=True))

        self.layers.append(nn.Softmax())

        self.net = nn.Sequential()
        for l_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)
        self.init_param()
        
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
    def forward(self,x):
        return self.net(x)

C = ConvolutionalNeuralNetworkClass(name='cnn',
                                    xdim=[1,28,28],
                                    ksize=3,
                                    cdims=[28,14,7],
                                    hdims=[10],
                                    ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(C.parameters(),lr=1e-3)

print(C)

from torchsummary import summary
summary(C, (1,28,28), batch_size = 64)