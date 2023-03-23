import torch
import torch.nn as nn
import copy
import random

    
class encodingUnit(nn.Module):
    
    def __init__(self, compressionProps, prevLayerProps):
        super(encodingUnit, self).__init__()
        
        self.compressionProps = compressionProps 
        
        
        prevLayerChannels = prevLayerProps['PrevLayerOutChannel']
        compressedChannelNum = int(prevLayerChannels / compressionProps['feature_compression_factor'])
        
        self.scaler = 15
        prune_inn = self.scaler * torch.rand((1,compressedChannelNum,1,1))
        self.prune_filter = nn.Parameter(prune_inn)
        self.count = 0
            
        kernelHeight = 2 + compressionProps['resolution_compression_factor']
        kernelWidth = 2 + compressionProps['resolution_compression_factor']
        strideVal = compressionProps['resolution_compression_factor']
        paddingVal = compressionProps['resolution_compression_factor']
        
        self.convIn1 = nn.Conv2d(in_channels=prevLayerChannels,
                                    out_channels=compressedChannelNum,
                                    kernel_size=(kernelHeight,kernelWidth),
                                    stride = strideVal,
                                    padding = paddingVal)
        self.batchNormIn = nn.BatchNorm2d(compressedChannelNum, momentum=0.03, eps=1E-4)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, prune=True):   
        x = torch.sigmoid(self.batchNormIn(self.convIn1(x)))
        
        #if self.count == 0:
        #    channel_len = len(x.shape[2:])
        #    for entry in range(channel_len):
        #        self.prune_filter = nn.Parameter(self.prune_filter.unsqueeze(-1))
        #    self.count += 1
            
        
        if prune == True:
            self.prune_filter.requires_grad_(True) #convert to variable
            filter_relu = self.sigmoid(self.prune_filter)
            x = torch.mul(x,filter_relu) 
            return x, self.prune_filter
        else:
            self.prune_filter.requires_grad_(False) #convert back to constant
            filter_relu = self.sigmoid(self.prune_filter)
            x = torch.mul(x,filter_relu) 
            return x, self.prune_filter
        
    def resetPrune(self):
        self.prune_filter.requires_grad_(False)
        count = 0
        for entry in self.prune_filter[0]:
            if entry[0] > 0.9: 
                self.prune_filter[0][count] = random.uniform(0,self.scaler)
            else:
                self.prune_filter[0][count] = -self.scaler 
            count += 1
        self.prune_filter.requires_grad_(True)
        
        
    def resetdePrune(self):
        self.prune_filter.requires_grad_(False)
        count = 0
        for entry in self.prune_filter[0]:
            if entry[0] < 0.9: 
                self.prune_filter[0][count] = random.uniform(0,self.scaler)
            else:
                self.prune_filter[0][count] = self.scaler 
            count += 1
        self.prune_filter.requires_grad_(True)
        
        
        
class decodingUnit(nn.Module):
    
    def __init__(self, compressionProps, prevLayerProps):
        super(decodingUnit, self).__init__()
        
        self.compressionProps = compressionProps 
        
        prevLayerChannels = prevLayerProps['PrevLayerOutChannel']
        compressedChannelNum = int(prevLayerChannels / compressionProps['feature_compression_factor'])
            
        kernelHeight = 2 + compressionProps['resolution_compression_factor']
        kernelWidth = 2 + compressionProps['resolution_compression_factor']
        strideVal = compressionProps['resolution_compression_factor']
        paddingVal = compressionProps['resolution_compression_factor']
        
        self.convOut1 = nn.ConvTranspose2d(in_channels=compressedChannelNum,
                                        out_channels=prevLayerChannels,
                                        kernel_size=(kernelHeight,kernelWidth),
                                        stride = strideVal,
                                        padding  = paddingVal)
        
        self.batchnormOut = nn.BatchNorm2d(prevLayerChannels, momentum=0.03, eps=1E-4)

    def forward(self,x):
        x = torch.relu(self.batchnormOut(self.convOut1(x)))
        return x