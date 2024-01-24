import torch.nn as nn
import torch

from models.compressorVGG import encodingUnit
from models.compressorVGG import decodingUnit



class VGG(nn.Module):
    def __init__(self, features, num_classes, init_weights=False, local = False):
        super(VGG, self).__init__()
        self.encoder = []
        self.decoder = []
        if local == True:
            self.features, self.encoder = features
        else: 
            self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()


    def forward(self, x, local = False, prune = False):

        if local == False:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        else:
            x, self.prune_filter = self.features(x)
            return x, self.prune_filter


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def resetPrune(self, threshold=0.9):
        self.encoder.resetPrune(threshold=threshold)


    def resetdePrune(self, rightSideValue=3):
        self.encoder.resetdePrune(rightSideValue=rightSideValue)


def make_layers(cfg, compressionProps=None, in_channels=3, batch_norm=True):
    layers = []  
    return_encoder = False
    for v in cfg:
        if isinstance(v, str) and v.endswith("CL"):
            out_channels = int(v[:-2])
            prevLayerProps = {}
            prevLayerProps["PrevLayerOutChannel"] = in_channels
            prevLayerProps["NextLayerInChannel"] = out_channels
            encoder = encodingUnit(compressionProps,prevLayerProps)
            layers += [encoder]
            return_encoder = True
        elif v == 'CS':
            prevLayerProps = {}
            prevLayerProps["PrevLayerOutChannel"] = in_channels
            layers += [decodingUnit(compressionProps,prevLayerProps)]
            in_channels = in_channels 
        elif v == 'M':   
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if return_encoder == True:
        return nn.Sequential(*layers), encoder
    else:
        return nn.Sequential(*layers)


def split_list(input_list):
    list1, list2 = [], ['CS']

    for item in input_list:
        if isinstance(item, str) and item.endswith("CL"):
            # Found a string ending with "CL", add it to list1 and break the loop
            list1.append(item)
            break
        else:
            list1.append(item)

    # Add the remaining items to list2
    list2.extend(input_list[len(list1):])

    print(f"Client layers: {list1}")
    print(f"Server layers: {list2}")
    return list1, list2

## To make add Compression to a specific layer, just add 'CL' term as a string
## e.g. 256 -> '256CL'

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'ACL': [64, 'M', '128CL', 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_local, cfg_server = split_list(cfg['ACL'])



def NeuralNetwork(**kwargs):
    model = VGG(make_layers(cfg['A']), **kwargs)     #E is 19, A is 11
    return model

def NeuralNetwork_local(compressionProps,**kwargs):
    model = VGG(make_layers(cfg_local, compressionProps), local=True, **kwargs)     #E is 19, A is 11, D is 16
    return model

def NeuralNetwork_server(compressionProps,**kwargs):
    #make sure its 128, 64 or something for in-channel correctly
    # Last value of cfg_local
    in_channels = int(cfg_local[-1][:-2])
    model = VGG(make_layers(cfg_server, compressionProps, in_channels = in_channels), **kwargs)     #for prev_channel, it is the last conv layer out channels in local
    return model


def resetPrune(model):
    model[-1].resetPrune()