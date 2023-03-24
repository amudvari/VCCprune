import random
from tracemalloc import start

random.seed(57)
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from models.vccModel import NeuralNetwork
from models.vccModel import NeuralNetwork_local
from models.vccModel import NeuralNetwork_server
from datasets.cifar10 import load_CIFAR10_dataset

from torchsummary import summary

RESOLUTION_COMPRESSION_FACTOR = 2


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"Using {device} device")
    return device

def prunetest(dataloader, model_local, model_server, loss_fn, quantizeDtype = torch.float16, realDtype = torch.float32):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #model.eval()
    model_local.eval()
    model_server.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            #pred = model(X)
            
            X, y = X.to('cpu'), y.to(device)

            # Compute prediction error
            split_vals, prune_filter = model_local(X, local=True, prune=False)
            detached_split_vals = split_vals.detach()
            quantized_split_vals = detached_split_vals.to(quantizeDtype)
            
            mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
            mask_allowed = 0
            for entry in range(len(mask)):
                if mask[entry] < 0.1: 
                    mask[entry] = 0
                else:
                    mask_allowed += 1
                    
            unsqueezed_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask,0),2),3)
            masked_split_val = torch.mul(quantized_split_vals,unsqueezed_mask)
            
            transfererd_split_vals = masked_split_val.detach().to(device)
            dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
            serverInput_split_vals = Variable(dequantized_split_vals, requires_grad=True)
            pred = model_server(serverInput_split_vals)
            #loss = loss_fn(pred, y)
            
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return(100*correct, test_loss)

compressionProps = {} ### 
compressionProps['feature_compression_factor'] = 1 ### resolution compression factor, compress by how many times
compressionProps['resolution_compression_factor'] = RESOLUTION_COMPRESSION_FACTOR ###layer compression factor, reduce by how many times TBD
num_classes = 10

device = get_device()
model1 = NeuralNetwork_local(compressionProps, num_classes = num_classes).to('cpu')
print(device)
model2 = NeuralNetwork_server(compressionProps, num_classes = num_classes)
#input_lastLayer = model2.classifier[6].in_features
#model2.classifier[6] = nn.Linear(input_lastLayer,10)
model2 = model2.to(device)
train_dataloader, test_dataloader, _ = load_CIFAR10_dataset(batch_size = 16)   #batch_size


budget = 32
model1_path = "savedModels/modelvgg1_"+str(budget)+".pth"
#torch.save(model1.state_dict(), model1_path)
#print("Saved PyTorch Model State to {:s}".format(model1_path))
model2_path = "savedModels/modelvgg2_"+str(budget)+".pth"
#torch.save(model2.state_dict(), model2_path)
#print("Saved PyTorch Model State to {:s}".format(model2_path))
loss_fn = nn.CrossEntropyLoss()

print(model1)
print(model2)

prunetest(test_dataloader, model1, model2, loss_fn)

model1.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))

print(model1)
print(model2)

prunetest(test_dataloader, model1, model2, loss_fn)






