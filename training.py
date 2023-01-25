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

import matplotlib.pyplot as plt
import csv

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"Using {device} device")
    return device


def train(dataloader, model_local, model_server, loss_fn, optimizer_local, optimizer_server, quantizeDtype = torch.float16, realDtype = torch.float32):
    size = len(dataloader.dataset)
    model_local.train()
    model_server.train()
    
    total_loss = 0
      
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to(device)

        # Compute prediction error
        split_vals, prune_filter = model_local(X, local=True, prune=False)
        detached_split_vals = split_vals.detach()
        quantized_split_vals = detached_split_vals.to(quantizeDtype)
        transfererd_split_vals = quantized_split_vals.detach().to(device)
        
        dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
        serverInput_split_vals = Variable(dequantized_split_vals, requires_grad=True)
        pred = model_server(serverInput_split_vals)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer_local.zero_grad()
        optimizer_server.zero_grad()
        
        loss.backward()
        grad_store = serverInput_split_vals.grad
        split_grad = grad_store.detach().to('cpu')   
        
        split_vals.backward(split_grad)  
        optimizer_server.step()    
        optimizer_local.step()      

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            total_loss += loss
        else:
            total_loss += loss.item()    
            
        #if batch * len(X) > 100: #4800:
        #    return total_loss
           
            
    return total_loss     
         
def pruneLoss(loss_fn, pred, y, prune_filter, budget, epsilon=1000):
    
    prune_filter_squeezed = prune_filter.squeeze()
    prune_filter_control = torch.exp( 0.1 * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget)   )
    #(( (sum(prune_filter_squeezed)-budget) > 0 ).float() * 10000 ).squeeze()
    #print(prune_filter)
    #print(prune_filter_control)
    #print(prune_filter_squeezed)
    entropyLoss = loss_fn(pred,y)
    diff = entropyLoss + epsilon * prune_filter_control
    return diff
    
    
           
def prune(dataloader, model_local, model_server, loss_fn, optimizer_local, optimizer_server, budget, 
          quantizeDtype = torch.float16, realDtype = torch.float32):
    size = len(dataloader.dataset)
    model_local.train()
    model_server.train()  
    
    total_loss = 0
    total_mask_loss = 0
      
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to(device)

        # Compute prediction error
        split_vals, prune_filter = model_local(X, local = True, prune=True)
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
        loss = pruneLoss(loss_fn, pred, y, prune_filter, budget)              #loss_fn(pred,y) #pruneLoss(pred, y, prune_filter, budget)
        realLoss = loss_fn(pred,y)
        
        # Backpropagation
        optimizer_local.zero_grad()
        optimizer_server.zero_grad()  
        
        loss.backward()
        grad_store = serverInput_split_vals.grad
        split_grad = grad_store.detach().to('cpu')   
        
        split_vals.backward(split_grad)  
        optimizer_server.step()    
        optimizer_local.step()      

        if batch % 1000 == 0 or batch < 10:
            loss, current = loss.item(), batch * len(X)
            realLoss= realLoss.item()
            #print(masked_split_val[0,:])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Real loss: {realLoss:>7f}  [{current:>5d}/{size:>5d}]")
            #a = torch.square(torch.sigmoid(prune_filter.squeeze()))
            #print("filter is: ", a)
            print("masks allowed: ",mask_allowed)
            total_loss += realLoss
            total_mask_loss += loss
        else:
            total_loss += realLoss.item()
            total_mask_loss += loss.item()
            
        #if batch * len(X) > 4800:
        #    return total_loss, total_mask_loss  

    a = torch.square(torch.sigmoid(prune_filter.squeeze()))
    print("filter is: ", a)
    return total_loss, total_mask_loss  


def test(dataloader, model_local, model_server, loss_fn, quantizeDtype = torch.float16, realDtype = torch.float32):
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
            transfererd_split_vals = quantized_split_vals.detach().to(device)
            
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
compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

device = get_device()
model1 = NeuralNetwork_local(compressionProps).to('cpu')
print(device)
model2 = NeuralNetwork_server(compressionProps)
#input_lastLayer = model2.classifier[6].in_features
#model2.classifier[6] = nn.Linear(input_lastLayer,10)
model2 = model2.to(device)
train_dataloader, test_dataloader, classes = load_CIFAR10_dataset(batch_size = 16)   #batch_size

loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2, momentum=0.0, weight_decay=5e-4)
optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2, momentum=0.0, weight_decay=5e-4) #torch.optim.Adam(model2.parameters())#
#optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2)
#optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2)


#error track: 
avg_errors = []
avg_mask_errors = []
test_accs = []
test_errors = []

# Training
epochs = 17
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error = train(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2)
    avg_errors.append(avg_error)
    avg_mask_errors.append(0)
    test_acc, test_error = test(test_dataloader, model1, model2, loss_fn)
    test_accs.append(test_acc)
    test_errors.append(test_error)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

'''
test(test_dataloader, model1, model2, loss_fn)
model1_path = "savedModels/model1.pth"
#torch.save(model1.state_dict(), model1_path)
#print("Saved PyTorch Model State to {:s}".format(model1_path))
model2_path = "savedModels/model2.pth"
#torch.save(model2.state_dict(), model2_path)
#print("Saved PyTorch Model State to {:s}".format(model2_path))

model1.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))
'''
test(test_dataloader, model1, model2, loss_fn)


#optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.3e-3, momentum=0.0, weight_decay=5e-4)
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.3e-3, momentum=0.0, weight_decay=5e-4)

model1.resetPrune()
        
#pruning
epochs = 7
budget = 32
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error, mask_error =  prune(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2, budget)
    avg_errors.append(avg_error)
    avg_mask_errors.append(mask_error)
    test_acc, test_error = prunetest(test_dataloader, model1, model2, loss_fn)
    test_accs.append(test_acc)
    test_errors.append(test_error)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

model1.resetPrune()

#pruning
epochs = 7
budget = 4
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error, mask_error =  prune(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2, budget)
    avg_errors.append(avg_error)
    avg_mask_errors.append(mask_error)
    test_acc, test_error = prunetest(test_dataloader, model1, model2, loss_fn)
    test_accs.append(test_acc)
    test_errors.append(test_error)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

print("errors across: ", avg_errors)
plt.plot(avg_errors)
plt.show()

print("mask errors across: ", avg_mask_errors)
plt.plot(avg_mask_errors)
plt.show()

print("test accuracy across: ", test_accs)
plt.plot(test_accs)
plt.show()

print("test errors across: ", test_errors)
plt.plot(test_errors)
plt.show()

filename = 'prune_data.csv'
epochs = np.arange(1,len(avg_errors)+1)
rows = zip(epochs,avg_errors,avg_mask_errors,test_accs)
with open(filename, 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epochs","avg_errors","avg_mask_errors","test_accs"])
    for row in rows:
        writer.writerow(row)


'''
model.to('cpu')
# Evaluation
model.eval()
x, y = next(iter(test_dataloader))
with torch.no_grad():
    pred = model(x[0])
    predicted, actual = classes[pred[0].argmax(0)], classes[y[0]]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
'''


def compute_nllloss_manual(x,y0):
    """
    x is the vector with shape (batch_size,C) 
    Note: official example uses log softmax(some vector) as x, so it becomes CELoss.
    y0 shape is the same (batch_size), whose entries are integers from 0 to C-1
    Furthermore, for C>1 classes, the other classes are ignored (see below

    """
    loss = 0.
    n_batch, n_class = x.shape
    # print(n_class)
    for x1,y1 in zip(x,y0):
        class_index = int(y1.item())
        loss = loss + x1[class_index] # other class terms, ignore.
    loss = - loss/n_batch
    return loss
