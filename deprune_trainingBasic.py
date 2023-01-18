import random
from tracemalloc import start

random.seed(57)
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from models.basicModel import NeuralNetwork
from models.basicModel import NeuralNetwork_local
from models.basicModel import NeuralNetwork_server
from datasets.mnist import load_mnist_dataset



import matplotlib.pyplot as plt

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
        split_vals, prune_filter = model_local(X, prune=False)
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            a = torch.square(torch.sigmoid(prune_filter.squeeze()))
            print("filter is: ", a)
            total_loss += loss
        else:
            total_loss += loss.item() 
            
        #if batch * len(X) > 12800:
        #    return total_loss
            
    return total_loss     
         
def depruneLoss(loss_fn, pred, y, prune_filter, budget, epsilon=1000):
    
    prune_filter_squeezed = prune_filter.squeeze()
    prune_filter_control_exp = torch.exp( 1 * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget)   )
    prune_filter_control_signed = torch.sgn( (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget) ) + 1 
    #print(prune_filter)
    #print(prune_filter_control)
    #print(prune_filter_squeezed)
    entropyLoss = loss_fn(pred,y)
    diff = entropyLoss + epsilon * prune_filter_control_exp
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
        split_vals, prune_filter = model_local(X, prune=True)
        detached_split_vals = split_vals.detach()
        quantized_split_vals = detached_split_vals.to(quantizeDtype)
        
        
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
        masknp = mask.detach().numpy()
        partitioned = np.partition(masknp, -budget)[-budget]
        for entry in range(len(mask)):
            if mask[entry] < partitioned: 
                mask[entry] = 0
 
        #print(mask)
 
        unsqueezed_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask,0),2),3)
        masked_split_val = torch.mul(quantized_split_vals,unsqueezed_mask)
        
        #print(masked_split_val)
        transfererd_split_vals = masked_split_val.detach().to(device)
        dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
        serverInput_split_vals = Variable(dequantized_split_vals, requires_grad=True)
        pred = model_server(serverInput_split_vals)
        loss = depruneLoss(loss_fn, pred, y, prune_filter, budget)              #loss_fn(pred,y) #pruneLoss(pred, y, prune_filter, budget)
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            realLoss= realLoss.item()
            #print(masked_split_val[0,:])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Real loss: {realLoss:>7f}  [{current:>5d}/{size:>5d}]")
            a = torch.square(torch.sigmoid(prune_filter.squeeze()))
            print("filter is: ", a)
            total_loss += realLoss
            total_mask_loss += loss
        else:
            total_loss += realLoss.item()
            total_mask_loss += loss.item()
            
        #if batch * len(X) > 12800:
        #    return total_loss, total_mask_loss  

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
            split_vals, prune_filter = model_local(X, prune=False)
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
    return(100*correct)


def prunetest(dataloader, model_local, model_server, loss_fn, budget, quantizeDtype = torch.float16, realDtype = torch.float32):
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
            split_vals, prune_filter = model_local(X, prune=True)
            detached_split_vals = split_vals.detach()
            quantized_split_vals = detached_split_vals.to(quantizeDtype)
            
            
            mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to('cpu')
            masknp = mask.detach().numpy()
            partitioned = np.partition(masknp, -budget)[-budget]
            for entry in range(len(mask)):
                if mask[entry] < partitioned: 
                    mask[entry] = 0

            #print(mask)

            unsqueezed_mask = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mask,0),2),3)
            masked_split_val = torch.mul(quantized_split_vals,unsqueezed_mask)
            
            #print(masked_split_val)
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
    return(100*correct)

'''
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
'''

compressionProps = {} ### 
compressionProps['feature_compression_factor'] = 1 ### resolution compression factor, compress by how many times
compressionProps['resolution_compression_factor'] = 1 ###layer compression factor, reduce by how many times TBD

device = get_device()
model1 = NeuralNetwork_local(compressionProps).to('cpu')
print(device)
model2 = NeuralNetwork_server(compressionProps).to(device)
train_dataloader, test_dataloader, classes = load_mnist_dataset(batch_size = 64)   #batch_size

loss_fn = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=1e-3)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-3)

#error track: 
avg_errors = []
avg_mask_errors = []


'''
# Training
epochs = 10
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error = train(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2)
    avg_errors.append(avg_error)
    avg_mask_errors.append(0)
    #test(test_dataloader, model, loss_fn)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.3e-3)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.3e-3)
'''
    
#pruning
epochs = 14
budget = 1
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error, mask_error =  prune(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2, budget)
    avg_errors.append(avg_error)
    avg_mask_errors.append(mask_error)
    #test(test_dataloader, model, loss_fn)
    prunetest(test_dataloader, model1, model2, loss_fn, budget)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)

model1.resetdePrune()

#pruning
epochs = 0
budget = 3
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error, mask_error =  prune(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2, budget)
    avg_errors.append(avg_error)
    avg_mask_errors.append(mask_error)
    #test(test_dataloader, model, loss_fn)
    prunetest(test_dataloader, model1, model2, loss_fn, budget)
    print("entire epoch's error: ", avg_error)
print("Done!")
end_time = time.time() 
print("time taken in seconds: ", end_time-start_time)


epochs = 7
start_time = time.time() 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_error = train(train_dataloader, model1, model2, loss_fn, optimizer1, optimizer2)
    avg_errors.append(avg_error)
    avg_mask_errors.append(0)
    #test(test_dataloader, model, loss_fn)
    test(test_dataloader, model1, model2, loss_fn)
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


# torch.save(model.state_dict(), model_path)
# print("Saved PyTorch Model State to {:s}".format(model_path))

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