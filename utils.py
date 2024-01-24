import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.cifar10 import load_CIFAR10_dataset
from datasets.stl10 import load_STL10_dataset
from datasets.imagenet100 import load_Imagenet100_dataset
from models.vggModel import NeuralNetwork
import torch.nn as nn
import time

def get_device(dev: str = None):
    if dev:
        device = dev
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device


def get_dataloaders(dataset: str = "CIFAR10"):
    if dataset == "CIFAR10":
        train_dataloader, test_dataloader, num_classes = load_CIFAR10_dataset(
            batch_size=16)
    elif dataset == "STL10":
        train_dataloader, test_dataloader, num_classes = load_STL10_dataset(
            batch_size=16)
    elif dataset == "Imagenet100":
        train_dataloader, test_dataloader, num_classes = load_Imagenet100_dataset(
            batch_size=16)
    else:
        print(f"No dataset named {dataset} exists.")
        return

    return train_dataloader, test_dataloader, num_classes


def train(dataloader, model_local, model_server, loss_fn, optimizer_local,
          optimizer_server, quantizeDtype=torch.float32,
          realDtype=torch.float32, **kwargs):
    size = len(dataloader.dataset)
    model_local.train()
    model_server.train()

    total_loss = 0
    device = kwargs['device']

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        split_vals, prune_filter = model_local(X, local=True, prune=False)
        detached_split_vals = split_vals.detach()
        quantized_split_vals = detached_split_vals.to(quantizeDtype)
        transfererd_split_vals = quantized_split_vals.detach().to(device)

        dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
        serverInput_split_vals = Variable(dequantized_split_vals,
                                          requires_grad=True)
        pred = model_server(serverInput_split_vals)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer_local.zero_grad()
        optimizer_server.zero_grad()

        loss.backward()
        grad_store = serverInput_split_vals.grad
        split_grad = grad_store.detach().to(device)

        split_vals.backward(split_grad)
        optimizer_server.step()
        optimizer_local.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            total_loss += loss
        else:
            total_loss += loss.item()

    return total_loss


def train_single_device_step(dataloader, model, loss_fn, optimizer,
          quantizeDtype=torch.float32,
          realDtype=torch.float32, **kwargs):
    size = len(dataloader.dataset)
    model.train()

    total_loss = 0
    device = kwargs['device']

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            total_loss += loss
        else:
            total_loss += loss.item()

    return total_loss


def pruneLoss(loss_fn, pred, y, prune_filter, budget, epsilon=1000, delta=0.001):

    prune_filter_squeezed = prune_filter.squeeze()
    prune_filter_control_1 = torch.exp(
        delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
    prune_filter_control_2 = torch.exp(
        - delta * (sum(torch.square(torch.sigmoid(prune_filter_squeezed)))-budget))
    prune_filter_control = prune_filter_control_1 + prune_filter_control_2
    entropyLoss = loss_fn(pred, y)
    diff = entropyLoss + epsilon * prune_filter_control
    return diff


def prune(dataloader, model_local, model_server, loss_fn,
          optimizer_local, optimizer_server, budget,  pruneBackward=True,
          quantizeDtype=torch.float16, realDtype=torch.float32,
          mask_filtering_method="partition", **kwargs):
    size = len(dataloader.dataset)
    model_local.train()
    model_server.train()

    total_loss = 0
    total_mask_loss = 0
    device = kwargs['device']

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        split_vals, prune_filter = model_local(X, local=True, prune=True)
        detached_split_vals = split_vals.detach()
        quantized_split_vals = detached_split_vals.to(quantizeDtype)
        mask = torch.square(torch.sigmoid(prune_filter.squeeze())).to(device)
        if mask_filtering_method == "partition":
            masknp = mask.to('cpu').detach().numpy()
            partitioned = np.partition(masknp, -budget)[-budget]
            mask_allowed = 0
            for entry in range(len(mask)):
                if mask[entry] < partitioned:
                    mask[entry] = 0
                else:
                    mask_allowed += 1
        else:
            mask_allowed = 0
            for entry in range(len(mask)):
                if mask[entry] < 0.1:
                    mask[entry] = 0
                else:
                    mask_allowed += 1

        unsqueezed_mask = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(mask, 0), 2), 3)
        masked_split_val = torch.mul(quantized_split_vals, unsqueezed_mask)

        transfererd_split_vals = masked_split_val.detach().to(device)
        dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
        serverInput_split_vals = Variable(
            dequantized_split_vals, requires_grad=True)
        pred = model_server(serverInput_split_vals)
        # loss_fn(pred,y) #pruneLoss(pred, y, prune_filter, budget)
        loss = pruneLoss(loss_fn, pred, y, prune_filter,
                         budget, delta=kwargs['delta'])
        realLoss = loss_fn(pred, y)

        # Backpropagation
        optimizer_local.zero_grad()
        optimizer_server.zero_grad()

        loss.backward()
        grad_store = serverInput_split_vals.grad

        if pruneBackward:
            mask_upload = mask.to(device)
            unsqueezed_mask_upload = torch.unsqueeze(
                torch.unsqueeze(torch.unsqueeze(mask_upload, 0), 2), 3)
            masked_split_grad_store = torch.mul(
                grad_store, unsqueezed_mask_upload)
            split_grad = masked_split_grad_store.detach().to(device)
        else:
            split_grad = grad_store.detach().to(device)

        split_vals.backward(split_grad)
        optimizer_server.step()
        optimizer_local.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            realLoss = realLoss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Real loss: {realLoss:>7f}  [{current:>5d}/{size:>5d}]")
            total_loss += realLoss
            total_mask_loss += loss
        else:
            total_loss += realLoss.item()
            total_mask_loss += loss.item()
    a = torch.square(torch.sigmoid(prune_filter.squeeze()))

    return total_loss, total_mask_loss


def test(dataloader, model_local, model_server, loss_fn,
         quantizeDtype=torch.float16, realDtype=torch.float32, **kwargs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_local.eval()
    model_server.eval()
    test_loss, correct = 0, 0
    device = kwargs['device']
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            split_vals, prune_filter = model_local(X, local=True, prune=False)
            detached_split_vals = split_vals.detach()
            quantized_split_vals = detached_split_vals.to(quantizeDtype)
            transfererd_split_vals = quantized_split_vals.detach().to(device)

            dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
            serverInput_split_vals = Variable(
                dequantized_split_vals, requires_grad=True)
            pred = model_server(serverInput_split_vals)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct, test_loss)


def test_single_device_step(dataloader, model, loss_fn, **kwargs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    device = kwargs['device']
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct, test_loss)


def prunetest(dataloader, model_local, model_server, loss_fn, budget,
              quantizeDtype=torch.float16, realDtype=torch.float32,
              mask_filtering_method="partition", **kwargs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_local.eval()
    model_server.eval()
    test_loss, correct = 0, 0
    device = kwargs['device']
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            split_vals, prune_filter = model_local(X, local=True, prune=False)
            detached_split_vals = split_vals.detach()
            quantized_split_vals = detached_split_vals.to(quantizeDtype)
            mask = torch.square(torch.sigmoid(
                prune_filter.squeeze())).to(device)
            if mask_filtering_method == "partition":
                masknp = mask.to('cpu').detach().numpy()
                partitioned = np.partition(masknp, -budget)[-budget]
                mask_allowed = 0
                for entry in range(len(mask)):
                    if mask[entry] < partitioned:
                        mask[entry] = 0
                    else:
                        mask_allowed += 1
            else:
                mask_allowed = 0
                for entry in range(len(mask)):
                    if mask[entry] < 0.1:
                        mask[entry] = 0
                    else:
                        mask_allowed += 1

            unsqueezed_mask = torch.unsqueeze(
                torch.unsqueeze(torch.unsqueeze(mask, 0), 2), 3)
            masked_split_val = torch.mul(quantized_split_vals, unsqueezed_mask)

            transfererd_split_vals = masked_split_val.detach().to(device)
            dequantized_split_vals = transfererd_split_vals.detach().to(realDtype)
            serverInput_split_vals = Variable(
                dequantized_split_vals, requires_grad=True)
            pred = model_server(serverInput_split_vals)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct, test_loss)


def plot(filenames, result_name="Figure_6", directory="figures/Figure_6"):

    colors = ['b', 'r', 'g', 'm']
    lines = ['solid', 'solid', 'dashed', 'dashed']
    fig = plt.figure(figsize=(12, 7))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Test Accuracy (%)', size=15)

    for f,c,l in zip(filenames, colors, lines):

        df = pd.read_csv(directory + '/' + f)
        ax.plot(df['epochs'], df['test_accs'], color=c, ls=l,
                label=f.split('.csv')[0])

    plt.scatter(df['epochs'], df['test_accs'], color=c)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
            fancybox=True, shadow=True, ncol=5, prop=dict(size=12))
    plt.savefig(directory + '/' + result_name)


def plot_with_offset(filenames, result_name="Figure_6",
                     directory="figures/Figure_6",
                     offsets = {"no_compression": [0, 30],
                                "4Comp": [40, 50],
                                "32Comp": [30, 40]}):

    colors = ['b', 'r', 'g', 'm']
    lines = ['solid', 'solid', 'dashed', 'dashed']
    fig = plt.figure(figsize=(12, 7))
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Test Accuracy (%)', size=15)

    for f, c, l in zip(filenames, colors, lines):

        df = pd.read_csv(directory + '/' + f)
        label = f.split('.csv')[0].split('/')[0]
        print(label)

        if label in offsets.keys():
            # print(df)
            df = df.iloc[:offsets[label][1]]
            print(df)
            df = df.shift(periods=offsets[label][0])
            print(df)

        ax.plot(df.index, df['test_accs'], color=c, ls=l,
                label=f.split('.csv')[0])

    plt.scatter(df.index, df['test_accs'], color=c)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5, prop=dict(size=12))
    plt.savefig(directory + '/' + result_name)


def train_single_device(dataset, training_epochs=50,device=None,
                        filename="test.csv", lr=1e-3, result_directory="figures/Figure_6",
                        model_path="results/savedModels/single_device/model1.pth"):


    start_time = time.time()
    train_dataloader, test_dataloader, num_classes = get_dataloaders(dataset)

    device = get_device(device)
    model = NeuralNetwork(num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),  lr=lr,
                                 momentum=0.9, weight_decay=5e-4)

    #error track: 
    avg_errors = []
    avg_mask_errors = []
    test_accs = []
    test_errors = []

    # Training
    epochs = training_epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_error = train_single_device_step(train_dataloader, model, loss_fn,
                          optimizer, device=device)
        avg_errors.append(avg_error)
        avg_mask_errors.append(0)
        test_acc, test_error = test_single_device_step(test_dataloader, model, 
                                    loss_fn, device=device)
        test_accs.append(test_acc)
        test_errors.append(test_error)
        print("entire epoch's error: ", avg_error)

    end_time = time.time()
    print("Total time: ", end_time - start_time)

    # torch.save(model.state_dict(), model_path)
    # print("Saved PyTorch Model State to {:s}".format(model_path))