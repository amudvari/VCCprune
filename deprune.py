import random
import numpy as np
import torch
import csv

from utils import get_device, get_dataloaders, train, test, prune, prunetest
from torch import nn
from models.vccModel import NeuralNetwork_local
from models.vccModel import NeuralNetwork_server
from itertools import product


def depruning(dataset,
              training_epochs=50, prune_1_epochs=15, prune_2_epochs=15,
              prune_1_budget=16, prune_2_budget=4,
              delta=0.001, resolution_comp=1, device="cuda", rightSideValue=1):

    compressionProps = {}
    compressionProps['feature_compression_factor'] = 1
    compressionProps['resolution_compression_factor'] = resolution_comp

    train_dataloader, test_dataloader, num_classes = get_dataloaders(dataset)
    
    device = get_device(device)
    model1 = NeuralNetwork_local(compressionProps, num_classes=num_classes).to(device)
    model2 = NeuralNetwork_server(compressionProps, num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2, momentum=0.0,
                                 weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2, momentum=0.0,
                                 weight_decay=5e-4)


    #error track: 
    avg_errors = []
    avg_mask_errors = []
    test_accs = []
    test_losses = []

    #pruning
    epochs = prune_1_epochs #5
    budget = prune_1_budget
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_error, mask_error =  prune(train_dataloader, model1, model2,
                                       loss_fn, optimizer1, optimizer2, budget,
                                       delta=delta, device=device)
        avg_errors.append(avg_error)
        avg_mask_errors.append(mask_error)
        test_acc, test_loss =  prunetest(test_dataloader, model1, model2,
                                         loss_fn, budget, device=device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print("entire epoch's error: ", avg_error)

    model1.resetdePrune(rightSideValue=rightSideValue)
    optimizer1 = torch.optim.SGD(
        model1.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(
        model2.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)


    #pruning
    epochs = prune_2_epochs #5
    budget = prune_2_budget
    for t in range(epochs):
        if t >= 3:
            optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2,
                                         momentum=0.0, weight_decay=5e-4)
            optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2,
                                         momentum=0.0, weight_decay=5e-4)
        print(f"Epoch {t+1}\n-------------------------------")
        avg_error, mask_error =  prune(train_dataloader, model1, model2,
                                       loss_fn, optimizer1, optimizer2, budget,
                                       delta=delta, device=device)
        avg_errors.append(avg_error)
        avg_mask_errors.append(mask_error)
        test_acc, test_loss =  prunetest(test_dataloader, model1, model2,
                                         loss_fn, budget, device=device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print("entire epoch's error: ", avg_error)

    model1.resetdePrune(rightSideValue=rightSideValue)

    optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2, momentum=0.0,
                                 weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2, momentum=0.0,
                                 weight_decay=5e-4)

    #full training
    epochs = training_epochs #5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_error = train(train_dataloader, model1, model2, loss_fn,
                          optimizer1, optimizer2, device=device)
        avg_errors.append(avg_error)
        avg_mask_errors.append(0)
        test_acc, test_loss =  test(test_dataloader, model1, model2,
                                    loss_fn, device=device)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        print("entire epoch's error: ", avg_error)

    filename = f'results/depruning/{dataset}/data_{prune_1_epochs}_{prune_2_epochs}_{training_epochs}_{resolution_comp}_{delta}.csv'
    epochs = np.arange(1, len(avg_errors)+1)
    rows = zip(epochs, avg_errors, avg_mask_errors, test_accs)
    with open(filename, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epochs", "avg_errors", "avg_mask_errors", "test_accs"])
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":

    # Python random seed
    random.seed(56)
    # PyTorch random seed
    torch.manual_seed(56)
    # NumPy random seed
    np.random.seed(56)

    datasets = [
        # 'STL10',
        # 'CIFAR10',
        # 'CIFAR100',
        'Imagenet100',
    ]
    prune_1_epochs = [45]
    prune_2_epochs = [0]
    training_epochs = [0]
    prune_1_budgets = [128]
    prune_2_budgets = [0]
    deltas = [0.1]  
    resolution_comps = [1]
    device = "cuda:1"
    rightSideValues = [3]

    for dataset, resolution_comp, training_epoch, prune_1_epoch, prune_2_epoch, \
        prune_1_budget, prune_2_budget, delta, rightSideValue  \
        in product(datasets, resolution_comps, training_epochs, prune_1_epochs, prune_2_epochs,
                   prune_1_budgets, prune_2_budgets, deltas, rightSideValues):
        print(f"""
              ---------------------------
              Parameters
              ---------------------------
              Dataset: {dataset},
              Resolution Compression Factor: {resolution_comp},
              Prune 1 Epochs: {prune_1_epoch},
              Prune 2 Epochs: {prune_2_epoch},
              Training Epochs: {training_epoch},
              Prune 1 Budget: {prune_1_budget},
              Prune 2 Budget: {prune_2_budget},
              Delta: {delta}
              RightSideValue: {rightSideValue}
              ---------------------------
              """)

        depruning(dataset, training_epochs=training_epoch,
                 prune_1_epochs=prune_1_epoch, prune_2_epochs=prune_2_epoch,
                 prune_1_budget=prune_1_budget, prune_2_budget=prune_2_budget,
                 delta=delta, resolution_comp=resolution_comp, device=device,
                 rightSideValue=rightSideValue)