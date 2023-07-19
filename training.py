import random
import time
import numpy as np
import torch
import csv
import datetime
import json

from datasets.cifar10 import load_CIFAR10_dataset
from datasets.cifar100 import load_CIFAR100_dataset
from datasets.stl10 import load_STL10_dataset
from datasets.imagenet100 import load_Imagenet100_dataset
from utils import get_device, train, test, prune, prunetest
from torch import nn
from torch.autograd import Variable
from models.vccModel import NeuralNetwork_local
from models.vccModel import NeuralNetwork_server
from itertools import product


def training(dataset,
             training_epochs=50, prune_1_epochs=15, prune_2_epochs=15,
             prune_1_budget=16, prune_2_budget=4,
             delta=0.001, resolution_comp=1, device="cuda", threshold=0.9,
             lr_boost=False, mask_filtering_method="partition",
             lr_boost_epoch=3, lr_boost_lr=5e-2):

    compressionProps = {} ### 
    compressionProps['feature_compression_factor'] = 1 ### resolution compression factor, compress by how many times
    compressionProps['resolution_compression_factor'] = resolution_comp ###layer compression factor, reduce by how many times TBD

    if dataset == "CIFAR10":
        train_dataloader, test_dataloader, num_classes = load_CIFAR10_dataset(batch_size = 16)   #batch_size
    elif dataset == "CIFAR100":
        train_dataloader, test_dataloader, num_classes = load_CIFAR100_dataset(batch_size = 16)   #batch_size
    elif dataset == "STL10":
        train_dataloader, test_dataloader, num_classes = load_STL10_dataset(batch_size = 16)   #batch_size
    elif dataset == "Imagenet100":
        train_dataloader, test_dataloader, num_classes = load_Imagenet100_dataset(batch_size=16)  # batch_size

    device = get_device(device)
    model1 = NeuralNetwork_local(compressionProps, num_classes=num_classes).to(device)
    print(device)
    model2 = NeuralNetwork_server(compressionProps, num_classes=num_classes)
    model2 = model2.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-3,
                                 momentum=0.9, weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-3,
                                 momentum=0.9, weight_decay=5e-4)

    #error track: 
    avg_errors = []
    avg_mask_errors = []
    test_accs = []
    test_errors = []

    # Training
    epochs = training_epochs
    start_time = time.time() 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        avg_error = train(train_dataloader, model1, model2, loss_fn,
                          optimizer1, optimizer2, device=device)
        avg_errors.append(avg_error)
        avg_mask_errors.append(0)
        test_acc, test_error = test(test_dataloader, model1, model2,
                                    loss_fn, device=device)
        test_accs.append(test_acc)
        test_errors.append(test_error)
        print("entire epoch's error: ", avg_error)
    print("Done!")
    end_time = time.time() 
    print("time taken in seconds: ", end_time-start_time)


    test(test_dataloader, model1, model2, loss_fn, device=device)
    model1_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/model1.pth"
    torch.save(model1.state_dict(), model1_path)
    print("Saved PyTorch Model State to {:s}".format(model1_path))
    model2_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/model2.pth"
    torch.save(model2.state_dict(), model2_path)
    print("Saved PyTorch Model State to {:s}".format(model2_path))


    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))

    test(test_dataloader, model1, model2, loss_fn, device=device)


    # model1.resetPrune(threshold=threshold)
    if lr_boost:
        optimizer1 = torch.optim.SGD(
            model1.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)
        optimizer2 = torch.optim.SGD(
            model2.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)
            
    #pruning
    epochs = prune_1_epochs
    budget = prune_1_budget
    start_time = time.time() 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if lr_boost:
            if t >= 3:
                optimizer1 = torch.optim.SGD(model1.parameters(),  lr=1e-2,
                                             momentum=0.0, weight_decay=5e-4)
                optimizer2 = torch.optim.SGD(model2.parameters(),  lr=1e-2,
                                             momentum=0.0, weight_decay=5e-4)
        avg_error, mask_error =  prune(train_dataloader, model1, model2,
                                       loss_fn, optimizer1, optimizer2,
                                       budget, delta=delta, device=device,
                                       mask_filtering_method="partition")
        avg_errors.append(avg_error)
        avg_mask_errors.append(mask_error)
        test_acc, test_error = prunetest(test_dataloader, model1, model2,
                                                       loss_fn, budget, device=device,
                                                       mask_filtering_method="partition")
        test_accs.append(test_acc)
        test_errors.append(test_error)
        print("entire epoch's error: ", avg_error)
    print("Done!")
    end_time = time.time() 
    print("time taken in seconds: ", end_time-start_time)

    test(test_dataloader, model1, model2, loss_fn, device=device)
    model1_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/modelvgg1_"+str(budget)+".pth"
    torch.save(model1.state_dict(), model1_path)
    print("Saved PyTorch Model State to {:s}".format(model1_path))
    model2_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/modelvgg2_"+str(budget)+".pth"
    torch.save(model2.state_dict(), model2_path)
    print("Saved PyTorch Model State to {:s}".format(model2_path))


    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))

    print("Test loaded")
    test(test_dataloader, model1, model2, loss_fn, device=device)

    # model1.resetPrune(threshold=threshold)
    if lr_boost:
        optimizer1 = torch.optim.SGD(
            model1.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)
        optimizer2 = torch.optim.SGD(
            model2.parameters(),  lr=5e-2, momentum=0.0, weight_decay=5e-4)


    #pruning
    epochs = prune_2_epochs
    budget = prune_2_budget
    start_time = time.time() 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if lr_boost:
            if t >= lr_boost_epoch:
                optimizer1 = torch.optim.SGD(
                    model1.parameters(),  lr=lr_boost_lr,
                    momentum=0.0, weight_decay=5e-4)
                optimizer2 = torch.optim.SGD(
                    model2.parameters(),  lr=lr_boost_lr, 
                    momentum=0.0, weight_decay=5e-4)
        avg_error, mask_error =  prune(train_dataloader, model1, model2,
                                       loss_fn, optimizer1, optimizer2,
                                       budget, delta=delta, device=device,
                                       mask_filtering_method="partition")
        avg_errors.append(avg_error)
        avg_mask_errors.append(mask_error)
        test_acc, test_error = prunetest(test_dataloader, model1, model2,
                                                       loss_fn, budget, device=device,
                                                       mask_filtering_method="partition")
        test_accs.append(test_acc)
        test_errors.append(test_error)
        print("entire epoch's error: ", avg_error)
    print("Done!")
    end_time = time.time() 
    print("time taken in seconds: ", end_time-start_time)


    test(test_dataloader, model1, model2, loss_fn, device=device)
    model1_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/modelvgg1_"+str(budget)+".pth"
    torch.save(model1.state_dict(), model1_path)
    print("Saved PyTorch Model State to {:s}".format(model1_path))
    model2_path = f"savedModels/{dataset}/res_comp_{resolution_comp}/modelvgg2_"+str(budget)+".pth"
    torch.save(model2.state_dict(), model2_path)
    print("Saved PyTorch Model State to {:s}".format(model2_path))

    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))

    print("Test loaded")
    test(test_dataloader, model1, model2, loss_fn, device=device)

    print("errors across: ", avg_errors)
    print("mask errors across: ", avg_mask_errors)
    print("test accuracy across: ", test_accs)
    print("test errors across: ", test_errors)

    t = time.time_ns()
    filename = f'results/pruning/{dataset}/data_{training_epochs}_{prune_1_epochs}_{prune_2_epochs}_{resolution_comp}_{delta}.csv'
    epochs = np.arange(1,len(avg_errors)+1)
    rows = zip(epochs,avg_errors,avg_mask_errors,test_accs)
    with open(filename, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epochs","avg_errors","avg_mask_errors","test_accs"])
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    
    random.seed(57)
    
    datasets = [
        # 'STL10',
        'CIFAR10',
        # 'CIFAR100',
        # 'Imagenet100',
    ]
    training_epochs = [30]
    prune_1_epochs = [15]
    prune_2_epochs = [20]
    prune_1_budgets = [8]
    prune_2_budgets = [2]
    deltas = [0.01]
    resolution_comps = [1]
    device = "cuda:0"
    thresholds = [0.1]
    lr_boosts = [True]
    mask_filtering_methods = ["partition"]
    lr_boost_epochs = [3]
    lr_boost_lrs = [5e-3]

    
    for dataset, resolution_comp, training_epoch, prune_1_epoch, prune_2_epoch, \
            prune_1_budget, prune_2_budget, delta, threshold, lr_boost, mask_filtering_method, \
            lr_boost_epoch, lr_boost_lr \
        in product(datasets, resolution_comps, training_epochs, prune_1_epochs, prune_2_epochs,
                   prune_1_budgets, prune_2_budgets, deltas, thresholds, lr_boosts, mask_filtering_methods,
                   lr_boost_epochs, lr_boost_lrs):
        print(f"""
              ---------------------------
              Parameters
              ---------------------------
              Dataset: {dataset},
              Resolution Compression Factor: {resolution_comp},
              Training Epochs: {training_epoch},
              Prune 1 Epochs: {prune_1_epoch},
              Prune 2 Epochs: {prune_2_epoch},
              Prune 1 Budget: {prune_1_budget},
              Prune 2 Budget: {prune_2_budget},
              Delta: {delta},
              LR_Boost: {lr_boost},
              LR_Boost_Epochs: {lr_boost_epoch},
              LR_Boost_lrs: {lr_boost_lrs},
              Threshold: {threshold},
              Mask Filtering Method: {mask_filtering_method}
              Device: {device}
              ---------------------------
              """)

        training(dataset, training_epochs=training_epoch,
                prune_1_epochs=prune_1_epoch, prune_2_epochs=prune_2_epoch,
                prune_1_budget=prune_1_budget, prune_2_budget=prune_2_budget,
                delta=delta, resolution_comp=resolution_comp, device=device,
                threshold=threshold,
                lr_boost=lr_boost, mask_filtering_method=mask_filtering_method,
                lr_boost_epoch=lr_boost_epoch, lr_boost_lr=lr_boost_lr)
