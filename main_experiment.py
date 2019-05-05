from types import FunctionType
from fastai.vision import create_cnn, ImageDataBunch
from fastai.train import AdamW

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time as t
from models import *
from csv_logger import record_metrics, log_to_csv
from torchvision.models import vgg16 as vgg16real
from fastai.vision import DataBunch, Learner, create_cnn
from fastai import *
from fastai.vision import *
from alternative_loaders import get_n_fold_datasets_test, get_n_fold_datasets_train
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
import sys
import psutil


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
global SAMPLER
global IMBALANCE
SAMPLER = None


def memReport():
    print('Memory REPORT')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                print(type(obj), obj.size())
        except:
            continue

def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

def _shuffle_classes(class_list):
    c_samples = np.random.choice(class_list, len(class_list), replace=False)
    class_list = np.asarray(c_samples)[class_list]
    return class_list

def get_class_probas(class_list, skew_val=1.0):
    class_list = _shuffle_classes(class_list)
    probas = np.linspace(-len(class_list)//2, len(class_list)//2, len(class_list)) * 0.01 * skew_val
    probas += 0.1
    print(probas)
    return probas

def get_sampler_with_random_imbalance(skew_val, num_samples, n_classes, labels):
    classes = list(range(n_classes))
    class_probas = get_class_probas(classes, skew_val)
    weights = np.zeros(num_samples)
    for cls in classes:
        prob = class_probas[cls]
        w = weights[np.asarray(labels) == cls]
        weights[np.asarray(labels) == cls] = class_probas[cls]
    weights = weights / np.linalg.norm(weights)
    global IMBALANCE
    print(class_probas)
    return WeightedRandomSampler(weights, num_samples, replacement=True)

def train_set_cifar(transform, batch_size):
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return train_loader

def test_set_cifar(transform, batch_size):
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader


def train_set_cifar100(transform, batch_size):
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=9
    )
    return train_loader

def test_set_cifar100(transform, batch_size):
    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=9
    )

    return test_loader


def train_set_imbalanced_cifar(transformer, batch_size, skew_val):
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transformer
    )
    global SAMPLER
    if SAMPLER is None:
        SAMPLER = get_sampler_with_random_imbalance(skew_val, len(train_set.targets), n_classes=10, labels=train_set.targets)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=SAMPLER
    )
    return train_loader

def train(network, dataset, test_set, logging_dir, batch_size):
    print('setting netowrk device to', device)
    network.to(device)
    print('parallelizing')
   # network = nn.DataParallel(network)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())
    #stats = CheckLayerSat(logging_dir, network, log_interval=len(dataset)//batch_size)
    #stats = CheckLayerSat(logging_dir, network, log_interval=60, sat_method='cumvar99', conv_method='mean')
    stats = CheckLayerSat(logging_dir, network, log_interval=80, sat_method='cumvar99', conv_method='mean')


    epoch_acc = 0
    thresh = 0.95
    epoch = 0
    total = 0
    correct = 0
    value_dict = None
    while epoch <= 20:
        print('Start Training Epoch', epoch, '\n')
        start = t.time()
        epoch_acc = 0
        train_loss = 0
        total = 0
        correct = 0
        network.train()
        for i, data in enumerate(dataset):
            step = epoch*len(dataset) + i
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            print(i,'of', len(dataset),'acc:', correct/total, sys.getsizeof(stats.logs)/(1024), 'KB Stat Memory')
            # display layer saturation levels
        end = t.time()
        stats.saturation()
        test_loss, test_acc = test(network, test_set, criterion, stats, epoch)
        epoch_acc = correct / total
        print('Epoch', epoch, 'finished', 'Acc:', epoch_acc, 'Loss:', train_loss / total,'\n')
        stats.add_scalar('train_loss', train_loss / total, epoch)  # optional
        stats.add_scalar('train_acc', epoch_acc, epoch)  # optional
        value_dict = record_metrics(value_dict, stats.logs, epoch_acc, train_loss/total, test_acc, test_loss, epoch, (end-start) / total)
        log_to_csv(value_dict, logging_dir)
        epoch += 1

        cpuStats()
        memReport()

    stats.close()
#    test_stats.close()

    return criterion


def test(network, dataset, criterion, stats, epoch):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #if batch_idx % 200 == 199:  # print every 200 mini-batches
            print(batch_idx,'of', len(dataset),'acc:', correct/total)
        stats.saturation()
        print('Test finished', 'Acc:', correct / total, 'Loss:', test_loss/total,'\n')
        stats.add_scalar('test_loss', test_loss/total, epoch)  # optional
        stats.add_scalar('test_acc', correct/total, epoch)  # optional
    return test_loss/total, correct/total

def execute_experiment(network: nn.Module, in_channels: int, n_classes: int, l1: int, l2: int , l3: int, train_set: FunctionType, test_set: FunctionType):

    print('Experiment has started')

    batch_size = 128

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    check = 25
    i = 0
    for l1_config in l1:
        for l2_config in l2:
            for l3_config in l3:
                i += 1
                if i <= check:
                    continue
                print('Creating Network')

                net = network(in_channels=in_channels,
                        l1=l1_config,
                        l2=l2_config,
                        l3=l3_config,
                        n_classes=n_classes)
                print('Network created')



                train_loader = train_set(transform, batch_size)
                test_loader = test_set(transform, batch_size)


                print('Datasets fetched')
                train(net, train_loader, test_loader, '{}_{}_{}'.format(l1_config, l2_config, l3_config), batch_size)

                del net

def execute_experiment_vgg(network: nn.Module, net_name: str, train_set: FunctionType, test_set: FunctionType, n_claases=2):

    print('Experiment has started')

    batch_size = 128

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        #transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print('Creating Network')
    train_loader = train_set(transform_train, batch_size)
    test_loader = test_set(transform_test, batch_size)

    net = network(num_classes=n_claases)#create_cnn(data, vgg16).model#network()
    print(net)
    print('Network created')


    print('Datasets fetched')
    train(net, train_loader, test_loader, net_name, batch_size)

    return


if '__main__' == __name__:

    #executions = [vgg11_XXXS, vgg13_XXXS, vgg16_XXXS, vgg19_XXXS,
    #              vgg11_XXS, vgg13_XXS, vgg16_XXS, vgg19_XXS,
    #              vgg11_XS, vgg13_XS, vgg16_XS, vgg19_XS,
    #              vgg11_S, vgg13_S, vgg16_S, vgg19_S,
    #              vgg11, vgg13, vgg16, vgg19,
    #              ]
    #names = ['11_XXXS', '13_XXXS', '16_XXXS', '19_XXXS',
    #         '11_XXS', '13_XXS', '16_XXS', '19_XXS',
    #         '11_XS', '13_XS', '16_XS', '19_XS',
    #         '11_S', '13_S', '16_S', '19_S',
    #         '11', '13', '16', '19',
    #         ]

    executions = [vgg19]
    names = ['test19']

    train_set = lambda transform_train, batch_size: get_n_fold_datasets_train(transform_train, batch_size, ['automobile', 'frog'])
    test_set = lambda transform_test, batch_size: get_n_fold_datasets_test(transform_test, batch_size, ['automobile', 'frog'])

    executions.reverse()
    names.reverse()

    train_set_imbalanced_cifar
    counter = 0
    for j in [0]:

        #sampler = WeightedRandomSampler()
        for i in range(len(names)):
            counter += 1
            print('COUNTER:',counter)
            print(device)
            print(torch.cuda.device_count())
            print(torch.cuda.current_device())
            print(names[i])

            #configVGG_cifar = {
            #    'network': executions[i],
            #    'train_set': train_set,
            #    'test_set': test_set,
            #    'net_name': 'automobilefrog_VGG' + names[i] + '_A' + str(j),
            #    'n_claases': 2
            #}

           # execute_experiment_vgg(**configVGG_cifar)

            configVGG_cifar = {
                'network': executions[i],
                'train_set': train_set_cifar,#lambda t, batch_size: train_set_imbalanced_cifar(t, batch_size=batch_size, skew_val=1.0),
                'test_set': test_set_cifar,
                'net_name': 'BIG10_VGG' + names[i] + '_A' + str(j),
                'n_claases': 10
            }
            execute_experiment_vgg(**configVGG_cifar)


    configCNN_cifar = {
        'network': SimpleCNN,
        'in_channels': 3,
        'n_classes': 10,
        'l1' : [4, 16, 64],
        'l2' : [8, 32, 128],
        'l3' : [16, 64, 256],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    configCNNKernel_cifar = {
        'network': SimpleCNNKernel,
        'in_channels': 3,
        'n_classes': 10,
        'l1': [3, 5, 7],
        'l2': [3, 5, 7],
        'l3': [3, 5, 7],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    configFCN_cifar = {
        'network': SimpleFCNet,
        'in_channels': 32*32*3,
        'n_classes': 10,
        'l1' : [4*3*3, 16*3*3, 64*3*3],
        'l2' : [8*3*3, 32*3*3, 128*3*3],
        'l3' : [16*3*3, 64*3*3, 256*3*3],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    #execute_experiment(**configCNN_cifar)


