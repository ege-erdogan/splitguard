import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(dataset, batch_size=64):
    trn = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                            ])

    if dataset == 'mnist':
        trainset = datasets.MNIST('sg_data/data/mnist', download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.MNIST('sg_data/data/mnist', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'f_mnist':
        trainset = datasets.FashionMNIST('sg_data/data/f_mnist', download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.FashionMNIST('sg_data/data/f_mnist', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'cifar':
        trainset = datasets.CIFAR10('sg_data/data/cifar', download=True, train=True, transform=transforms.ToTensor())
        testset = datasets.CIFAR10('sg_data/data/cifar', download=True, train=False, transform=transforms.ToTensor())
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100('sg_data/cifar100', download=True, train=True, transform=trn)
        testset = datasets.CIFAR100('sg_data/cifar100', download=True, train=False, transform=trn)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True)
    return trainloader, testloader


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def get_test_score(model, testloader, device):
    model.eval()
    score = 0
    for idx, (image, label) in enumerate(testloader):
        if idx ==  1000:
            break
        image, label = image.to(device), label.to(device)
        probabilities = model(image)
        pred = torch.argmax(probabilities, dim=None)
        score += pred == label
    return 100 * score / 1000


def get_optims(name, client, server, split_index=2):
    if name == 'adam':
        client_opt = torch.optim.Adam(list(client.parameters())[:split_index], lr=0.001, amsgrad=True)
        server_opt = torch.optim.Adam(list(server.parameters())[split_index:], lr=0.001, amsgrad=True)
    elif name == 'adagrad':
        client_opt = torch.optim.Adagrad(list(client.parameters())[:split_index], lr=0.01)
        server_opt = torch.optim.Adagrad(list(server.parameters())[split_index:], lr=0.01)
    elif name == 'rmsprop':
        client_opt = torch.optim.RMSprop(list(client.parameters())[:split_index], lr=0.01)
        server_opt = torch.optim.RMSprop(list(server.parameters())[split_index:], lr=0.01)
    elif name == 'sgd':
        client_opt = torch.optim.SGD(list(client.parameters())[:split_index], lr=0.001)
        server_opt = torch.optim.SGD(list(server.parameters())[split_index:], lr=0.001)
    elif name == 'sgd-m':
        client_opt = torch.optim.SGD(list(client.parameters())[:split_index], lr=0.001, momentum=0.1)
        server_opt = torch.optim.SGD(list(server.parameters())[split_index:], lr=0.001, momentum=0.1)
    return client_opt, server_opt


def plot_scores(scores, vals, p_fake=0.1, hline=None, xticks=None, ylim=1.1, ylabel='SplitGuard Score', xlabel='No. of fake batches'):
    colors = ['forestgreen', 'red', 'dodgerblue', 'brown']
    plt.figure(figsize=(9,7))
    plt.rc('font', size=16)
    
    length = min([len(s) for s in scores])
    for i, (lst, b_fake) in enumerate(zip(scores, vals)):
        plt.plot(range(len(lst[:length])), lst[:length], label=f'{b_fake}', linewidth=2.5, color=colors[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0,ylim)
    plt.grid()
    if hline is not None:
        plt.axhline(hline, color='orange')
    if xticks is not None:
        plt.xticks(range(len(xticks)), xticks, size='small', rotation='vertical')
    else:
        plt.xticks(range(1, length, 4))
    plt.legend()
    plt.show()


def get_fsha_scores(dataset, p_fake=0.1, mult=1, exp=1, shift=0, raw=False):
    with open(f'./{dataset}_fsha_grads', 'rb+') as f:
        grads = pickle.load(f)[0]
    fsha_grads = [torch.tensor(g).cpu() for g in grads]
    scores = []
    fakes, regulars = [], [[], []]
    for g in fsha_grads:
        p = random.random()
        if p < (1-p_fake)/2:
            regulars[0].append(g.flatten())
        elif p < (1-p_fake):
            regulars[1].append(g.flatten())
        else:
            fakes.append(g.flatten())
            if len(regulars[0]) > 0 and len(regulars[1]) > 0:
                scores.append(sg_score(fakes, regulars[0], regulars[1], shift=shift, mult=mult, exp=exp, raw=raw))
    return scores

