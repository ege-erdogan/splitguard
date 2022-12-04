import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random
import math
import numpy as np
from models import *

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_vector_1, unit_vector_2)

    angle = np.arccos(dot_product)
    return math.degrees(angle)

def normalize(result):
    min_v = torch.min(result)
    range_v = torch.max(result) - min_v
    if range_v > 0:
        normalized = (result - min_v) / range_v
    else:
        normalized = torch.zeros(result.size())
    return normalized


def get_test_score(model):
    model.eval()
    score = 0
    for idx, (image, label) in enumerate(testloader):
        image, label = image.to(device), label.to(device)
        probabilities = model(image)
        pred = torch.argmax(probabilities, dim=None)
        score += pred == label
    return 100 * score / len(testloader)


def get_models(model_str, dataset, device):
    if model_str == 'resnet':
        if 'mnist' in dataset:
            model = ResNetMNIST().to(device)
        else:
            model = resnet18(num_classes=10).to(device)
    elif model_str == 'mnistnet':
        if 'mnist' in dataset:
            model = MnistNet().to(device), MnistNet().to(device)
        else:
            model = CifarNet().to(device), CifarNet().to(device)
    return model

def plot_scores(scores, vals, p_fake=0.1, hline=None, xticks=None, ylim=1.1, ylabel='SplitGuard Score', xlabel='No. of fake batches'):
    colors = ['forestgreen', 'red', 'dodgerblue']
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