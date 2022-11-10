# SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning

## Abstract

Distributed deep learning frameworks, such as *split learning*, have recently been proposed to enable a group of participants to collaboratively train a deep neural network without sharing their raw data. Split learning in particular achieves this goal by dividing a neural network between a client and a server so that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to steal the client's private data: the server can direct the client model towards learning a task of its choice. With a concrete example already proposed, such training-hijacking attacks present a significant risk for the data privacy of split learning clients. 

In this paper, we propose SplitGuard, a method by which a split learning client can detect whether it is being targeted by a training-hijacking attack or not. We experimentally evaluate its effectiveness, and discuss in detail various points related to its use. We conclude that SplitGuard can effectively detect training-hijacking attacks while minimizing the amount of information recovered by the adversaries.

https://arxiv.org/abs/2108.09052

## Code

The Jupyter notebook `splitguard.ipynb` contains a sample run of the SplitGuard protocol against an honest and a random-labeling server.

## Cite Our Work
```
@inproceedings{10.1145/3559613.3563198,
author = {Erdogan, Ege and K\"{u}p\c{c}\"{u}, Alptekin and Cicek, A. Ercument},
title = {SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning},
year = {2022},
isbn = {9781450398732},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3559613.3563198},
doi = {10.1145/3559613.3563198},
booktitle = {Proceedings of the 21st Workshop on Privacy in the Electronic Society},
pages = {125–137},
numpages = {13},
keywords = {model inversion, split learning, data privacy, machine learning},
location = {Los Angeles, CA, USA},
series = {WPES'22}
}
```
