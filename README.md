# SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning

## Abstract

Distributed deep learning frameworks, such as \textit{split learning}, have recently been proposed to enable a group of participants to collaboratively train a deep neural network without sharing their raw data. Split learning in particular achieves this goal by dividing a neural network between a client and a server so that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to steal the client's private data: the server can direct the client model towards learning a task of its choice. With a concrete example already proposed, such training-hijacking attacks present a significant risk for the data privacy of split learning clients. 

In this paper, we propose SplitGuard, a method by which a split learning client can detect whether it is being targeted by a training-hijacking attack or not. We experimentally evaluate its effectiveness, and discuss in detail various points related to its use. We conclude that SplitGuard can effectively detect training-hijacking attacks while minimizing the amount of information recovered by the adversaries.

TODO: Link to paper

## Code

The Jupyter notebook `splitguard.ipynb` contains a sample run of the SplitGuard protocol against an honest and a random-labeling server.

## Cite Our Work
```
TODO: Citation
```