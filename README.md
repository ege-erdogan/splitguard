# Defense Mechanisms Against Training-Hijacking Attacks in Split Learning

## Abstract

Distributed deep learning frameworks enable more efficient and privacy-aware training of deep neural networks across multiple clients. Split learning achieves this by splitting a neural network between a client and a server such that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to recover the client’s private inputs: the server can direct the client model towards learning any task of its choice, e.g. towards outputting easily invertible values. With a concrete example already proposed (Pasquini et al., ACM CCS ’21), such training-hijacking attacks present a significant risk for the data privacy of split learning clients.

We propose two methods for a split learning client to detect if it is being targeted by a training-hijacking attack or not. We experimentally evaluate our methods’ effectiveness, compare them with other potential solutions, and discuss various points related to their use. Our conclusion is that by using the method that best suits their use case, split learning clients can consistently detect training-hijacking attacks and thus keep the information gained by the attacker at a minimum.

https://arxiv.org/abs/2108.09052

## Related and Follow-up Research

Apart from SplitGuard, we offer a **newer** detection and defense method called SplitOut. Please check our **new paper**: **[SplitOut: Out-of-the-Box Training-Hijacking Detection in Split Learning via Outlier Detection](https://github.com/ege-erdogan/splitout)**

## Cite Our Papers
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

```
@article{erdogan2023defense,
  title={Defense Mechanisms Against Training-Hijacking Attacks in Split Learning},
  author={Erdogan, Ege and Teksen, Unat and Celiktenyildiz, Mehmet Salih and Kupcu, Alptekin and Cicek, A Ercument},
  journal={arXiv preprint arXiv:2302.08618},
  year={2023}
}
```
