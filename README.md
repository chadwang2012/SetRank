# SetRank

This is our implementation for the paper:

Chao Wang, Hengshu Zhu, Chen Zhu, Chuan Qin and Hui Xiong (2020). SetRank: A Setwise Bayesian Approach for Collaborative Ranking from Implicit Feedback. In Proceedings of AAAI'20, New York, New York, USA, February 7-12, 2020.

Please cite our AAAI'20 paper if you use our codes. Thanks!

The code has been tested running under Python 3.6.5 (tensorflow 1.11.0) and Julia 0.6.4.

The dataset in this file was provided by https://github.com/wuliwei9278/SQL-Rank.

We provide two implementations, MF-SetRank and Deep-SetRank. 

You can run the code for MF-SetRank like this:

```Julia
julia MF-SetRank.jl
```

You can run the code for Deep-SetRank like this:

```Python
python sparse-Deep-SetRank.py -print 10 -reg 1.8 -lr 0.0001 -negnum 30 -posnum 20
```
