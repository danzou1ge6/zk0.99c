# NTT

## Overview

NTT的实现在计算量上几乎是无法改进的，因此主要优化的点在于访存的模式和次数，这就与具体目标架构息息相关。
因此我们的NTT实现主要都集中在如何优化访存模式。

简单来说，我们的NTT使用self sort in place算法来消除Cooley-Tukey算法中单独的一次shuffle操作，同时比起
Stockham算法来说，不需要额外一倍的内存开销。

为了减少内存开销，我们充分利用了warp上的shuffle操作，block上的shared memory，并优化了对global和shared的
访存模式，同时做到了对读入数据没有额外的格式要求。