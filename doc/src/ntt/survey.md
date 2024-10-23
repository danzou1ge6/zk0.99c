# Survey
## 算法
- Cooley-Tukey
优点：可以原地进行更新，只需要n的空间来存取数据
缺点：在开始前需要做一次rearrange的操作，有较大overhead。
- Stockham
优点：将重排的过程融入到算法本身中，不需要额外的一次重排
缺点：由于每次读取的stride和写回的stride不同，无法原地更新，需要double buffer，即数据需要2n空间
- SELF-SORTING IN-PLACE FAST FOURIER TRANSFORMS  CLIVE TEMPERTON
优势：时间上：比起C-T算法，不需要做rearrange操作。
空间上：比起Stockham算法，只需要n的空间，而非2n，可以将多出来的空间用于存放预先计算好的twiddle factor。
## GPU实现
使用shared memory(Govindaraju et al., 2008)和使用warp shuffle(Durrani et al., 2021)来优化ntt对global memory的访问是已经被实现的做法，但之前的工作把这两种优化视为独立的两种优化，而没有把他们看作一个整体性的多层内存结构来同时使用两种优化。

通过融合不同线程间的蝴蝶操作(Wang et al., 2023)，block间的同步可以被减少，但我们的内核并不会使用这一技巧因为如下两个原因：第一，由于使用了warp shuffle的技巧，我们对同步的使用次数实际上是很小的，甚至会比进行进行融合蝴蝶操作后的次数更小。第二，由于我们针对的场景是ZKP的大整数，因此每个线程使用的寄存器数量已经很多（64个以上），融合多组蝴蝶操作意味着每个线程使用的寄存器数量更多，使得一个SM上的最大thread数降低，减少了一个SM上的warp数，使得通过warp间切换来掩盖访存变得困难，该理由同样使得增大radix来提高计算利用率的方法失效(Kim et al., 2020)。

在GZKP(Ma et al., 2023)中，作者通过使用将蝴蝶操作分组的方式来合并global访存，但这一方法尤其局限性：第一，想要使用这一方法需要数据在global里已经以col major的方式存储，而这与正常内存中数据的组织方式是不同的，意味着很可能在将数据传输到GPU上后需要再做一次转置的操作，开销反而更大。第二，想要将蝴蝶操作分组意味着每个block中可以支持的最大的蝴蝶操作的轮数减小，也就意味着每个kernel可以处理的蝴蝶操作的轮数减少，需要launch kernel的次数增多，增多global访存次数。
