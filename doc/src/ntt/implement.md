# Implement

NTT的实现在计算量上几乎是无法改进的，因此主要优化的点在于访存的模式和次数，这就与具体目标架构息息相关。因此我们的NTT实现主要都集中在如何优化访存模式。简单来说，我们的NTT使用self sort in place算法来消除Cooley-Tukey算法中单独的一次shuffle操作，同时比起Stockham算法来说，不需要额外一倍的内存开销。为了减少内存开销，我们充分利用了warp上的shuffle操作，block上的shared memory，并优化了对global和shared的访存模式，同时做到了对读入数据没有额外的格式要求。

## 预处理twiddle factor

Step1: 使用cub::DeviceCopy::Batched得到[1, w, w, …, w]

Step2: 使用cub::DeviceScan::InclusiveScan做一次前缀积，得到[1, w, w^2, …, w^n]

## 分割任务

由于stage1和stage2各自需要处理deg/2层，且stage1和stage2每次最多处理的层数不同，因此需要对deg进行划分来使得每次处理的层数尽可能平均，防止出现1个kernel只处理1层的情况。
## stage1 shared memory布局

由于stage1单次最大处理的deg是8-11，因此需要利用shared memory。为了减少bank conflict, shared memory中数据的存储方式是col major的即：

a0_word0 a1_word0 a2_word0 a3_word0 ... an_word0 [empty] a0_word1 a1_word1 a2_word1 a3_word1 ...

此时一个warp读取数字时lane间的步长就是下标的步长，而非row major时下标的步长*8，因此可以减少bank conflict。

而在an_word0后插入空格是因为从global load时的顺序是a0_word 0,1,2,…, 而n又是2^k，因此插入空格减少读取时的bank conflict。

## stage1从global读到shared

为了合并global访问，这里利用了大整数所占字节数多的特点。

Thread0-7会分别读入a0_word0, 1, 2 … 7，因此对global的访问是连续的从而合并了global访存。

## 进一步消除load时的bank conflict
按照当前read的模式，假设deg=5，那么load时thread0-7访问的shared memory是

0, 65, 130…, 1, 66, 131. …, …, 7, …

Thread 16-23访问的是

32, 97…, 因此需要让16-32的访问顺序调转。

## 使用warp shuffle减少shared读取次数

Warp shuffle xor可以模拟蝴蝶操作，因此一个warp内的蝴蝶操作都可以用shuffle完成，让shared memory读取进一步减少，其中lanemask控制shuffle的步长，对应了蝴蝶操作的步长。

## stage2完全不使用shared memory

由于stage2的max deg=6，因此一个warp内正好可以完成所有的蝴蝶操作，因此不需要使用shared memory。因此，为了合并global访问，需要做一次显式的数据重排。

## stage2去除__syncthreads

由于不需要依赖shared memory，因此只需要保证在所有数据读入后，才发生第一次写回即可，因此可以使用一个barrier而非__syncthreads，减少同步开销。

## 恒定block内的线程数

由于stage2的max deg=6，因此一个warp内正好可以完成所有的蝴蝶操作，因此不需要使用shared memory。因此，为了合并global访问，需要做一次显式的数据重排。
