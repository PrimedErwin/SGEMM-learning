# SGEMM
  GEMM does matrix multiplication, mainly FFMA\(multiple and add\). There are SGEMM(single-precision), HGEMM(half-precision) etc. The algorithm runs on GPU with CUDA or Tensor cores pretty fast. 
  
  This is a project of my road to SGEMM. It uses CUDA core only. 
## What can SGEMM do
  matrix A * matrix B = matrix C. A lot of calculations contains this. I major in communication engineering in the university, convolution can be implemented with matrix multiplicaation(with im2col). It's also used in deep learning, neural networks(I haven't use them yet).
## Platform
  i9-13950HX
  
  RTX3500Ada
  
  64GB DDR5
## Physical limitations
  Chasing the speed of CUBLAS and hardware limitations. 
  - TODO
  CUBLAS will be used in `simpleCUBLAS.cu` for comparision.
    ### Hardware limitations of RTX3500Ada(no boost)
      RTX3500Ada: 5120 CUDA cores + 1545 MHz clock
    
      Single precision floating-point performance = 5120*2*1545*1e6/1e9 = 15.82 TFLOPS
## Implementation
### Core: How to reach the hardware limitation
  Hiding the latency is important. Threads can't just wait for data, they should always be computing.
- From gmem to smem

  For my gpu, the ratio of computing and data loading is 15.82T/432G = 36.6(without L2 cache hit).
  
  For algorithm, the ratio is defined as dataComputes/dataLoads.
  
  If Ralgo>Rgpu, the gmem is no longer a problem.

  Hint: Matrix multiplication is A's row * B's col, so when you store one block into smem, L2 cache will probably be hit when you read the next time. Hit ratio arises especially for matrix A, because the same row reads the same block.
- From smem to reg

  Smem is fast enough. Why still register is needed? Due to [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)\(CUDA 12.4, Chapter3.2.4\), The matrix multiplication is performed by smem directly. In smem, matrix A is read by row, but matrix B is read by col. Though here are no bank conflicts, each time only one element needed in matrix B, which results waste of other 31 bytes. So even smem is running with full bandwidth, matrix B still wastes a lot of bandwidth.

  So based on the reference\(end of the current doc\), we need to consider latency here. Assuming that it's fully running with pipeline due to the latency of smem. Each cycle the smem pops data out\(might be 128 bytes, depends on your gpu\), the CUDA core computes. But a lot of data is wasted, the CUDA core could perform 128\(still depends on your gpu\) FFMA ops, now it can only perform several. So read by row in smem is bad, we need registers.

  
## Reference
  [YHs_Sample](https://github.com/Yinghan-Li/YHs_Sample/tree/master)
