# SGEMM
  GEMM does matrix multiplication. There are SGEMM(single-precision), HGEMM(half-precision) etc. The algorithm runs on GPU with CUDA or Tensor cores pretty fast. 
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
  For my gpu, the ratio of computing and data loading is 15.82T/432G = 36.6(without L2 cache hit).
  For algorithm, the ratio is defined as dataComputes/dataLoads.
  If Ralgo>Rgpu, the gmem is no longer a problem.
      
## Reference
  [YHs_Sample](https://github.com/Yinghan-Li/YHs_Sample/tree/master)
