//This file implements a naive gemm based on block tiling
//ê€§œ§…§≥§´§È§‰§√§∆§Ø§Î§Œ§«§∑§Á§¶

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//*********************
//Hereby set the params
// ********************
//matrix A=MxK, matrix B=KxN, row-major
//16x16 block computes 128x128 matrix
//1 thread computes 8x8
constexpr int blockSize = 16;
constexpr int M_tile = 128;
constexpr int N_tile = 128;
constexpr int K_tile = 8;
constexpr int M_num = M_tile / blockSize;//8
constexpr int N_num = N_tile / blockSize;//8
//smem size for each block
constexpr int smem_size_A = M_tile * K_tile;
constexpr int smem_size_B = K_tile * N_tile;
constexpr int smem_nByte = (smem_size_A + smem_size_B) * sizeof(float);
//naive smem gemm func for startup.cpp
void naiveSmemGemm(const float* A, const float* B, float* C,
	const int M, const int N, const int K);

//gemm func
__global__
void matrixMul(const float* A, const float* B, float* C,
	const int M, const int N, const int K)
{
	//basic params
	const unsigned int bIdx = threadIdx.x + blockDim.x * threadIdx.y;//index in current block
		//baseX baseY assists the offset of block, reference in Readme.md
	const unsigned int baseX = blockIdx.x * blockDim.x * N_num;//blockIDx128
	const unsigned int baseY = blockIdx.y * blockDim.y * M_num;//
		//these params marks the storage position of current thread in tiled A, B
	const unsigned int rowA = bIdx / 2;//coord of Y axis
	const unsigned int colA = (bIdx & 1) * 4;//float4 makes K_block = 2, colA = 0,4
	const unsigned int rowB = bIdx / 32;//float4 makes N_block = 32
	const unsigned int colB = (bIdx * 4) % N_tile;//step 4 float same to colA
		//current block tile's base address
	const float* baseA = A + baseY * K;
	const float* baseB = B + baseX;//A and B need additional offset
	float* baseC = C + N * (baseY + threadIdx.y * N_num) + threadIdx.x * M_num + baseX;
	//smem define, no double buffer, 4KB each
	__shared__ float matA[M_tile * K_tile];
	__shared__ float matB[K_tile * N_tile];
	//here we need M_num reg for A, N_num reg for B
	//M_num*N_num reg for C
	//float4 usage is vector request based on reference in Readme.md
	float4 regA[M_num / 4];//128 bit vector request
	float4 regB[N_num / 4];
	float regC[M_num * N_num];

	for (int i = 0; i < K / K_tile; i++)
	{
		//for each loop read tiled A and B from gmem
		//Ada can store data to smem directly without additional registers
		//but we need vector request in K-loop, so here still need explicit register

		//matB is easier.Directly put the value into smem
		regB[0] = *reinterpret_cast<const float4*>(baseB + rowB * N + colB + i * N * K_tile);
		*reinterpret_cast<float4*>(&matB[bIdx * 4]) = regB[0];
		//matA
		//in KMN loop A is read by column, bad
		//here we transpose it(and avoid bank conflict)
		regA[0] = *reinterpret_cast<const float4*>(baseA + rowA * K + colA + i * K_tile);
		
	}
}




void naiveSmemGemm(const float* A, const float* B, float* C,
	const int M, const int N, const int K)
{
	//define block and grid, 16x16 thread block processes 128x128 val in matrix C
	dim3 blocksize(blockSize, blockSize);
	dim3 gridSize((N - 1) / N_tile + 1, (M - 1) / M_tile + 1);
	matrixMul << <gridSize, blocksize >> > (A, B, C, M, N, K);
}