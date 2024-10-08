//This file implements a naive gemm based on block tiling
//registers are not influenced by fixed values.
//This file has been deprecated.

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "smem_gemm.cuh"

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
constexpr int M_num = 8;//8
constexpr int N_num = 8;//8
//smem size for each block
constexpr int smem_size_A = 1024;
constexpr int smem_size_B = 1024;
constexpr int smem_nByte = 2048 * sizeof(float);
//naive smem gemm func for startup.cpp
//void naiveSmemGemm(const float* A, const float* B, float* C,
//	const int M, const int N, const int K);

//gemm func
__global__
void Fixed_Smem_GEMM::matrixMul(const float* A, const float* B, float* C,
	const int M, const int N, const int K)
{
	//basic params
	const unsigned int bIdx = threadIdx.x + blockDim.x * threadIdx.y;//index in current block
		//baseX baseY assists the offset of block, reference in Readme.md
	const unsigned int baseX = blockIdx.x * blockDim.x * 8;//blockIDx128
	const unsigned int baseY = blockIdx.y * blockDim.y * 8;//
		//these params marks the storage position of current thread in tiled A, B
	const unsigned int rowA = bIdx / 2;//coord of Y axis
	const unsigned int colA = (bIdx & 1) * 4;//float4 makes K_block = 2, colA = 0,4
	const unsigned int rowB = bIdx / 32;//float4 makes N_block = 32
	const unsigned int colB = (bIdx * 4) % 128;//step 4 float same to colA
		//current block tile's base address
	const float* baseA = A + baseY * K;
	const float* baseB = B + baseX;//A and B need additional offset
	float* baseC = C + N * (baseY + threadIdx.y * 8) + threadIdx.x * 8 + baseX;
	//smem define, no double buffer, 4KB each
	__shared__ float matA[1024];
	__shared__ float matB[1024];
	//here we need M_num reg for A, N_num reg for B
	//M_num*N_num reg for C
	//float4 usage is vector request based on reference in Readme.md
	float4 regA[2];//128 bit vector request
	float4 regB[2];
	float regC[64] = { 0 };

	for (int i = 0; i < K / K_tile; i++)
	{
		//for each loop read tiled A and B from gmem
		//Ada can store data to smem directly without additional registers
		//but we need vector request in K-loop, so here still need explicit register

		//matB is easier.Directly put the value into smem
		regB[0] = *reinterpret_cast<const float4*>(baseB + rowB * N + colB + i * N * 8);
		*reinterpret_cast<float4*>(&matB[bIdx * 4]) = regB[0];
		//matA
		//in KMN loop A is read by column, bad
		//here we transpose it(and avoid bank conflict)
		regA[0] = *reinterpret_cast<const float4*>(baseA + rowA * K + colA + i * 8);
		//after transpose, the float4 become vertical, so add M_tile each element
		matA[rowA + (colA + 0) * M_tile] = regA[0].x;
		matA[rowA + (colA + 1) * M_tile] = regA[0].y;
		matA[rowA + (colA + 2) * M_tile] = regA[0].z;
		matA[rowA + (colA + 3) * M_tile] = regA[0].w;
		__syncthreads();

		//we have transposed matA, matB in smem
		for (int k = 0; k < K_tile; k++)
		{
			//e.g. tid.x=1,tid.y=0, this thread reads matA[0], matB[1](of float4)
			regA[0] = *reinterpret_cast<float4*>(&matA[threadIdx.y * M_num + k * M_tile]);
			regA[1] = *reinterpret_cast<float4*>(&matA[threadIdx.y * M_num + k * M_tile + 4]);
			regB[0] = *reinterpret_cast<float4*>(&matB[threadIdx.x * N_num + k * N_tile]);
			regB[1] = *reinterpret_cast<float4*>(&matB[threadIdx.x * N_num + k * N_tile + 4]);
			//for each thread 8x8
			for (int m = 0; m < 2; m++)
			{
				for (int n = 0; n < 2; n++)
				{
					//here inside we perform 8x8 multiplication for each thread
					//8x8=four 4x4(float4)
					//each float4 needs 4 FFMA, total 16 FFMA
					//and add it back in row-major in regC

					//row0
					regC[m * 4 * N_num + n * 4] += regA[m].x * regB[n].x;
					regC[m * 4 * N_num + n * 4 + 1] += regA[m].x * regB[n].y;
					regC[m * 4 * N_num + n * 4 + 2] += regA[m].x * regB[n].z;
					regC[m * 4 * N_num + n * 4 + 3] += regA[m].x * regB[n].w;
					//row1, +8
					regC[(m * 4 + 1) * N_num + n * 4] += regA[m].y * regB[n].x;
					regC[(m * 4 + 1) * N_num + n * 4 + 1] += regA[m].y * regB[n].y;
					regC[(m * 4 + 1) * N_num + n * 4 + 2] += regA[m].y * regB[n].z;
					regC[(m * 4 + 1) * N_num + n * 4 + 3] += regA[m].y * regB[n].w;
					//row2, +16
					regC[(m * 4 + 2) * N_num + n * 4] += regA[m].z * regB[n].x;
					regC[(m * 4 + 2) * N_num + n * 4 + 1] += regA[m].z * regB[n].y;
					regC[(m * 4 + 2) * N_num + n * 4 + 2] += regA[m].z * regB[n].z;
					regC[(m * 4 + 2) * N_num + n * 4 + 3] += regA[m].z * regB[n].w;
					//row3, +24
					regC[(m * 4 + 3) * N_num + n * 4] += regA[m].w * regB[n].x;
					regC[(m * 4 + 3) * N_num + n * 4 + 1] += regA[m].w * regB[n].y;
					regC[(m * 4 + 3) * N_num + n * 4 + 2] += regA[m].w * regB[n].z;
					regC[(m * 4 + 3) * N_num + n * 4 + 3] += regA[m].w * regB[n].w;

				}
			}
		}
		//sync block after each tiled block was calculated
		__syncthreads();
	}
	//all the results are now in regC of each block
	//store them back into C in gmem
	//store regC with float4 needs 8x2 times
	for (int m = 0; m < M_num; m++)
	{
		for (int n = 0; n < N_num / 4; n++)
		{
			*reinterpret_cast<float4*>(&baseC[m * N + n * 4]) = *reinterpret_cast<float4*>(&regC[m * M_num + n * 4]);
		}
	}
}




void Fixed_Smem_GEMM::naiveSmemGemm(const float* A, const float* B, float* C,
	const int M, const int N, const int K)
{
	//define block and grid, 16x16 thread block processes 128x128 val in matrix C
	dim3 blocksize(blockSize, blockSize);
	dim3 gridSize((N - 1) / N_tile + 1, (M - 1) / M_tile + 1);
	Fixed_Smem_GEMM::matrixMul << <gridSize, blocksize >> > (A, B, C, M, N, K);
}