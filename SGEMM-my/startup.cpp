//This file tests naive gemm to efficient gemm

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <Windows.h>

#include <nvtx3/nvToolsExt.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "smem_gemm.cuh"
#include "warp_op_gemm.cuh"
#include "db_gemm.cuh"

#define CUBLAS_COMPARISION

//basic param
#define matM 2560
#define matN 2560
#define matK 2560
#define DIFF 1e-3

#define checkerror(msg) checkerrors(msg, __FILE__, __LINE__)

void checkerrors(const char* msg, const char* file = NULL, const int line = -1)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA error at %s(%d): ", file, line);
		fprintf(stderr, "%s, %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	return;
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; i++) data[i] = rand() / (float)RAND_MAX;
}

struct matSize
{
	int wA, wB, wC, hA, hB, hC;
	matSize(int wwA = 0, int hhA = 0, int wwB = 0, int hhB = 0, int wwC = 0, int hhC = 0) :wA(wwA),
		hA(hhA), wB(wwB), hB(hhB), wC(wwC), hC(hhC) {}
};

void matrixMulCPU(const float* A, const float* B, float* C, matSize& matsize)
{
	double sum;
	for (int k = 0; k < matsize.wA; k++)
	{
		for (int m = 0; m < matsize.hA; m++)
		{
			for (int n = 0; n < matsize.wB; n++)
			{
				sum = A[m * matsize.wA + k] * B[k * matsize.wB + n];
				C[m * matsize.wB + n] += (float)sum;
			}
		}
	}
}

void checkMulResult(float* hC, float* dC, int width, int height)
{
	int cnt = 0;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int index = j * width + i;
			float diff = fabs(hC[index] - dC[index]);
			if (diff > DIFF)
			{
				cnt++;
				if (cnt < 21) fprintf(stderr, "hC result = %f\tdC result = %f\n", hC[index], dC[index]);
			}
		}
	}
	fprintf(stderr, "Error result count = %d\n", cnt);
}

#pragma comment (lib, "cublas.lib")

//extern void naiveSmemGemm(const float* A, const float* B, float* C,
//	const int M, const int N, const int K);

int main(int argc, char** argv)
{
	printf("Starting mySGEMM on device 0: ");
	int devID = 0;
	cudaSetDevice(devID);
	cudaDeviceProp Prop;
	cudaGetDeviceProperties(&Prop, devID);
	printf("\"%s\" compute capability %d.%d\n", Prop.name, Prop.major, Prop.minor);

	//basic things
	matSize matsize(matK, matM, matN, matK, matM, matN);
	printf("Matrix size: \n\
		%d x %d * %d x %d\n",
		matsize.hA, matsize.wA, matsize.hB, matsize.wB);
	srand(time(nullptr));
	unsigned int sizeA = matsize.hA * matsize.wA;
	unsigned int sizeB = matsize.hB * matsize.wB;
	unsigned int sizeC = matsize.hC * matsize.wC;
	unsigned int nByteA = sizeA * sizeof(float);
	unsigned int nByteB = sizeB * sizeof(float);
	unsigned int nByteC = sizeC * sizeof(float);
	//static const float alpha = 1.0f;
	//static const float beta = .0f;

	//basic stopwatch
	LARGE_INTEGER IpFreq, IpStart, IpEnd;
	cudaEvent_t cuStart, cuEnd, cuEnd_CUBLAS;
	cudaEventCreate(&cuStart);
	cudaEventCreate(&cuEnd_CUBLAS);
	cudaEventCreate(&cuEnd);
	float cpu_time, gpu_time;
	QueryPerformanceFrequency(&IpFreq);

	//define matrix
	float* hA = (float*)malloc(nByteA);
	float* hB = (float*)malloc(nByteB);
	float* hC = (float*)malloc(nByteC);
	float* hC_from_gpu = (float*)malloc(nByteC);
	float* dA, * dB, * dC;
	cudaMallocAsync(&dA, nByteA, cudaStreamPerThread);
	cudaMallocAsync(&dB, nByteB, cudaStreamPerThread);
	cudaMallocAsync(&dC, nByteC, cudaStreamPerThread);
	checkerror("Malloc Error");

	//init matrix
	randomInit(hA, sizeA);
	randomInit(hB, sizeB);
	memset(hC, 0, nByteC);
	cudaMemcpyAsync(dA, hA, nByteA, cudaMemcpyHostToDevice, cudaStreamPerThread);
	cudaMemcpyAsync(dB, hB, nByteB, cudaMemcpyHostToDevice, cudaStreamPerThread);


#ifndef CUBLAS_COMPARISION
	//calculate by CPU
	QueryPerformanceCounter(&IpStart);
	matrixMulCPU(hA, hB, hC, matsize);
	QueryPerformanceCounter(&IpEnd);
	cpu_time = (double)(IpEnd.QuadPart - IpStart.QuadPart) * 1e3 / IpFreq.QuadPart;
	printf("CPU time cost %.4f ms\n\n", cpu_time);
	cudaDeviceSynchronize();
	checkerror("Sync error");
	//calculate by GPU, warmup first
	Db_exp_GEMM::naiveSmemGemm(dA, dB, dC, matsize.hA, matsize.wB, matsize.wA);
	checkerror("mycublas error");
	//calculate by GPU
	nvtxRangeId_t range = nvtxRangeStart("ProfileA");
	cudaEventRecord(cuStart);
	for (int i = 0; i < 1; i++)
	{
		Db_exp_GEMM::naiveSmemGemm(dA, dB, dC, matsize.hA, matsize.wB, matsize.wA);
		//NC_WarpOp_GEMM::naiveSmemGemm(dA, dB, dC, matsize.hA, matsize.wB, matsize.wA);
	}
	cudaEventRecord(cuEnd);
	cudaEventSynchronize(cuEnd);
	cudaEventElapsedTime(&gpu_time, cuStart, cuEnd);
	checkerror("Event sync error");
	nvtxRangeEnd(range);
	gpu_time /= 1;
	printf("GPU time cost %.4f ms\n\n", gpu_time);

	//compare result
	cudaMemcpy(hC_from_gpu, dC, nByteC, cudaMemcpyDeviceToHost);
	checkMulResult(hC, hC_from_gpu, matsize.wC, matsize.hC);
	double flops = (double)matsize.hA * (double)matsize.wA * (double)matsize.wB * 2.0;
	double gigaFlops = (flops * 1.0e-9) / (gpu_time / 1000.0);
	printf("FP32 %.4f TFlop/s, Ops %.4f Ops\n", gigaFlops / 1000.0, flops);
#endif // !CUBLAS_COMPARISION


#ifdef CUBLAS_COMPARISION
	cudaDeviceSynchronize();
	static const float alpha = 1.0f;
	static const float beta = .0f;
	printf("\nStarting CUBLAS comparision on device 0: \n");
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);
	checkerror("CUBLAS create error");
	cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		matsize.wB, matsize.hA, matsize.wA,
		&alpha, dB, matsize.wB,
		dA, matsize.wA, &beta,
		dC, matsize.wB);
	cudaEventRecord(cuStart);
	for (int i = 0; i < 50; i++)
	{
		Db_exp_GEMM::naiveSmemGemm(dA, dB, dC, matsize.hA, matsize.wB, matsize.wA);
	}
	cudaEventRecord(cuEnd);
	cudaEventSynchronize(cuEnd);
	cudaEventElapsedTime(&gpu_time, cuStart, cuEnd);
	gpu_time /= 50;
	printf("MyGEMM time cost %.4f ms\n", gpu_time);
	double flops = (double)matsize.hA * (double)matsize.wA * (double)matsize.wB * 2.0;
	double gigaFlops = (flops * 1.0e-9) / (gpu_time / 1000.0);
	printf("FP32 %.4f TFlop/s, Ops %.4f Ops\n\n", gigaFlops / 1000.0, flops);

	cudaEventRecord(cuStart);
	for (int i = 0; i < 50; i++)
	{
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			matsize.wB, matsize.hA, matsize.wA,
			&alpha, dB, matsize.wB,
			dA, matsize.wA, &beta,
			dC, matsize.wB);
		//cublasSgemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		//	matsize.wB, matsize.hA, matsize.wA,
		//	&alpha, dB, (cudaDataType_t)0, matsize.wB,
		//	dA, (cudaDataType_t)0, matsize.wA, &beta,
		//	dC, (cudaDataType_t)0, matsize.wA
		//);
	}
	cudaEventRecord(cuEnd_CUBLAS);
	cudaEventSynchronize(cuEnd_CUBLAS);
	cudaEventElapsedTime(&gpu_time, cuStart, cuEnd_CUBLAS);
	checkerror("Event sync error");
	gpu_time /= 50;
	printf("CUBLAS time cost %.4f ms\n", gpu_time);
	flops = (double)matsize.hA * (double)matsize.wA * (double)matsize.wB * 2.0;
	double gigaFlops_CUBLAS = (flops * 1.0e-9) / (gpu_time / 1000.0);
	printf("FP32 %.4f TFlop/s, Ops %.4f Ops\n", gigaFlops_CUBLAS / 1000.0, flops);
	printf("\n**********************\nResult: %.4f%% CUBLAS\n", gigaFlops/gigaFlops_CUBLAS*1e2);
#endif // CUBLAS_COMPARISION


	//free matrix
	free(hA);
	free(hB);
	free(hC);
	cudaFreeAsync(dA, cudaStreamPerThread);
	cudaFreeAsync(dB, cudaStreamPerThread);
	cudaFreeAsync(dC, cudaStreamPerThread);
	cudaEventDestroy(cuStart);
	cudaEventDestroy(cuEnd);
	cudaEventDestroy(cuEnd_CUBLAS);

	return 0;
}