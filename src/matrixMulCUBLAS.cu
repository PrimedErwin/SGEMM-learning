//This is a CUBLAS matrix multiplication program
//running on WINDOWS

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__


#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <Windows.h>

//basic param
#define M 1024
#define N 2048
#define K 512
#define DIFF 1e-3

//CUDA error check
void checkerror(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA error at %d: ", __LINE__);
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
				if(cnt<21) fprintf(stderr, "hC result = %f\tdC result = %f\n", hC[index], dC[index]);
			}
		}
	}
	fprintf(stderr, "Error result count = %d\n", cnt);
}


/// <summary>
/// Performs small matrix mul by CPU for result checking
/// </summary>
/// <param name="A">=matrix A</param>
/// <param name="B">=matrix B</param>
/// <param name="C">=matrix C</param>
/// <param name="hA">=M, height of A</param>
/// <param name="wB">=N, width of B</param>
/// <param name="wA">=K, width of A, height of B</param>
void matrixMulCPU(const float* A, const float* B, float* C, matSize &matsize)
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

#pragma comment (lib, "cublas.lib")
int main(int argc, char** argv)
{
	printf("Starting CUBLAS matrix multiplication on device 0: ");
	int devID = 0;
	cudaSetDevice(devID);
	cudaDeviceProp Prop;
	cudaGetDeviceProperties(&Prop, devID);
	printf("\"%s\" compute capability %d.%d\n", Prop.name, Prop.major, Prop.minor);
	
	//basic things
	matSize matsize(K, M, N, K, M, N);
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
	static const float alpha = 1.0f;
	static const float beta = .0f;

	//basic stopwatch
	LARGE_INTEGER IpFreq, IpStart, IpEnd;
	cudaEvent_t cuStart, cuEnd;
	cudaEventCreate(&cuStart);
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

	//calculate by CPU
	QueryPerformanceCounter(&IpStart);
	matrixMulCPU(hA, hB, hC, matsize);
	QueryPerformanceCounter(&IpEnd);
	cpu_time = (double)(IpEnd.QuadPart - IpStart.QuadPart) * 1e3 / IpFreq.QuadPart;
	printf("CPU time cost %.4f ms\n\n", cpu_time);
	cudaDeviceSynchronize();
	checkerror("Sync error");

	//CUBLAS setup
	cublasHandle_t cublasHandle;
	cublasCreate_v2(&cublasHandle);
	checkerror("CUBLAS create error");

	//calculate by GPU, warmup first
	cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		matsize.wB, matsize.hA, matsize.wA,
		&alpha, dB, matsize.wB,
		dA, matsize.wA, &beta,
		dC, matsize.wB);
	checkerror("CUBLAS warmup failed");

	//calculate by GPU
	cudaEventRecord(cuStart);
	for (int i = 0; i < 50; i++)
	{
		cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			matsize.wB, matsize.hA, matsize.wA,
			&alpha, dB, matsize.wB,
			dA, matsize.wA, &beta,
			dC, matsize.wB);
	}
	cudaEventRecord(cuEnd);
	cudaEventSynchronize(cuEnd);
	cudaEventElapsedTime(&gpu_time, cuStart, cuEnd);
	checkerror("Event sync error");
	gpu_time /= 50;
	printf("GPU time cost %.4f ms\n\n", gpu_time);

	//compare result
	cudaMemcpy(hC_from_gpu, dC, nByteC, cudaMemcpyDeviceToHost);
	checkMulResult(hC, hC_from_gpu, matsize.wC, matsize.hC);
	double flops = (double)matsize.hA * (double)matsize.wA * (double)matsize.wB * 2.0;
	double gigaFlops = (flops * 1.0e-9) / (gpu_time / 1000.0);
	printf("FP32 %.4f TFlop/s, Ops %.4f Ops\n", gigaFlops/1000.0, flops);


	//free matrix
	free(hA);
	free(hB);
	free(hC);
	cudaFreeAsync(dA, cudaStreamPerThread);
	cudaFreeAsync(dB, cudaStreamPerThread);
	cudaFreeAsync(dC, cudaStreamPerThread);
	cublasDestroy(cublasHandle);
	cudaEventDestroy(cuStart);
	cudaEventDestroy(cuEnd);
	
	return 0;
}