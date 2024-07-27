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

//basic param
#define M 1024
#define N 1024
#define K 1024
#define DIFF 1e-5

//CUDA error check
constexpr void checkerror(const char *msg)
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
			if (diff > DIFF) cnt++;
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


/// <summary>
/// matrix multiplication by CUBLAS
/// </summary>
/// <param name="A">=matrix A</param>
/// <param name="B">=matrix B</param>
/// <param name="C">=matrix C</param>
/// /// <param name="matsize">size of matrix</param>
void matrixMulGPU(const float *A, const float *B, float *C, matSize &matsize)
{

}

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
	srand(time(nullptr));
	unsigned int sizeA = matsize.hA * matsize.wA;
	unsigned int sizeB = matsize.hB * matsize.wB;
	unsigned int sizeC = matsize.hC * matsize.wC;
	unsigned int nByteA = sizeA * sizeof(float);
	unsigned int nByteB = sizeB * sizeof(float);
	unsigned int nByteC = sizeC * sizeof(float);

	//define matrix
	float* hA = (float*)malloc(nByteA);
	float* hB = (float*)malloc(nByteB);
	float* hC = (float*)malloc(nByteC);
	float* hC_from_gpu = (float*)malloc(nByteC);
	float* dA, * dB, * dC;
	cudaMallocAsync(&dA, nByteA, cudaStreamPerThread);
	cudaMallocAsync(&dB, nByteB, cudaStreamPerThread);
	cudaMallocAsync(&dC, nByteC, cudaStreamPerThread);

	//init matrix
	randomInit(hA, sizeA);
	randomInit(hB, sizeB);
	memset(hC, 0, nByteC);

	//calculate by CPU
	matrixMulCPU(hA, hB, hC, matsize);

	//calculate by GPU

	//compare result

	//free matrix
	free(hA);
	free(hB);
	free(hC);
	cudaFreeAsync(dA, cudaStreamPerThread);
	cudaFreeAsync(dB, cudaStreamPerThread);
	cudaFreeAsync(dC, cudaStreamPerThread);
	
	return 0;
}