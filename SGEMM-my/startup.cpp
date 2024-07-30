//This file tests naive gemm to efficient gemm

#include <cstdio>
#include <iostream>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//basic param
#define matM 1024
#define matN 2048
#define matK 512
#define DIFF 1e-3

void checkerror(const char* msg)
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
				if (cnt < 21) fprintf(stderr, "hC result = %f\tdC result = %f\n", hC[index], dC[index]);
			}
		}
	}
	fprintf(stderr, "Error result count = %d\n", cnt);
}

extern void naiveSmemGemm(const float* A, const float* B, float* C,
	const int M, const int N, const int K);

int main(int argc, char **argv)
{



	return 0;
}