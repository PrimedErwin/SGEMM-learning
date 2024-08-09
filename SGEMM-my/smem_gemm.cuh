#pragma once

namespace Smem_GEMM
{
	__global__
		void matrixMul(const float* A, const float* B, float* C,
			const int M, const int N, const int K);

	void naiveSmemGemm(const float* A, const float* B, float* C,
		const int M, const int N, const int K);

}

namespace Fixed_Smem_GEMM
{
	__global__
		void matrixMul(const float* A, const float* B, float* C,
			const int M, const int N, const int K);

	void naiveSmemGemm(const float* A, const float* B, float* C,
		const int M, const int N, const int K);

}
