#pragma once

namespace Db_GEMM
{
	__global__
		void matrixMul(const float* A, const float* B, float* C,
			const int M, const int N, const int K);

	void naiveSmemGemm(const float* A, const float* B, float* C,
		const int M, const int N, const int K);

}
namespace Db_exp_GEMM
{
	__global__
		void matrixMul(const float* A, const float* B, float* C,
			const int M, const int N, const int K);

	void naiveSmemGemm(const float* A, const float* B, float* C,
		const int M, const int N, const int K);

}