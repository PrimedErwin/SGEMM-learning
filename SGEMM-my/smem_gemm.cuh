#pragma once

namespace Smem_GEMM
{
	void naiveSmemGemm(const float* A, const float* B, float* C,
		const int M, const int N, const int K);

}