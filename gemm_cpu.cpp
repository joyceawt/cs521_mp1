//
// Created by damitha on 1/29/25.
//
#include <chrono>
#include "utils.h"

void gemm_cpu(float* A, float* B, float *C, int M, int N, int K) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k < K; k++) {
                C[i * N + j]  += A[i * K + k]  * B[k * N + j];
			}
		}
	}
}

// The scafolding for optimized GEMM implementations
void gemm_cpu_opt(float* A, float* B, float *C, int M, int N, int K) {

}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);

	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// CPU Implementation
    // Check if the result is correct
	float* refC = new float[Ref::M * Ref::N]();
    auto ref = Ref();
	gemm_cpu(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);
    if (!ref.checkRef(refC)){
      std::cerr << "check ref failed!" << std::endl;
    };
	// Skip the first 5 runs
	for (int i = 0; i < 5; i++)
	{
		gemm_cpu(A, B, C, M, N, K);
	}
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++)
	{
		gemm_cpu(A, B, C, M, N, K);
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end_time - start_time;
	std::cout << "Time taken for GEMM (CPU,unoptimized): " << duration.count() << "ms" << std::endl;

	/// Optimized CPU Implementation
	// Skip the first 5 runs.
	for (int i = 0; i < 5; i++)
	{
		gemm_cpu_opt(A, B, C, M, N, K);
	}
	auto start_time_opt = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++)
	{
		gemm_cpu_opt(A, B, C, M, N, K);
	}
	auto end_time_opt = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration_opt = end_time_opt - start_time_opt;
	std::cout << "Time taken for GEMM (CPU,optimized): " << duration_opt.count() << "ms" << std::endl;

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}