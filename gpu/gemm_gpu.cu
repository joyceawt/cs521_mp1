#include <cuda_runtime.h>

#include "../include/utils.h"

#define NUM_RUNS 10
#define TILE_WIDTH 16  // also block size
#define BLOCK_SIZE 32

#define CUDA_CHECK(func)                                                   \
  do {                                                                     \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CHECK(name)                                                           \
  float *d_Aref_##name, *d_Bref_##name, *d_Cref_##name;                       \
  std::cerr << "checking " << #name << std::endl;                             \
  CUDA_CHECK(cudaMalloc(&d_Aref_##name, Ref::M* Ref::K * sizeof(float)));     \
  CUDA_CHECK(cudaMalloc(&d_Bref_##name, Ref::K* Ref::N * sizeof(float)));     \
  CUDA_CHECK(cudaMalloc(&d_Cref_##name, Ref::M* Ref::N * sizeof(float)));     \
  CUDA_CHECK(cudaMemcpy(d_Aref_##name, ref.A, Ref::M* Ref::K * sizeof(float), \
                        cudaMemcpyHostToDevice));                             \
  CUDA_CHECK(cudaMemcpy(d_Bref_##name, ref.B, Ref::K* Ref::N * sizeof(float), \
                        cudaMemcpyHostToDevice));                             \
  float* d_Cref_INI_##name = new float[M * N]();                              \
  for (int i = 0; i < Ref::M; i++) {                                          \
    for (int j = 0; j < Ref::N; j++) {                                        \
      d_Cref_INI_##name[i * Ref::N + j] = 0;                                  \
    }                                                                         \
  }                                                                           \
  CUDA_CHECK(cudaMemcpy(d_Cref_##name, d_Cref_INI_##name,                     \
                        Ref::M* Ref::N * sizeof(float),                       \
                        cudaMemcpyHostToDevice));                             \
  name(d_Aref_##name, d_Bref_##name, d_Cref_##name, Ref::M, Ref::N, Ref::K);  \
  cudaError_t err_c_##name = cudaGetLastError();                              \
  if (err_c_##name != cudaSuccess) {                                          \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_##name)           \
              << std::endl;                                                   \
  }                                                                           \
  CUDA_CHECK(cudaMemcpy(refC, d_Cref_##name, Ref::M* Ref::N * sizeof(float),  \
                        cudaMemcpyDeviceToHost));                             \
  if (!ref.checkRef(refC)) {                                                  \
    std::cerr << "check ref failed!" << std::endl;                            \
  };

#define TIME(name)                                                          \
  float *d_A_##name, *d_B_##name, *d_C_##name;                              \
  CUDA_CHECK(cudaMalloc(&d_A_##name, M* K * sizeof(float)));                \
  CUDA_CHECK(cudaMalloc(&d_B_##name, K* N * sizeof(float)));                \
  CUDA_CHECK(cudaMalloc(&d_C_##name, M* N * sizeof(float)));                \
  CUDA_CHECK(cudaMemcpy(d_A_##name, A, M* K * sizeof(float),                \
                        cudaMemcpyHostToDevice));                           \
  CUDA_CHECK(cudaMemcpy(d_B_##name, B, K* N * sizeof(float),                \
                        cudaMemcpyHostToDevice));                           \
  cudaEvent_t start_##name, end_##name;                                     \
  cudaEventCreate(&start_##name);                                           \
  cudaEventCreate(&end_##name);                                             \
  float* d_C_INI_##name = new float[M * N]();                               \
  for (int i = 0; i < M; i++) {                                             \
    for (int j = 0; j < N; j++) {                                           \
      d_C_INI_##name[i * N + j] = 0;                                        \
    }                                                                       \
  }                                                                         \
  for (int i = 0; i < 2; i++) {                                             \
    CUDA_CHECK(cudaMemcpy(d_C_##name, d_C_INI_##name, M* N * sizeof(float), \
                          cudaMemcpyHostToDevice));                         \
    name(d_A_##name, d_B_##name, d_C_##name, M, N, K);                      \
  }                                                                         \
  cudaError_t err_t_##name = cudaGetLastError();                            \
  if (err_t_##name != cudaSuccess) {                                        \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_##name)         \
              << std::endl;                                                 \
  }                                                                         \
  float milliseconds_##name = 0;                                            \
  for (int i = 0; i < NUM_RUNS; i++) {                                      \
    CUDA_CHECK(cudaMemcpy(d_C_##name, d_C_INI_##name, M* N * sizeof(float), \
                          cudaMemcpyHostToDevice));                         \
    cudaDeviceSynchronize();                                                \
    cudaEventRecord(start_##name);                                          \
    name(d_A_##name, d_B_##name, d_C_##name, M, N, K);                      \
    cudaEventRecord(end_##name);                                            \
    cudaEventSynchronize(end_##name);                                       \
    float milliseconds_##i = 0;                                             \
    cudaEventElapsedTime(&milliseconds_##i, start_##name, end_##name);      \
    milliseconds_##name += milliseconds_##i;                                \
  }                                                                         \
  cudaMemcpy(C, d_C_##name, M* N * sizeof(float), cudaMemcpyDeviceToHost);  \
  std::cout << "Time taken for GEMM (GPU, " << #name                        \
            << "): " << milliseconds_##name / (float)NUM_RUNS << "ms"       \
            << std::endl;                                                   \
  cudaFree(d_A_##name);                                                     \
  cudaFree(d_B_##name);                                                     \
  cudaFree(d_C_##name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float* C, int M, int N,
                                   int K) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
  }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K) {
  // Init block and grid size
  dim3 blockSize(1);
  dim3 gridSize(1);
  gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
__global__ void gemm_gpu_o1_kernel(float* A, float* B, float* C, int M, int N,
                                   int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
void gemm_gpu_o1(float* A, float* B, float* C, int M, int N,
                 int K) {  // Init block and grid size
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);
  gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o2_kernel(float* A, float* B, float* C, int M, int N,
                                   int K) {
  // Initialized shared memory array As and Bs to store the sub-matrix of A and
  // B
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0f;

  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix (K sub-matrices/dimension)
  int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;  // handle partial tiles
  for (int t = 0; t < numTiles; t++) {
    // A sub-block; Load one tile of A into shared memory (if in bounds)
    int kA = t * TILE_WIDTH + threadIdx.x;  // column index of A
    if (row < M && kA < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + kA];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;  // if out of bounds, set to 0
    }

    // B sub-block; Load one tile of B into shared memory (if in bounds)
    int kB = t * TILE_WIDTH + threadIdx.y;  // row index of B
    if (kB < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Multiply the two sub-matrices for this tile
    for (int i = 0; i < TILE_WIDTH; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    };

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K) {
  // Init block and grid size
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);
  gemm_gpu_o2_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o3_kernel(float* A, float* B, float* C, int M, int N,
                                   int K) {
  // Initialized shared memory array As and Bs to store the sub-matrix of A and
  // B
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Loop over all the sub-matrices of A and B required to compute the block
  // sub-matrix (K sub-matrices/dimension)
  int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;  // handle partial tiles
  for (int t = 0; t < numTiles; t++) {
    // A sub-block; Load one tile of A into shared memory (if in bounds)
    int kA = t * BLOCK_SIZE + threadIdx.x;  // column index of A
    if (row < M && kA < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + kA];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;  // if out of bounds, set to 0
    }

    // B sub-block; Load one tile of B into shared memory (if in bounds)
    int kB = t * BLOCK_SIZE + threadIdx.y;  // row index of B
    if (kB < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B[kB * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Multiply the two sub-matrices for this tile
    for (int i = 0; i < BLOCK_SIZE; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    };

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K) {
  // Init block and grid size// Init block and grid size
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  gemm_gpu_o3_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
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

  /// GPU Implementation
  // Check if implementation is correct
  auto ref = Ref();
  float* refC = new float[Ref::M * Ref::N]();
  CHECK(gemm_gpu_o0)
  CHECK(gemm_gpu_o1)
  CHECK(gemm_gpu_o2)
  CHECK(gemm_gpu_o3)

  // Actual run
  TIME(gemm_gpu_o0)
  TIME(gemm_gpu_o1)
  TIME(gemm_gpu_o2)
  TIME(gemm_gpu_o3)

  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}