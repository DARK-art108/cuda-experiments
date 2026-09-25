#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 2

__global__
void matmul_tiled(float *A, float *B, float *C, int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main()
{
    const int N = 4;
    const int matsize = 16;

    float A[matsize] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float B[matsize] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float C[matsize] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matsize * sizeof(float));
    cudaMalloc(&d_B, matsize * sizeof(float));
    cudaMalloc(&d_C, matsize * sizeof(float));

    cudaMemcpy(d_A, A, matsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matsize * sizeof(float), cudaMemcpyHostToDevice);

// grid  = (2,2) → 4 blocks
// block = (2,2) → 4 threads/block
// total = 4 × 4 = 16 threads


    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, matsize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%6.0f", C[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
