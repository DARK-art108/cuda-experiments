#include <iostream>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>


namespace cg = cooperative_groups;

__global__
void SilU(float* matrix, int m, int n){

	unsigned int row = blockIdx.x;
	float* row_ptr = matrix + row * n;
    // check for 4 floats and 16 byte alignment
	bool vec_ok = (n % 4 == 0) &&
	              (reinterpret_cast<uintptr_t>(row_ptr) % sizeof(float4) == 0);
	int vec_end = vec_ok ? (n / 4) * 4 : 0;

	if (vec_ok) {
		float4* matrix4 = reinterpret_cast<float4*>(row_ptr);

		for(int col=threadIdx.x; col<n/4;col+=blockDim.x){
			float4 elements = matrix4[col];
	        //f(x) = x/(1 + e^-x)
			matrix4[col] = make_float4(
				elements.x/(1.0f+expf(-elements.x)),
				elements.y/(1.0f+expf(-elements.y)),
				elements.z/(1.0f+expf(-elements.z)),
				elements.w/(1.0f+expf(-elements.w))
				);
		}
	}

	for(int col=vec_end+threadIdx.x;col<n;col+=blockDim.x){
		float element = row_ptr[col];
		row_ptr[col] = element/(1.0f+expf(-element));
	}

}

int main() {
	const int M = 128;   
	const int N = 1027;  

	size_t bytes = M * N * sizeof(float);
	float* h_input  = new float[M * N];
	float* h_output = new float[M * N];

	for (int i = 0; i < M * N; i++) {
		h_input[i] = (i % 200 - 100) * 0.05f; 
	}

	float* d_matrix;
	cudaMalloc(&d_matrix, bytes);
	cudaMemcpy(d_matrix, h_input, bytes, cudaMemcpyHostToDevice);

	// 1 block per row, 256 threads per block
	int threads = 256;
	int blocks  = M;
	SilU<<<blocks, threads>>>(d_matrix, M, N);
	cudaDeviceSynchronize();
	cudaMemcpy(h_output, d_matrix, bytes, cudaMemcpyDeviceToHost);

	//print results 10 elements
	printf("First 10 elements:\n");
	printf("%-10s %-12s %-12s\n", "Index", "Input", "SiLU(x)");
	for (int i = 0; i < 10 && i < M * N; i++) {
		printf("%-10d %-12.6f %-12.6f\n", i, h_input[i], h_output[i]);
	}

	int mismatches = 0;
	for (int i = 0; i < M * N; i++) {
		float x = h_input[i];
		float expected = x / (1.0f + expf(-x));
		if (fabsf(h_output[i] - expected) > 1e-5f) {
			if (mismatches < 5) {
				printf("MISMATCH at [%d][%d]: got %.6f expected %.6f\n",
					   i / N, i % N, h_output[i], expected);
			}
			mismatches++;
		}
	}

	if (mismatches == 0)
		printf("OK: SiLU verified for %dx%d matrix\n", M, N);
	else
		printf("FAIL: %d mismatches\n", mismatches);

	cudaFree(d_matrix);
	delete[] h_input;
	delete[] h_output;
	return mismatches ? 1 : 0;
}