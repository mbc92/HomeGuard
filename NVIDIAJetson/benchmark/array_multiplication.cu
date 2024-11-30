#include <iostream>
#include <cuda_runtime.h>



// CUDA kernel to multiply two arrays element-wise
__global__ void multiplyArrays(float *a, float *b, float *result, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure we do not access out-of-bounds memory
    if (index < size) {
        result[index] = a[index] * b[index];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate memory for the arrays on the host
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_result = new float[N];

    // Initialize the arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }

    // Allocate memory for the arrays on the device
    float *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_result, size);

    // Copy the input arrays from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the kernel with N threads, one per array element
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks

    multiplyArrays<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);

    // Check for any errors during kernel execution
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

    // Output some of the results to verify
    for (int i = 0; i < 10; i++) {  // Print first 10 results
        std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
    }

    // Free the memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    // Free the memory on the host
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;

    return 0;
}
