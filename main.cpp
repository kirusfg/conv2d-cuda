#include "cublas_v2.h"
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "utils.hpp"

// N - number of images
// C - number of channels
// H - height of the image
// W - width of the image
// K - number of filters (kernels)
// KH - height of the filters
// KW - width of the filters
// P - height of the output image
// Q - width of the output image
// SH - stride height
// SW - stride width
// left, right, top, bottom - amount of zero padding
// BS - block size
// V - 1 for mmGlobal, 2 for mmShared, 3 for cublasSgemm
typedef struct {
    int N;
    int C, H, W;
    int K, P, Q;
    int KH, KW;
    int SH, SW;
    int left, right, top, bottom;
    int BS;
    int V;
} Config;

std::vector<Config> config {
    //N,C,H,W   K,P,Q   KH,KW  SH,SW   L,R,T,B   BS    V
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   1 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   2 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   3 },

    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   1 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   2 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   3 },

    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   1 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   2 },
    { 1,1,512,512,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   3 },


    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   1 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   2 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   3 },

    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   1 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   2 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   3 },

    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   1 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   2 },
    { 1,1,256,256,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   3 },


    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   1 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   2 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  32,   3 },

    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   1 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   2 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   3 },

    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   1 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   2 },
    { 1,1,64,64,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  8,   3 },

    // { 1,1,7,7,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   1 },
    // { 1,1,7,7,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   2 },
    // { 1,1,7,7,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   3 },

    // { 1,1,9,9,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   1 },
    // { 1,1,9,9,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   2 },
    // { 1,1,9,9,  1,0,0,  3, 3,  1, 1,   0,0,0,0,  16,   3 },

    // {3,128,240,240,  64,240,240,  3,3,  1,1,  1,1,1,1, 32,  1},
    // {3,128,240,240,  64,240,240,  3,3,  1,1,  1,1,1,1, 32,  2},
    // {3,128,240,240,  64,240,240,  3,3,  1,1,  1,1,1,1, 32,  3},
};

int main()
{
    // deviceQuery();
    // cudaSetDevice(0);
    cudaStream_t stream;
    cublasHandle_t cublasHandle;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cublasCreate(&cublasHandle));
    CUDA_CHECK(cublasSetStream(cublasHandle, stream));

    for (auto c : config) {
    		// Calculate output image dimensions
    		c.P = ((c.H + c.top + c.bottom - c.KH) / c.SH) + 1;
    		c.Q = ((c.W + c.left + c.right - c.KW) / c.SW) + 1;

    		// printf(" input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n\n", c.N, c.C, c.H, c.W, c.K, c.C, c.KH, c.KW, c.N, c.K, c.P, c.Q);

    		std::vector<float> data(c.N * c.C * c.H * c.W);
    		std::vector<float> weight(c.K * c.C * c.KH * c.KW);

    		inititalizedData(data);
    		inititalizedDataOne(weight);

    		// std::cout << "Data(Input)" << std::endl;
    		// valueCheck(data, c.N, c.C, c.H, c.W, 1);
    		// std::cout << "kernel" << std::endl;
    		// valueCheck(weight, c.K, c.C, c.KH, c.KW, 1);

    		float* d_weight;
    		CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
    		CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    		float* d_data;
    		CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));
    		CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

    		std::vector<float> data_output(c.C * c.KH * c.KW * c.N * c.P * c.Q);

    		float* d_data_output;
    		CUDA_CHECK(cudaMalloc(&d_data_output, data_output.size() * sizeof(float)));

    		std::vector<float> m_output(c.K * c.P * c.Q * c.N);
    		float* d_m_output;
    		CUDA_CHECK(cudaMalloc(&d_m_output, m_output.size() * sizeof(float)));

    		float* d_f_m_output;
    		CUDA_CHECK(cudaMalloc(&d_f_m_output, m_output.size() * sizeof(float)));

    		int ITER = 1000;
    		uint64_t total_time = 0;
    		for (int iIdx = 0; iIdx < ITER; iIdx++) {
      			uint64_t start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

      			conv2d_gemm(d_f_m_output, d_data_output, d_m_output, d_weight, d_data, c.N, c.K, c.P, c.Q, c.C, c.H, c.W, c.KH, c.KW, c.SH, c.SW, c.left, c.top, stream, cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, c.BS, c.V);

      			CUDA_CHECK(cudaStreamSynchronize(stream));

      			total_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start_time;
    		}

    		CUDA_CHECK(cudaMemcpy(m_output.data(), d_f_m_output, m_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    		// double checksum = 0;
    		// for (auto d : m_output)
    		// 	checksum += fabs((double)d);
    		// printf("   Gemm       avg_dur_time=%6.3f[msec] checksum=%.6f\n", total_time / 1000.f / ITER, checksum);
    		// valueCheck(m_output, c.N, c.K, c.P, c.Q, 1);
		
    		printf("Configuration:\n");

    		printf("GEMM - %d\n", c.V);
    		printf("Kernel size - %dx%d\n", c.KH, c.KW);
    		printf("Image size - %dx%d\n", c.H, c.W);
    		printf("Block size - %dx%d\n", c.BS, c.BS);
    		printf("\n");
    		printf("Average time - %6.3f msec\n\n", total_time / 1000.f / ITER);

    		cudaFree(d_data);
    		cudaFree(d_weight);
    		cudaFree(d_data_output);
    		cudaFree(d_m_output);
    		cudaFree(d_f_m_output);
    }

    return 0;
}