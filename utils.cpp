#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

void deviceQuery()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("Device %d : \"%s\"\n", dev, deviceProp.name);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("  Multiprocessors (MP) :                         %d\n", deviceProp.multiProcessorCount);
		printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	}
	printf("\n");
}

void valueCheck(std::vector<float>& valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int n_idx = 0; n_idx < input_n; n_idx++)
	{
		int temp2 = n_idx * temp1;
		for (int c_idx = 0; c_idx < input_c; c_idx++)
		{
			int temp3 = c_idx * input_w * input_h + temp2;
			for (int h_idx = 0; h_idx < input_h; h_idx++)
			{
				int temp4 = h_idx * input_w + temp3;
				std::cout << "  ";
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;

					//cout.setf(ios::fixed);
					//cout.precision(6);
					std::cout << std::setw(8) << valueCheckInput[g_idx] << " ";
				}std::cout << std::endl;
			}std::cout << std::endl << std::endl;
		}std::cout << std::endl;
	}std::cout << std::endl;
}

void inititalizedData(std::vector<float>& container)
{
	int count = 1;
	for (std::vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
		count++;
	}
}

void inititalizedDataOne(std::vector<float>& container)
{
	int count = 1;
	for (std::vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
	}
}
