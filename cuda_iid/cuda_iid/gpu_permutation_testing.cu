
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <curand_kernel.h>
#include "header.h"
#include "kernel_functions.cuh"

int gpu_permutation_testing(double *dGPU_runtime, uint32_t *counts, double *results, double mean, double median,
	uint8_t *data, uint32_t size, uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread)
{
	uint32_t loop = 10000 / N;
	if ((10000 % N) != 0)	loop++;
	cudaError_t cudaStatus;
	uint8_t *dev_data, *dev_Ndata;
	double *dev_results;
	curandState *dev_curand;
	uint32_t *dev_cnt;
	cudaEvent_t cuda_time_start, cuda_time_end;
	cudaEventCreate(&cuda_time_start);
	cudaEventCreate(&cuda_time_end);

	uint32_t blen;
	uint8_t *dev_bNdata;
	if (size == 1) {
		blen = len / 8;
		if ((len % 8) != 0)	blen++;
	}

	/* choose which GPU to run on. */
	CUDA_ERRORCHK((cudaSetDevice(0) != cudaSuccess));

	/* allocate memory on the GPU. */
	size_t Nlen = N * len;
	size_t Nblen = N * blen;
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_data, len * sizeof(uint8_t)) != cudaSuccess));
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_Ndata, Nlen * sizeof(uint8_t)) != cudaSuccess));
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_results, 18 * sizeof(double)) != cudaSuccess));
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_curand, N * sizeof(curandState)) != cudaSuccess));
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_cnt, 54 * sizeof(uint32_t)) != cudaSuccess));
	if (size == 1)
		CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_bNdata, Nblen * sizeof(uint8_t)) != cudaSuccess));

	/* copy data from the CPU to the GPU. */
	CUDA_MEMCPY_ERRORCHK((cudaMemcpy(dev_data, data, len * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess));
	CUDA_MEMCPY_ERRORCHK((cudaMemcpy(dev_results, results, 18 * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess));
	CUDA_MEMCPY_ERRORCHK((cudaMemcpy(dev_cnt, counts, 54 * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess));

	/* start the CUDA timer. */
	cudaEventRecord(cuda_time_start, 0);

	/* initialize the seeds used by curand() function. */
	setup_curand_kernel << < num_block, num_thread >> > (dev_curand, (uint32_t)time(NULL));
	cudaDeviceSynchronize();

	for (int i = 0; i < loop; i++) {
		if (size == 1) {
			binary_shuffling_kernel << < num_block, num_thread >> > (dev_Ndata, dev_bNdata, dev_data, dev_curand, len, blen, N);
			cudaDeviceSynchronize();
			binary_b4_statistical_tests_kernel << < num_block * 4, num_thread >> > (dev_cnt, dev_results, mean, median, dev_Ndata, dev_bNdata, size, len, blen, N, num_block);
			cudaDeviceSynchronize();
		}
		else {
			shuffling_kernel << < num_block, num_thread >> > (dev_Ndata, dev_data, dev_curand, len, N);
			cudaDeviceSynchronize();
			b2_statistical_tests_kernel << < num_block * 2, num_thread >> > (dev_cnt, dev_results, mean, median, dev_Ndata, size, len, N, num_block);
			cudaDeviceSynchronize();
		}

		/* copy data from the GPU to the CPU. */
		CUDA_MEMCPY_ERRORCHK((cudaMemcpy(counts, dev_cnt, 54 * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess));
		uint8_t check = 0;
		for (int t = 0; t < 18; t++) {
			if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
				check++;
		}
		if (check == 18)
			break;
	}

	/* stop the CUDA timer. */
	cudaEventRecord(cuda_time_end, 0);
	cudaDeviceSynchronize();


	/* check for any errors launching the kernel. */
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "gpu_permutation_testing launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching gpu_permutation_testing!\n", cudaStatus);
		goto Error;
	}

	/* calculate the run-time of the permutation testing (measured by CUDA timer) */
	float cuda_time = 0;
	cudaEventElapsedTime(&cuda_time, cuda_time_start, cuda_time_end);
	*dGPU_runtime = (double)cuda_time;


	/* cuda device reset. */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	
Error:
	cudaFree(dev_data);
	cudaFree(dev_Ndata);
	cudaFree(dev_results);
	cudaFree(dev_curand);
	cudaFree(dev_cnt);
	if (size == 1) 
		cudaFree(dev_bNdata);
	cudaEventDestroy(cuda_time_start);
	cudaEventDestroy(cuda_time_end);
	
	return 0;
}