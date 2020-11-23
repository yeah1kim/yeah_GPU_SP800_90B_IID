/*
 * GPU-based parallel implementation of the IID test of NIST SP 800-90B.
 *
 * Copyright(C) < 2020 > <Yewon Kim>
 *
 * This program is free software : you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < https://www.gnu.org/licenses/>.
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include <curand_kernel.h>
#include "header.h"
#include "kernel_functions.cuh"

 /**
  * @brief Perform 10,000 iterations in parallel on the GPU. (exclude the compression test)
  *  - That is, perform {$N} iterations in the GPU and repeat ceil(10,000 / $N) times.
  *  - In each iteration, the original data are shuffled, 18 statistical tests are performed on the shuffled data,
  *    and the results are compared with the original test statistics.
  * @param double $dGPU_runtime: Runtime of 10,000 iterations measured by CUDA timer
  * @param uint32_t $counts[]: The counters, that is original test statistics's rankings
  * @param double $results[]: The results of 19 statistical tests on the original data
  * @param double $mean: Mean value of the original data(input)
  * @param double $median: Median value of the original data(input)
  * @param uint8_t $data[]: The original data(input), which consists of (noise) samples
  * @param uint32_t $size: The size of sample in bits (1~8)
  * @param uint32_t $len: The number of samples in the original data
  * @param uint32_t $N: The number of iterations processing in parallel on the GPU
  * @param uint32_t $num_block: The number of CUDA blocks
  * @param uint32_t $num_thread: The number of CUDA threads per block
  * @return bool $iid_check_result
  */
bool gpu_permutation_testing(double *dgpu_runtime, uint32_t *counts, double *results, double mean, double median,
	uint8_t *data, uint32_t size, uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread)
{
	int32_t i;
	uint8_t num_runtest = 0;
	uint32_t loop = 10000 / N;
	if ((10000 % N) != 0)	loop++;
	uint32_t blen;
	if (size == 1) {
		blen = len / 8;
		if ((len % 8) != 0)	blen++;
	}
	size_t Nlen = N * len;
	size_t Nblen = N * blen;

	cudaError_t cudaStatus;
	uint8_t *dev_data;
	uint8_t *dev_Ndata, *dev_bNdata;
	double *dev_results;
	curandState *dev_curand;
	uint32_t *dev_cnt;
	cudaEvent_t cuda_time_start, cuda_time_end;
	cudaEventCreate(&cuda_time_start);
	cudaEventCreate(&cuda_time_end);

	/* choose which GPU to run on. */
	CUDA_ERRORCHK((cudaSetDevice(0) != cudaSuccess));

	/* allocate memory on the GPU. */
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_data, len * sizeof(uint8_t)) != cudaSuccess));
	CUDA_CALLOC_ERRORCHK((cudaMalloc((void**)&dev_Ndata, Nlen) != cudaSuccess));
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

	/* generate {$N} shuffled data by permuting the original data {$N} times in parallel.
	 * perform 18 statistical tests on each of {$N} shuffled data and compares the shuffled and original test statistics in parallel.
	 */
	for (i = 0; i < loop; i++) {
		if (size == 1) {
			binary_shuffling_kernel << < num_block, num_thread >> > (dev_Ndata, dev_bNdata, dev_data, dev_curand, len, blen, N);
			binary_statistical_tests_kernel << < num_block * 4, num_thread >> > (dev_cnt, dev_results, mean, median, dev_Ndata, dev_bNdata, size, len, blen, N, num_block);

			/* copy data from the GPU to the CPU. */
			CUDA_MEMCPY_ERRORCHK((cudaMemcpy(counts, dev_cnt, 54 * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess));
			num_runtest = 0;
			for (int t = 0; t < 18; t++) {
				if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
					num_runtest++;
			}
			if (num_runtest == 18)
				break;
		}
		else {
			shuffling_kernel << < num_block, num_thread >> > (dev_Ndata, dev_data, dev_curand, len, N);
			statistical_tests_kernel << < num_block * 2, num_thread >> > (dev_cnt, dev_results, mean, median, dev_Ndata, size, len, N, num_block);

			/* copy data from the GPU to the CPU. */
			CUDA_MEMCPY_ERRORCHK((cudaMemcpy(counts, dev_cnt, 54 * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess));
			num_runtest = 0;
			for (int t = 0; t < 18; t++) {
				if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
					num_runtest++;
			}
			if (num_runtest == 18)
				break;
		}
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
	*dgpu_runtime = (double)cuda_time;

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

	if (num_runtest == 18) // IID
		return true;
	else // Non-IID
		return false;
}