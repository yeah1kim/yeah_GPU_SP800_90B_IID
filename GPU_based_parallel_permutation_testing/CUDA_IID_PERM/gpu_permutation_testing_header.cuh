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

#ifndef _GPU_PERMUTATION_TESTING_H_
#define _GPU_PERMUTATION_TESTING_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cpu_statistical_test_header.h"
#include "kernel_functions_h.cuh"
#include "cuda_profiler_api.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

/* data structure that the value of the parameter is entered by the user for the GPU-based parallel permutation testing */
typedef struct _USER_INPUT_GPU_COMP {
	/* nNum_iteration_in_parallel = nNum_cuda_block x nNum_cuda_thread */
	uint32_t nNum_iteration_in_parallel;	/* the number of iterations processing in parallel on the GPU */
	uint32_t nNum_cuda_block;				/* the number of CUDA blocks using in the CUDA kernel */
	uint32_t nNum_cuda_thread;				/* the number of CUDA threads using in the CUDA kernel */

	bool verbose;							/* verbosity flag for more output */
}USER_INPUT_GPU_COMP;


/**
 * @brief Determine whether the noise source outputs are IID.
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @return uint32_t $permutation_testing_result
 */
uint32_t determine_IID_assumption(TEST_COMP *test_comp)
{
	uint32_t check_ranking = 0, permutation_testing_result = 0;
	for (uint32_t i = 0; i < NUM_TEST; i++) {
		if (((test_comp->nCount[0][i] + test_comp->nCount[1][i]) <= 5) || (test_comp->nCount[2][i] >= 9995))
			break;
		else
			check_ranking++;
	}
	if (check_ranking == NUM_TEST)
		permutation_testing_result = 1;

	return permutation_testing_result;
}

/**
 * @brief Initialize the data structure for the GPU-based parallel permutation testing.
 *  - Set the parameter such as the length of the original data, the number of CUDA blocks, and the number of CUDA threads.
 *  - Allocate memory on the GPU.
 *  - Copy data from the CPU to the GPU.
 * @param GPU_TEST_COMP $gpu Data structure for the GPU-based parallel permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param USER_INPUT_GPU_COMP $user_gpu_comp Data structure that the value of the parameter is entered by the user for the GPU-based parallel permutation testing
 * @return uint32_t $cuda_malloc_memcpy_error
 */
uint32_t gpu_test_components_initialization(GPU_TEST_COMP *gpu_comp, const DATA *data_comp,
	const TEST_COMP *test_comp, const USER_INPUT_GPU_COMP *user_gpu_comp)
{
	/* set the parameter */
	gpu_comp->num_iteration = user_gpu_comp->nNum_iteration_in_parallel;
	gpu_comp->num_block = user_gpu_comp->nNum_cuda_block;
	gpu_comp->num_statistical_test_block = gpu_comp->num_block * 9;
	gpu_comp->num_binary_statistical_test_block = gpu_comp->num_block * 7;
	gpu_comp->num_thread = user_gpu_comp->nNum_cuda_thread;
	gpu_comp->num_invoke_kernel = 10000 / gpu_comp->num_iteration;
	if (10000 % gpu_comp->num_iteration)
		gpu_comp->num_invoke_kernel++;

	gpu_comp->len = data_comp->nLen;
	gpu_comp->blen = data_comp->nBlen;

	/* create the parameter of CUDA timer. */
	cudaEventCreate(&gpu_comp->cuda_time_start);
	cudaEventCreate(&gpu_comp->cuda_time_end);

	/* allocate memory on the GPU. */
	if (cudaMalloc((void**)&gpu_comp->data, gpu_comp->len * sizeof(uint8_t)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_data)!\n");
		return 1;
	}
	if (cudaMalloc((void**)&gpu_comp->dataN, gpu_comp->len * gpu_comp->num_iteration * sizeof(uint8_t)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_Ndata)!\n");
		return 1;
	}
	if (data_comp->nSample_size == 1) {
		if (cudaMalloc((void**)&gpu_comp->b_dataN, gpu_comp->blen * gpu_comp->num_iteration * sizeof(uint8_t)) != cudaSuccess) {
			printf("cudaMalloc failed(dev_b_Ndata)!\n");
			return 1;
		}
	}
	if (cudaMalloc((void**)&gpu_comp->original_test_statistics, NUM_TEST * sizeof(double)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_original_test_statistics)!\n");
		return 1;
	}
	if (cudaMalloc((void**)&gpu_comp->count1, NUM_TEST * sizeof(uint32_t)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_count1)!\n");
		return 1;
	}
	if (cudaMalloc((void**)&gpu_comp->count2, NUM_TEST * sizeof(uint32_t)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_count2)!\n");
		return 1;
	}
	if (cudaMalloc((void**)&gpu_comp->count3, NUM_TEST * sizeof(uint32_t)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_count3)!\n");
		return 1;
	}
	if (cudaMalloc((void**)&gpu_comp->curand, gpu_comp->num_iteration * sizeof(curandState)) != cudaSuccess) {
		printf("cudaMalloc failed(dev_curand)!\n");
		return 1;
	}

	/* copy data from the CPU to the GPU. */
	if (cudaMemcpy(gpu_comp->data, data_comp->pData, gpu_comp->len * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy from cpu to gpu failed(dev_data)! \n");
		return 1;
	}
	if (cudaMemcpy(gpu_comp->original_test_statistics, test_comp->dOriginal_test_statistics, NUM_TEST * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy from cpu to gpu failed(dev_original_test_statistics)! \n");
		return 1;
	}
	if (cudaMemcpy(gpu_comp->count1, test_comp->nCount[0], NUM_TEST * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy from cpu to gpu failed(dev_count1)! \n");
		return 1;
	}
	if (cudaMemcpy(gpu_comp->count2, test_comp->nCount[1], NUM_TEST * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy from cpu to gpu failed(dev_count2)! \n");
		return 1;
	}
	if (cudaMemcpy(gpu_comp->count3, test_comp->nCount[2], NUM_TEST * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("cudaMemcpy from cpu to gpu failed(dev_count3)! \n");
		return 1;
	}

	return 0;
}

/**
 * @brief Clear the memory of the struct for the GPU-based parallel permutation testing.
 * @param GPU_TEST_COMP $gpu Data structure for the GPU-based parallel permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void gpu_test_components_clear(GPU_TEST_COMP *gpu_comp, const DATA *data_comp)
{
	cudaEventDestroy(gpu_comp->cuda_time_start);
	cudaEventDestroy(gpu_comp->cuda_time_end);

	cudaFree(&gpu_comp->data);
	cudaFree(&gpu_comp->dataN);
	if (data_comp->nSample_size == 1)
		cudaFree(&gpu_comp->b_dataN);
	cudaFree(&gpu_comp->original_test_statistics);
	cudaFree(&gpu_comp->count1);
	cudaFree(&gpu_comp->count2);
	cudaFree(&gpu_comp->count3);
	cudaFree(&gpu_comp->curand);
}

/**
 * @brief Copy the data from the GPU to the CPU.
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param GPU_TEST_COMP $gpu Data structure for the GPU-based parallel permutation testing
 * @return uint32_t $cuda_memcpy_error
 */
uint32_t copy_count_from_gpu_to_host(TEST_COMP *test_comp, GPU_TEST_COMP *gpu_comp)
{
	// Copies data from GPU to CPU.
	if (cudaMemcpy(test_comp->nCount[0], gpu_comp->count1, NUM_TEST * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("cudaMemcpy from gpu to host failed(count1)! \n");
		return 1;
	}
	if (cudaMemcpy(test_comp->nCount[1], gpu_comp->count2, NUM_TEST * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("cudaMemcpy from gpu to host failed(count1)! \n");
		return 1;
	}
	if (cudaMemcpy(test_comp->nCount[2], gpu_comp->count3, NUM_TEST * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("cudaMemcpy from gpu to host failed(count1)! \n");
		return 1;
	}

	return 0;
}

/**
 * @brief Check that the equation is satisfied for all statistical test, and if satisfied stop invoking the kernels.
 *  - equation: ((count1 + count2) > 5) && ((count2 + count3) > 5)
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param uint32_t $current_num_invoke_kernel The number of invoking the kernels so far
 * @param uint32_t $max_num_invoke_kernel The maximum number of invoking the kernels
 * @return uint32_t $itr The number of invoking the kernels
 */
uint32_t check_continue_invoke_kernel(TEST_COMP *test_comp, uint32_t current_num_invoke_kernel, uint32_t max_num_invoke_kernel)
{
	uint32_t check = 0;
	for (uint32_t i = 0; i < NUM_TEST; i++)
		if (((test_comp->nCount[0][i] + test_comp->nCount[1][i]) > 5) && ((test_comp->nCount[1][i] + test_comp->nCount[2][i]) > 5))
			check++;
	if (check == NUM_TEST)
		return max_num_invoke_kernel;
	else
		return current_num_invoke_kernel;
}

/**
 * @brief Perform 10,000 iterations in parallel on the GPU.
 *  - That is, perform {$nNum_iteration_in_parallel} iterations in the GPU and repeat ceil(10,000 / $nNum_iteration_in_parallel) times.
 *  - In each iteration, the original data are shuffled, the statistical tests are performed on the shuffled data,
 *    and the results are compared with the original test statistics.
 * @param double $dGPU_runtime Runtime of 10,000 iterations measured by CUDA timer
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @param USER_INPUT_GPU_COMP $user_gpu_comp Data structure that the value of the parameter is entered by the user for the GPU-based parallel permutation testing
 * @return cudaError_t $cudaStatus
 */
cudaError_t gpu_permutation_testing(double *dGPU_runtime, TEST_COMP *test_comp, const DATA *data_comp, const USER_INPUT_GPU_COMP *user_gpu_comp)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaError_t cudaStatus;
	GPU_TEST_COMP gpu = { 0x00, }; /* GPU_TEST_COMP Instantiation */

	/* choose which GPU to run on. */
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	/* initialize the struct for the GPU-based parallel permutation testing. */
	if (gpu_test_components_initialization(&gpu, data_comp, test_comp, user_gpu_comp))
		goto Error;

	/* start the CUDA timer. */
	cudaEventRecord(gpu.cuda_time_start, 0);

	/* initialize the seeds used by curand() function. */
	setup_curand_kernel << < gpu.num_block, gpu.num_thread >> > (gpu.curand, (uint32_t)time(NULL));
	cudaDeviceSynchronize();

	/* perform $num_iteration iterations in the GPU and repeat $num_invoke_kernel(= ceil(10,000/$num_iteration)) times.
	 * it isn't performed $num_invoke_kernel times completely if the equation is satisfied for all statistical test.
	 * equation: ((count1 + count2) > 5) && ((count2 + count3) > 5)
	 */
	for (uint32_t itr = 0; itr < gpu.num_invoke_kernel; itr++) {
		if (data_comp->nSample_size != 1) {
			/* generate {$num_iteration} shuffled data by permuting the original data {$num_iteration} times in parallel. */
			shuffling_kernel << < gpu.num_block, gpu.num_thread >> > (gpu.dataN, gpu.data, gpu.curand, gpu.len, gpu.num_iteration);
			cudaDeviceSynchronize();

			/* perform 18 statistical tests on each of {$num_iteration} shuffled data,
			 * and compares the shuffled and original test statistics in parallel.
			 */
			statistical_tests_kernel << < gpu.num_statistical_test_block, gpu.num_thread >> > (gpu.count1, gpu.count2, gpu.count3,
				gpu.dataN, test_comp->dMean, test_comp->dMedian, gpu.original_test_statistics, gpu.num_iteration, gpu.num_block, gpu.len);
			cudaDeviceSynchronize();
		}
		else {
			/* generate {$num_iteration} shuffled data by permuting the original data {$num_iteration} times in parallel.
			 * + perform two statistical tests(dev_excursion_test and dev_directional_runs_and_number_of_inc_dec) on the shuffled data in parallel.
			 * + perform the conversion II in parallel.
			 */
			binary_shuffling_kernel << < gpu.num_block, gpu.num_thread >> > (gpu.dataN, gpu.b_dataN, gpu.count1, gpu.count2, gpu.count3,
				gpu.curand, gpu.data, test_comp->dMean, test_comp->dMedian, gpu.original_test_statistics, gpu.num_iteration, gpu.len, gpu.blen);
			cudaDeviceSynchronize();

			/* perform 15 statistical tests on each of {$num_iteration} shuffled data,
			 * and compares the shuffled and original test statistics in parallel.
			 */
			binary_statistical_tests_kernel << < gpu.num_binary_statistical_test_block, gpu.num_thread >> > (gpu.count1, gpu.count2, gpu.count3,
				gpu.b_dataN, test_comp->dMean, test_comp->dMedian, gpu.original_test_statistics, gpu.num_iteration, gpu.num_block, gpu.blen);
			cudaDeviceSynchronize();
		}

		/* copy the data from the GPU to the CPU. */
		if (copy_count_from_gpu_to_host(test_comp, &gpu))
			goto Error;

		/* check that the equation is satisfied for all statistical test, and if satisfied stop invoking the kernels. */
		itr = check_continue_invoke_kernel(test_comp, itr, gpu.num_invoke_kernel);
	}

	/* stop the CUDA timer. */
	cudaEventRecord(gpu.cuda_time_end, 0);

	/* check for any errors launching the kernel. */
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shuffling_statisticaltests launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shuffling_statisticaltests!\n", cudaStatus);
		goto Error;
	}

	/* calculate the run-time of the permutation testing (measured by CUDA timer) */
	float cuda_time;
	cudaEventElapsedTime(&cuda_time, gpu.cuda_time_start, gpu.cuda_time_end);
	*dGPU_runtime = cuda_time;

	/* clear the memory of the struct for the GPU-based parallel permutation testing. */
	gpu_test_components_clear(&gpu, data_comp);

Error:
	gpu_test_components_clear(&gpu, data_comp);
	return cudaStatus;
}

/**
 * @brief Identify evidence against the null hypothesis that the noise source is IID.
 *  - The process follows the process of the algorithm in NIST SP 800-90B.
 * @param uint32_t $permutation_testing_result The result of the permutation testing
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return uint32_t $cuda_error
 */
uint32_t permutation_testing(uint32_t *permutation_testing_result, DATA *data_comp, USER_INPUT_GPU_COMP *user_gpu_comp)
{
	/* TEST_COMP Instantiation */
	TEST_COMP test_comp = { 0x00, };

	/* calculate mean and median for the original data(input). */
	calculate_statsistics(&test_comp, data_comp);
	if (user_gpu_comp->verbose) {
		printf(">---- Mean value of the original data(input): %f \n", test_comp.dMean);
		printf(">---- Median value of the original data(input): %f \n\n", test_comp.dMedian);
	}

	printf("Performing 18 Statisitcal tests on the original data. \n");

	/* perform 18 Statisitcal tests on the original data(input). */
	statistical_tests(&test_comp, data_comp);
	if (user_gpu_comp->verbose)
		print_original_test_statistics(test_comp.dOriginal_test_statistics);


	printf("Performing 10,000 iterations in parallel on the GPU. \n");
	double dGPU_runtime = 0; /* runtime of 10,000 iterations measured by CUDA timer */

	/* perform 10,000 iterations in parallel on the GPU. */
	cudaError_t cudaStatus = gpu_permutation_testing(&dGPU_runtime, &test_comp, data_comp, user_gpu_comp);
	if (user_gpu_comp->verbose)
		print_counters(&test_comp);

	printf("Run-time of the permutation testing processed in the GPU (measured by CUDA timer) : %.3f sec\n", dGPU_runtime / (double)CLOCKS_PER_SEC);

	/* determine whether the noise source outputs are IID. */
	*permutation_testing_result = determine_IID_assumption(&test_comp);

	/* cuda device reset. */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}


/*******
 * Initializing functions
*******/
void set_parameter_by_default(DATA *data_comp, USER_INPUT_GPU_COMP *user_gpu_comp)
{
	data_comp->nLen = 1000000;
	user_gpu_comp->nNum_iteration_in_parallel = 2048;
	user_gpu_comp->nNum_cuda_block = 8;
	user_gpu_comp->nNum_cuda_thread = 256;
	user_gpu_comp->verbose = false;
}

uint32_t input_by_user(DATA *data_comp, USER_INPUT_GPU_COMP *user_gpu_comp)
{
	FILE *fin;
	uint32_t user_len = 0;						/* the number of samples in the input data (= len) */
	uint32_t file_size = 1000000;						/* the size of the input file */
	uint32_t user_num_iteration_in_parallel;	/* the number of iterations processing in parallel on the GPU */
	uint32_t user_num_block;					/* the number of CUDA blocks using in the CUDA kernel */
	uint32_t user_num_thread;					/* the number of CUDA threads using in the CUDA kernel */
	uint32_t verbose_flag;						/* optional verbosity flag for more output */

	printf("<file_name>: Must be relative path to a binary file with 1 million entries (samples).\n");
	printf("	     ex) C:\\Users\\user\\Desktop\\test_data\\truerand_1bit.bin \n");
INPUT_FILE_NAME:
	scanf("%s", &data_comp->in_file_name);
	if ((fin = fopen(data_comp->in_file_name, "rb")) == NULL) {
		printf("(!!) File open fails. \n");
		goto INPUT_FILE_NAME;
	}

	printf("\n[the_number_of_samples]: Must be at least 1 million samples. If 0, set as the default(= 1,000,000). \n");
INPUT_LEN:
	scanf("%d", &user_len);
	if (user_len != 0) {
		if (user_len < 1000000) {
			printf("(!!) $the_number_of_samples must be at least 1,000,000. \n");
			goto INPUT_LEN;
		}
		fseek(fin, 0, SEEK_END);
		file_size = ftell(fin);
		if (file_size < user_len) {
			printf("(!!) The size of the file(%d-byte) is smaller than [the_number_of_samples]. \n", file_size);
			goto INPUT_LEN;
		}
		data_comp->nLen = user_len;
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	if (file_size < 1000000) {
		printf("(!!) The size of the file(%d-byte) is smaller than the size required for testing(= 1,000,000). \n", file_size);
		fclose(fin);
		return 0;
	}
	fclose(fin);

	printf("\n[bits_per_symbol]: Must be between 1-8, inclusive. \n");
INPUT_SAMPLE_SIZE:
	scanf("%d", &data_comp->nSample_size);
	if ((data_comp->nSample_size < 1) || (data_comp->nSample_size > 8)) {
		printf("(!!) [bits_per_symbol] must be between 1-8. \n");
		goto INPUT_SAMPLE_SIZE;
	}

	printf("\nEnter: [num_iteration_in_parallel] [num_cuda_block] [num_cuda_thread] \n");
	printf("   Must be [num_iteration_in_parallel] = [num_cuda_block] x [num_cuda_thread]. \n");
	printf("   If all 0, set as the defaults(2048, 8, 256), and at least 2GB of GPU global memory is used. \n");
	printf("     - [num_iteration_in_parallel]: The number of iterations in parallel. \n");
	printf("				    Must have ([num_iteration_in_parallel] x 1 million bytes) of GPU global memory. \n");
	printf("     - [num_cuda_block]: The number of CUDA blocks. \n");
	printf("     - [num_cuda_thread]: The number of CUDA blocks. \n");
GPU_PARAM:
	scanf("%d %d %d", &user_num_iteration_in_parallel, &user_num_block, &user_num_thread);
	if ((user_num_iteration_in_parallel != 0) || (user_num_block != 0) || (user_num_thread != 0)) {
		if (user_num_iteration_in_parallel != (user_num_block * user_num_thread)) {
			printf("(!!) Must be [num_iteration_in_parallel] = [num_cuda_block] x [num_cuda_thread]. \n");
			goto GPU_PARAM;
		}
		user_gpu_comp->nNum_iteration_in_parallel = user_num_iteration_in_parallel;
		user_gpu_comp->nNum_cuda_block = user_num_block;
		user_gpu_comp->nNum_cuda_thread = user_num_thread;
	}
	printf("\n[verbose] Optional verbosity flag(0/1) for more output. 0(false) is the default. \n");
	scanf("%d", &verbose_flag);
	if (verbose_flag)
		user_gpu_comp->verbose = true;
	printf("\n");

	data_comp->pData = (uint8_t*)calloc(data_comp->nLen, sizeof(uint8_t));
	if (data_comp->pData == NULL) {
		printf("calloc error \n");
		return 0;
	}
	if (data_comp->nSample_size == 1)
		data_comp->nBlen = data_comp->nLen / 8;

	return 0;
}

uint32_t read_data_from_file(DATA *data_comp)
{
	FILE *fin;
	uint8_t temp = 0;
	uint8_t mask = (1 << data_comp->nSample_size) - 1;
	uint32_t i = 0;

	if ((fin = fopen(data_comp->in_file_name, "rb")) == NULL) {
		printf("File open fails. \n");
		return 0;
	}
	for (uint32_t i = 0; i < data_comp->nLen; i++) {
		temp = 0;
		fread(&temp, sizeof(unsigned char), 1, fin);
		data_comp->pData[i] = (temp&mask);
	}
	fclose(fin);
}


#endif

