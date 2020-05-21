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

#ifndef _KERNEL_FUNCTIONS_H_
#define _KERNEL_FUNCTIONS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cpu_statistical_test_header.h"
#include "cuda_profiler_api.h"
#include <curand_kernel.h>
#include <stdio.h>

/* data structure for the GPU-based parallel permutation testing */
typedef struct _GPU_TEST_COMP {
	uint8_t *data;								/* the original data(input) */
	uint8_t *dataN;								/* N shuffled data, where N is $USER_INPUT_GPU_COMP->nNum_iteration_in_parallel */
	uint8_t *b_dataN;							/* N shuffled data after conversion II, where N is $USER_INPUT_GPU_COMP->nNum_iteration_in_parallel */
	curandState *curand;						/* the seeds used by curand() function */
	double *original_test_statistics;			/* the results of 18 Statisitcal tests on the original data(input) */
	uint32_t *count1;							/* the counter #(the shuffled test statistics > the original test statistics) */
	uint32_t *count2;							/* the counter #(the shuffled test statistics = the original test statistics) */
	uint32_t *count3;							/* the counter #(the shuffled test statistics < the original test statistics) */

	uint32_t num_iteration;						/* the number of iterations processing in parallel on the GPU */
	uint32_t num_block;							/* the number of CUDA blcoks */
	uint32_t num_statistical_test_block;		/* the number of CUDA blocks used in the kernel statistical test (= $num_block x 9) */
	uint32_t num_binary_statistical_test_block;	/* the number of CUDA blocks used in the kernel statistical test when input data is binary 
													(= $num_block x 7) */
	uint32_t num_thread;						/* the number of CUDA threads */
	uint32_t num_invoke_kernel;					/* the number of invoking the kernel (= ceil(10,000 / $num_iteration)) */

	cudaEvent_t cuda_time_start;				/* the parameter of CUDA timer */
	cudaEvent_t cuda_time_end;					/* the parameter of CUDA timer */

	uint32_t len;								/* the number of samples in the original data */
	uint32_t blen;								/* the number of samples after conversion I/II (= $len/8) */

}GPU_TEST_COMP;

/*******
 * Kernel funtions and statistical test functions for device
*******/
/**
 * @brief The kernel function: Initialize the seeds used by curand() function.
 * @param curandState $curand Seeds used by curand() function
 * @param uint32_t $seed Seed to initialize the seeds $curand
 * @return void
 */
__global__ void setup_curand_kernel(curandState *curand, const uint32_t seed);

/**
 * @brief The kernel function: Generate {$num_iteration} shuffled data by permuting the original data {$num_iteration} times in parallel.
 * @param uint8_t $dataN {$num_iteration} shuffled data
 * @param uint8_t $data Original data(input)
 * @param curandState $curand Seeds used by curand() function
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @return void
 */
__global__ void shuffling_kernel(uint8_t *dataN, const uint8_t *data, curandState *dev_curand, const uint32_t len, const uint32_t num_iteration);

/**
 * @brief The kernel function: Perform 18 statistical tests on each of {$num_iteration} shuffled data, 
 *							   and compares the shuffled and original test statistics in parallel.
 * @param uint32_t $count1 Counter #(the shuffled test statistics > the original test statistics)
 * @param uint32_t $count2 Counter #(the shuffled test statistics = the original test statistics)
 * @param uint32_t $count3 Counter #(the shuffled test statistics < the original test statistics)
 * @param uint8_t $dataN {$num_iteration} shuffled data
 * @param double $mean Mean value of the original data(input)
 * @param double $median Median value of the original data(input)
 * @param double $original_statistics Results of 18 Statisitcal tests on the original data(input)
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $num_block Number of CUDA blocks
 * @param uint32_t $len Number of samples in the original data
 * @return void
 */
__global__ void statistical_tests_kernel(uint32_t *count1, uint32_t *count2, uint32_t *count3,
	const uint8_t *dataN, const double mean, const double median, const double *original_statistics, const uint32_t num_iteration,
	const uint32_t num_block, const uint32_t len);

/**
 * @brief Statistical test for device: Perform excursion test.
 * @param double $out_max Result of excursion test
 * @param double $mean Mean value of the original data(input)
 * @param uint8_t $data One shuffled data processed by one CUDA thread
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @return void
 */
__device__ void dev_excursion_test(double *out_max, const double mean, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid);

/**
 * @brief Statistical test for device: Perform directional runs and number of inc/dec 
 *	= number of directional runs + length of directional runs + numbers of increases and decreases.
 * @param double $out_num Result of number of directional runs
 * @param double $out_len Result of length of directional runs
 * @param double $out_max Result of numbers of increases and decreases
 * @param uint8_t $data One shuffled data processed by one CUDA thread
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @return void
 */
__device__ void dev_directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data,
	const uint32_t len, const uint32_t num_iteration, uint32_t tid);

/**
 * @brief Statistical test for device: Perform runs based on median 
 *	= number of runs based on median + length of runs based on median.
 * @param double $out_num Result of number of runs based on median
 * @param double $out_len Result of length of length of runs based on median
 * @param double $median Median value of the original data(input)
 * @param uint8_t $data One shuffled data processed by one CUDA thread
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @return void
 */
__device__ void dev_runs_based_on_median(double *out_num, double *out_len, const double median, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid);

/**
 * @brief Statistical test for device: Perform collision test statistic
 *	= average collision test statistic + maximum collision test statistic.
 * @param double $out_average Result of average collision test statistic
 * @param double $out_max Result of maximum collision test statistic
 * @param uint8_t $data One shuffled data processed by one CUDA thread
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @return void
 */
__device__ void dev_collision_test_statistic(double *out_average, double *out_max, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid);

/**
 * @brief Statistical test for device: Perform periodicity/covariance test 
 *	= periodicity test + covariance test.
 * @param double $out_num Result of periodicity test
 * @param double $out_strength Result of covariance test
 * @param uint8_t $data One shuffled data processed by one CUDA thread
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @param uint32_t $lag Lag parameter
 * @return void
 */
__device__ void dev_periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid, uint32_t lag);


/*******
 * Kernel funtions and statistical test functions for device when input data is binary
*******/
/**
 * @brief The kernel function: Use when the input data is binary.
 *  - Generate {$num_iteration} shuffled data by permuting the original data {$num_iteration} times in parallel.
 *  - Perform two statistical tests(dev_excursion_test and dev_directional_runs_and_number_of_inc_dec) on the shuffled data in parallel.
 *  - Perform the conversion II in parallel.
 * @param uint8_t $dataN {$num_iteration} shuffled data
 * @param uint8_t $b_dataN {$num_iteration} shuffled data after conversion II
 * @param uint32_t $count1 Counter #(the shuffled test statistics > the original test statistics)
 * @param uint32_t $count2 Counter #(the shuffled test statistics = the original test statistics)
 * @param uint32_t $count3 Counter #(the shuffled test statistics < the original test statistics)
 * @param curandState $curand Seeds used by curand() function
 * @param uint8_t $data Original data(input)
 * @param double $mean Mean value of the original data(input)
 * @param double $median Median value of the original data(input)
 * @param double $original_statistics Results of 18 Statisitcal tests on the original data(input)
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $len Number of samples in the original data
 * @param uint32_t $b_len Number of samples after converion II
 * @return void
 */
__global__ void binary_shuffling_kernel(uint8_t *dataN, uint8_t *b_dataN, uint32_t *count1, uint32_t *count2, uint32_t *count3,
	curandState *dev_curand, const uint8_t *data, const double mean, const double median, const double *original_statistics,
	const uint32_t num_iteration, const uint32_t len, const uint32_t blen);

/**
 * @brief The kernel function: Use when the input data is binary.
 *  - Perform 15 statistical tests on each of {$num_iteration} shuffled data, and compares the shuffled and original test statistics in parallel.
 * @param uint32_t $count1 Counter #(the shuffled test statistics > the original test statistics)
 * @param uint32_t $count2 Counter #(the shuffled test statistics = the original test statistics)
 * @param uint32_t $count3 Counter #(the shuffled test statistics < the original test statistics)
 * @param uint8_t $b_dataN {$num_iteration} shuffled data after conversion II
 * @param double $mean Mean value of the original data(input)
 * @param double $median Median value of the original data(input)
 * @param double $original_statistics Results of 18 Statisitcal tests on the original data(input)
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $num_block Number of CUDA blocks
 * @param uint32_t $b_len Number of samples after converion II
 * @return void
 */
__global__ void binary_statistical_tests_kernel(uint32_t *count1, uint32_t *count2, uint32_t *count3,
	const uint8_t *b_dataN, const double mean, const double median, const double *original_statistics, const uint32_t num_iteration,
	const uint32_t num_block, const uint32_t blen);

/**
 * @brief Statistical test for device: Count the number of 1s in each data of conversion II.
 * @param uint8_t $data Output data of conversion II
 * @return uint8_t $data Output data of converion I
 */
__device__ uint8_t dev_hammingweight(uint8_t data);

/**
 * @brief Statistical test for device: When input data is binary, perform directional runs and number of inc/dec
 *	= number of directional runs + length of directional runs + numbers of increases and decreases.
 * @param double $out_num Result of number of directional runs
 * @param double $out_len Result of length of directional runs
 * @param double $out_max Result of numbers of increases and decreases
 * @param uint8_t $data One shuffled data after conversion II, processed by one CUDA thread
 * @param uint32_t $len Number of samples after conversion II
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @return void
 */
__device__ void dev_binary_directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data,
	const uint32_t len, const uint32_t num_iteration, uint32_t tid);

/**
 * @brief Statistical test for device: When input data is binary, perform periodicity/covariance test
 *	= periodicity test + covariance test.
 * @param double $out_num Result of periodicity test
 * @param double $out_strength Result of covariance test
 * @param uint8_t $data One shuffled data after conversion II, processed by one CUDA thread
 * @param uint32_t $len Number of samples after conversion II
 * @param uint32_t $num_iteration Number of iterations processing in parallel on the GPU
 * @param uint32_t $tid Index of CUDA thread
 * @param uint32_t $lag Lag parameter
 * @return void
 */
__device__ void dev_binary_periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid, const uint32_t lag);


/*******
 * Kernel funtions and statistical test functions for device 
*******/
__global__ void setup_curand_kernel(curandState *curand, const uint32_t seed)
{
	uint64_t id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &curand[id]);
}

__global__ void shuffling_kernel(uint8_t *dataN, const uint8_t *data, curandState *dev_curand, const uint32_t len, const uint32_t num_iteration)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t i = 0, j = len - 1, random = 0;
	uint8_t tmp = 0;

	for (i = 0; i < len; i++)
		dataN[i * num_iteration + tid] = data[i];

	while (j > 0) {
		random = curand(&dev_curand[tid]) % j;
		tmp = dataN[random * num_iteration + tid];
		dataN[random * num_iteration + tid] = dataN[j * num_iteration + tid];
		dataN[j * num_iteration + tid] = tmp;
		j--;
	}
}

__global__ void statistical_tests_kernel(uint32_t *count1, uint32_t *count2, uint32_t *count3,
	const uint8_t *dataN, const double mean, const double median, const double *original_statistics, const uint32_t num_iteration,
	const uint32_t num_block, const uint32_t len)
{
	double result1 = 0, result2 = 0, result3 = 0;
	uint32_t tid = threadIdx.x + (blockIdx.x % num_block)*blockDim.x;

	if ((blockIdx.x / num_block) == 0) {
		dev_collision_test_statistic(&result1, &result2, dataN, len, num_iteration, tid);
		if (result1 > original_statistics[6])			atomicAdd(&count1[6], 1);
		else if (result1 == original_statistics[6])		atomicAdd(&count2[6], 1);
		else											atomicAdd(&count3[6], 1);
		if (result2 > original_statistics[7])			atomicAdd(&count1[7], 1);
		else if (result2 == original_statistics[7])		atomicAdd(&count2[7], 1);
		else											atomicAdd(&count3[7], 1);
	}
	else if ((blockIdx.x / num_block) == 1) {
		dev_excursion_test(&result1, mean, dataN, len, num_iteration, tid);
		if ((float)result1 > (float)original_statistics[0])			atomicAdd(&count1[0], 1);
		else if ((float)result1 == (float)original_statistics[0])	atomicAdd(&count2[0], 1);
		else														atomicAdd(&count3[0], 1);
	}
	else if ((blockIdx.x / num_block) == 2) {
		dev_directional_runs_and_number_of_inc_dec(&result1, &result2, &result3, dataN, len, num_iteration, tid);
		if (result1 > original_statistics[1])			atomicAdd(&count1[1], 1);
		else if (result1 == original_statistics[1])		atomicAdd(&count2[1], 1);
		else											atomicAdd(&count3[1], 1);
		if (result2 > original_statistics[2])			atomicAdd(&count1[2], 1);
		else if (result2 == original_statistics[2])		atomicAdd(&count2[2], 1);
		else											atomicAdd(&count3[2], 1);
		if (result3 > original_statistics[3])			atomicAdd(&count1[3], 1);
		else if (result3 == original_statistics[3])		atomicAdd(&count2[3], 1);
		else											atomicAdd(&count3[3], 1);
	}
	else if ((blockIdx.x / num_block) == 3) {
		dev_runs_based_on_median(&result1, &result2, median, dataN, len, num_iteration, tid);
		if (result1 > original_statistics[4])			atomicAdd(&count1[4], 1);
		else if (result1 == original_statistics[4])		atomicAdd(&count2[4], 1);
		else											atomicAdd(&count3[4], 1);
		if (result2 > original_statistics[5])			atomicAdd(&count1[5], 1);
		else if (result2 == original_statistics[5])		atomicAdd(&count2[5], 1);
		else											atomicAdd(&count3[5], 1);
	}
	else if ((blockIdx.x / num_block) == 4) {
		dev_periodicity_covariance_test(&result1, &result2, dataN, len, num_iteration, tid, 1);
		if (result1 > original_statistics[8])			atomicAdd(&count1[8], 1);
		else if (result1 == original_statistics[8])		atomicAdd(&count2[8], 1);
		else											atomicAdd(&count3[8], 1);
		if (result2 > original_statistics[13])			atomicAdd(&count1[13], 1);
		else if (result2 == original_statistics[13])	atomicAdd(&count2[13], 1);
		else											atomicAdd(&count3[13], 1);
	}
	else if ((blockIdx.x / num_block) == 5) {
		dev_periodicity_covariance_test(&result1, &result2, dataN, len, num_iteration, tid, 2);
		if (result1 > original_statistics[9])			atomicAdd(&count1[9], 1);
		else if (result1 == original_statistics[9])		atomicAdd(&count2[9], 1);
		else											atomicAdd(&count3[9], 1);
		if (result2 > original_statistics[14])			atomicAdd(&count1[14], 1);
		else if (result2 == original_statistics[14])	atomicAdd(&count2[14], 1);
		else											atomicAdd(&count3[14], 1);
	}
	else if ((blockIdx.x / num_block) == 6) {
		dev_periodicity_covariance_test(&result1, &result2, dataN, len, num_iteration, tid, 8);
		if (result1 > original_statistics[10])			atomicAdd(&count1[10], 1);
		else if (result1 == original_statistics[10])	atomicAdd(&count2[10], 1);
		else											atomicAdd(&count3[10], 1);
		if (result2 > original_statistics[15])			atomicAdd(&count1[15], 1);
		else if (result2 == original_statistics[15])	atomicAdd(&count2[15], 1);
		else											atomicAdd(&count3[15], 1);
	}
	else if ((blockIdx.x / num_block) == 7) {
		dev_periodicity_covariance_test(&result1, &result2, dataN, len, num_iteration, tid, 16);
		if (result1 > original_statistics[11])			atomicAdd(&count1[11], 1);
		else if (result1 == original_statistics[11])	atomicAdd(&count2[11], 1);
		else											atomicAdd(&count3[11], 1);
		if (result2 > original_statistics[16])			atomicAdd(&count1[16], 1);
		else if (result2 == original_statistics[16])	atomicAdd(&count2[16], 1);
		else											atomicAdd(&count3[16], 1);
	}
	else if ((blockIdx.x / num_block) == 8) {
		dev_periodicity_covariance_test(&result1, &result2, dataN, len, num_iteration, tid, 32);
		if (result1 > original_statistics[12])			atomicAdd(&count1[12], 1);
		else if (result1 == original_statistics[12])	atomicAdd(&count2[12], 1);
		else											atomicAdd(&count3[12], 1);
		if (result2 > original_statistics[17])			atomicAdd(&count1[17], 1);
		else if (result2 == original_statistics[17])	atomicAdd(&count2[17], 1);
		else											atomicAdd(&count3[17], 1);
	}
}

__device__ void dev_excursion_test(double *out_max, const double mean, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid)
{
	double max = 0;
	double temp = 0;
	double running_sum = 0;
	uint32_t i = 0;
	for (i = 0; i < len; i++) {
		running_sum += data[i*num_iteration + tid];
		temp = fabs(running_sum - ((i + 1)*mean));
		if (max < temp)
			max = temp;
	}
	*out_max = max;
}

__device__ void dev_directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data,
	const uint32_t len, const uint32_t num_iteration, uint32_t tid)
{
	uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0, pos = 0;
	bool bflag1 = 0, bflag2 = 0;
	uint32_t i = 0;
	if (data[tid] <= data[num_iteration + tid])
		bflag1 = 1;

	for (i = 1; i < len - 1; i++) {
		pos += bflag1;
		bflag2 = 0;

		if (data[i*num_iteration + tid] <= data[(i + 1)*num_iteration + tid])
			bflag2 = 1;
		if (bflag1 == bflag2)
			len_runs++;
		else {
			num_runs++;
			if (len_runs > max_len_runs)
				max_len_runs = len_runs;
			len_runs = 1;
		}
		bflag1 = bflag2;
	}
	pos += bflag1;
	*out_num = (double)num_runs;
	*out_len = (double)max_len_runs;
	*out_max = (double)max(pos, len - pos);
}

__device__ void dev_runs_based_on_median(double *out_num, double *out_len, const double median, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid)
{
	uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0;
	bool bflag1 = 0, bflag2 = 0;
	uint32_t i = 0;

	if (data[tid] >= median)
		bflag1 = 1;

	for (i = 1; i < len; i++) {
		bflag2 = 0;

		if (data[i*num_iteration + tid] >= median)
			bflag2 = 1;
		if (bflag1 == bflag2)
			len_runs++;
		else {
			num_runs++;
			if (len_runs > max_len_runs)
				max_len_runs = len_runs;
			len_runs = 1;
		}
		bflag1 = bflag2;
	}

	*out_num = (double)num_runs;
	*out_len = (double)max_len_runs;
}

__device__ void dev_collision_test_statistic(double *out_average, double *out_max, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid)
{
	uint32_t sum = 0, start = 0, collision = 1, index = 0, max_collision = 0;
	uint32_t i = 0, j = 0;
	double average_collision = 0;
	uint8_t temp1 = 0, temp2 = 0;

	while (collision) {
		start = i;
		collision = 0;
		for (i = start + 1; ((i < len) && (collision == 0)); i++) {
			temp1 = data[i*num_iteration + tid];
			for (j = start; j < i; j++) {
				temp2 = data[j*num_iteration + tid];
				if (temp1 == temp2) {
					collision = i - start + 1;
					break;
				}
			}
			if (collision) {
				sum += collision;
				index++;
				if (max_collision < collision)
					max_collision = collision;
			}
		}
	}
	average_collision = sum / (double)index;
	*out_average = average_collision - 1;
	*out_max = (double)max_collision - 1;
}

__device__ void dev_periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid, const uint32_t lag)
{
	double temp1 = 0, temp2 = 0;
	uint32_t i = 0;
	for (i = 0; i < len - lag; i++) {
		if (data[i*num_iteration + tid] == data[(i + lag)*num_iteration + tid])
			temp1++;
		temp2 += (data[i*num_iteration + tid] * data[(i + lag)*num_iteration + tid]);
	}
	*out_num = temp1;
	*out_strength = temp2;
}


/*******
 * Kernel funtions and statistical test functions for device when input data is binary
*******/
__global__ void binary_shuffling_kernel(uint8_t *dataN, uint8_t *b_dataN, uint32_t *count1, uint32_t *count2, uint32_t *count3,
	curandState *dev_curand, const uint8_t *data, const double mean, const double median, const double *original_statistics,
	const uint32_t num_iteration, const uint32_t len, const uint32_t blen)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t i = 0, j = len - 1, random = 0;
	uint8_t tmp = 0;

	for (i = 0; i < len; i++)
		dataN[i * num_iteration + tid] = data[i];

	while (j > 0) {
		random = curand(&dev_curand[tid]) % j;
		tmp = dataN[random * num_iteration + tid];
		dataN[random * num_iteration + tid] = dataN[j * num_iteration + tid];
		dataN[j * num_iteration + tid] = tmp;
		j--;
	}

	double result1 = 0, result2 = 0;
	dev_excursion_test(&result1, mean, dataN, len, num_iteration, tid);
	if ((float)result1 > (float)original_statistics[0])			atomicAdd(&count1[0], 1);
	else if ((float)result1 == (float)original_statistics[0])	atomicAdd(&count2[0], 1);
	else														atomicAdd(&count3[0], 1);

	dev_runs_based_on_median(&result1, &result2, median, dataN, len, num_iteration, tid);
	if (result1 > original_statistics[4])						atomicAdd(&count1[4], 1);
	else if (result1 == original_statistics[4])					atomicAdd(&count2[4], 1);
	else														atomicAdd(&count3[4], 1);
	if (result2 > original_statistics[5])						atomicAdd(&count1[5], 1);
	else if (result2 == original_statistics[5])					atomicAdd(&count2[5], 1);
	else														atomicAdd(&count3[5], 1);

	for (i = 0; i < blen; i++) {
		tmp = (dataN[8 * i * num_iteration + tid] & 0x1) << 7;
		tmp ^= (dataN[(8 * i + 1) * num_iteration + tid] & 0x1) << 6;
		tmp ^= (dataN[(8 * i + 2) * num_iteration + tid] & 0x1) << 5;
		tmp ^= (dataN[(8 * i + 3) * num_iteration + tid] & 0x1) << 4;
		tmp ^= (dataN[(8 * i + 4) * num_iteration + tid] & 0x1) << 3;
		tmp ^= (dataN[(8 * i + 5) * num_iteration + tid] & 0x1) << 2;
		tmp ^= (dataN[(8 * i + 6) * num_iteration + tid] & 0x1) << 1;
		tmp ^= (dataN[(8 * i + 7) * num_iteration + tid] & 0x1);
		b_dataN[i * num_iteration + tid] = tmp;
	}
}

__global__ void binary_statistical_tests_kernel(uint32_t *count1, uint32_t *count2, uint32_t *count3,
	const uint8_t *b_dataN, const double mean, const double median, const double *original_statistics, const uint32_t num_iteration,
	const uint32_t num_block, const uint32_t blen)
{
	double result1 = 0, result2 = 0, result3 = 0;
	uint32_t tid = threadIdx.x + (blockIdx.x % num_block)*blockDim.x;

	if ((blockIdx.x / num_block) == 0) {
		dev_collision_test_statistic(&result1, &result2, b_dataN, blen, num_iteration, tid);
		if (result1 > original_statistics[6])			atomicAdd(&count1[6], 1);
		else if (result1 == original_statistics[6])		atomicAdd(&count2[6], 1);
		else											atomicAdd(&count3[6], 1);
		if (result2 > original_statistics[7])			atomicAdd(&count1[7], 1);
		else if (result2 == original_statistics[7])		atomicAdd(&count2[7], 1);
		else											atomicAdd(&count3[7], 1);
	}
	else if ((blockIdx.x / num_block) == 1) {
		dev_binary_directional_runs_and_number_of_inc_dec(&result1, &result2, &result3, b_dataN, blen, num_iteration, tid);
		if (result1 > original_statistics[1])			atomicAdd(&count1[1], 1);
		else if (result1 == original_statistics[1])		atomicAdd(&count2[1], 1);
		else											atomicAdd(&count3[1], 1);
		if (result2 > original_statistics[2])			atomicAdd(&count1[2], 1);
		else if (result2 == original_statistics[2])		atomicAdd(&count2[2], 1);
		else											atomicAdd(&count3[2], 1);
		if (result3 > original_statistics[3])			atomicAdd(&count1[3], 1);
		else if (result3 == original_statistics[3])		atomicAdd(&count2[3], 1);
		else											atomicAdd(&count3[3], 1);
	}
	else if ((blockIdx.x / num_block) == 2) {
		dev_binary_periodicity_covariance_test(&result1, &result2, b_dataN, blen, num_iteration, tid, 1);
		if (result1 > original_statistics[8])			atomicAdd(&count1[8], 1);
		else if (result1 == original_statistics[8])		atomicAdd(&count2[8], 1);
		else											atomicAdd(&count3[8], 1);
		if (result2 > original_statistics[13])			atomicAdd(&count1[13], 1);
		else if (result2 == original_statistics[13])	atomicAdd(&count2[13], 1);
		else											atomicAdd(&count3[13], 1);
	}
	else if ((blockIdx.x / num_block) == 3) {
		dev_binary_periodicity_covariance_test(&result1, &result2, b_dataN, blen, num_iteration, tid, 2);
		if (result1 > original_statistics[9])			atomicAdd(&count1[9], 1);
		else if (result1 == original_statistics[9])		atomicAdd(&count2[9], 1);
		else											atomicAdd(&count3[9], 1);
		if (result2 > original_statistics[14])			atomicAdd(&count1[14], 1);
		else if (result2 == original_statistics[14])	atomicAdd(&count2[14], 1);
		else											atomicAdd(&count3[14], 1);
	}
	else if ((blockIdx.x / num_block) == 4) {
		dev_binary_periodicity_covariance_test(&result1, &result2, b_dataN, blen, num_iteration, tid, 8);
		if (result1 > original_statistics[10])			atomicAdd(&count1[10], 1);
		else if (result1 == original_statistics[10])	atomicAdd(&count2[10], 1);
		else											atomicAdd(&count3[10], 1);
		if (result2 > original_statistics[15])			atomicAdd(&count1[15], 1);
		else if (result2 == original_statistics[15])	atomicAdd(&count2[15], 1);
		else											atomicAdd(&count3[15], 1);
	}
	else if ((blockIdx.x / num_block) == 5) {
		dev_binary_periodicity_covariance_test(&result1, &result2, b_dataN, blen, num_iteration, tid, 16);
		if (result1 > original_statistics[11])			atomicAdd(&count1[11], 1);
		else if (result1 == original_statistics[11])	atomicAdd(&count2[11], 1);
		else											atomicAdd(&count3[11], 1);
		if (result2 > original_statistics[16])			atomicAdd(&count1[16], 1);
		else if (result2 == original_statistics[16])	atomicAdd(&count2[16], 1);
		else											atomicAdd(&count3[16], 1);
	}
	else if ((blockIdx.x / num_block) == 6) {
		dev_binary_periodicity_covariance_test(&result1, &result2, b_dataN, blen, num_iteration, tid, 32);
		if (result1 > original_statistics[12])			atomicAdd(&count1[12], 1);
		else if (result1 == original_statistics[12])	atomicAdd(&count2[12], 1);
		else											atomicAdd(&count3[12], 1);
		if (result2 > original_statistics[17])			atomicAdd(&count1[17], 1);
		else if (result2 == original_statistics[17])	atomicAdd(&count2[17], 1);
		else											atomicAdd(&count3[17], 1);
	}
}

__device__ uint8_t dev_hammingweight(uint8_t data) {
	uint8_t tmp = 0;
	tmp = (data >> 7) & 0x1;
	tmp += (data >> 6) & 0x1;
	tmp += (data >> 5) & 0x1;
	tmp += (data >> 4) & 0x1;
	tmp += (data >> 3) & 0x1;
	tmp += (data >> 2) & 0x1;
	tmp += (data >> 1) & 0x1;
	tmp += data & 0x1;
	return tmp;
}

__device__ void dev_binary_directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data,
	const uint32_t len, const uint32_t num_iteration, uint32_t tid)
{
	uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0, pos = 0;
	bool bflag1 = 0, bflag2 = 0;
	uint32_t i = 0;
	if (dev_hammingweight(data[tid]) <= dev_hammingweight(data[num_iteration + tid]))
		bflag1 = 1;

	for (i = 1; i < len - 1; i++) {
		pos += bflag1;
		bflag2 = 0;

		if (dev_hammingweight(data[i*num_iteration + tid]) <= dev_hammingweight(data[(i + 1)*num_iteration + tid]))
			bflag2 = 1;
		if (bflag1 == bflag2)
			len_runs++;
		else {
			num_runs++;
			if (len_runs > max_len_runs)
				max_len_runs = len_runs;
			len_runs = 1;
		}
		bflag1 = bflag2;
	}
	pos += bflag1;
	*out_num = (double)num_runs;
	*out_len = (double)max_len_runs;
	*out_max = (double)max(pos, len - pos);
}

__device__ void dev_binary_periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t num_iteration, uint32_t tid, const uint32_t lag)
{
	double temp1 = 0, temp2 = 0;
	uint32_t i = 0;
	for (i = 0; i < len - lag; i++) {
		if (dev_hammingweight(data[i*num_iteration + tid]) == dev_hammingweight(data[(i + lag)*num_iteration + tid]))
			temp1++;
		temp2 += (dev_hammingweight(data[i*num_iteration + tid]) * dev_hammingweight(data[(i + lag)*num_iteration + tid]));
	}
	*out_num = temp1;
	*out_strength = temp2;
}


#endif // !_KERNEL_FUNCTIONS_H_
