#ifndef _KERNEL_FUNCTIONS_H_
#define _KERNEL_FUNCTIONS_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "header.h"
#include "device_functions.cuh"

__global__ void setup_curand_kernel(curandState *curand, const uint32_t seed)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &curand[id]);
}

__global__ void shuffling_kernel(uint8_t *Ndata, const uint8_t *data, curandState *dev_curand, const uint32_t len, const uint32_t N)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t i = 0, j = len - 1, random = 0;
	uint8_t tmp = 0;

	for (i = 0; i < len; i++) {
		Ndata[i * N + tid] = data[i];
	}

	while (j > 0) {
		random = curand(&dev_curand[tid]) % j;
		tmp = Ndata[random * N + tid];
		Ndata[random * N + tid] = Ndata[j * N + tid];
		Ndata[j * N + tid] = tmp;
		j--;
	}
}

__global__ void b2_statistical_tests_kernel(uint32_t *counts, const double *results, const double mean, const double median,
	const uint8_t *Ndata, const uint32_t size, const uint32_t len, const uint32_t N, const uint32_t num_block)
{
	uint32_t tid = threadIdx.x + (blockIdx.x % num_block)*blockDim.x;

	if ((blockIdx.x / num_block) == 0) {
		double result1 = 0, result2 = 0;
		dev_test7_8(&result1, &result2, Ndata, size, len, N, tid);
		if (result1 > results[6])			atomicAdd(&counts[18], 1);
		else if (result1 == results[6])		atomicAdd(&counts[19], 1);
		else								atomicAdd(&counts[20], 1);
		if (result2 > results[7])			atomicAdd(&counts[21], 1);
		else if (result2 == results[7])		atomicAdd(&counts[22], 1);
		else								atomicAdd(&counts[23], 1);
	}
	else if ((blockIdx.x / num_block) == 1) {
		double result1 = 0, result2 = 0, result3 = 0, result4 = 0, result5 = 0;

		dev_test1(&result1, mean, Ndata, len, N, tid);
		if ((float)result1 > (float)results[0])			atomicAdd(&counts[0], 1);
		else if ((float)result1 == (float)results[0])	atomicAdd(&counts[1], 1);
		else											atomicAdd(&counts[2], 1);

		dev_test2_6(&result1, &result2, &result3, &result4, &result5, median, Ndata, len, N, tid);
		if (result1 > results[1])			atomicAdd(&counts[3], 1);
		else if (result1 == results[1])		atomicAdd(&counts[4], 1);
		else								atomicAdd(&counts[5], 1);
		if (result2 > results[2])			atomicAdd(&counts[6], 1);
		else if (result2 == results[2])		atomicAdd(&counts[7], 1);
		else								atomicAdd(&counts[8], 1);
		if (result3 > results[3])			atomicAdd(&counts[9], 1);
		else if (result3 == results[3])		atomicAdd(&counts[10], 1);
		else								atomicAdd(&counts[11], 1);
		if (result4 > results[4])			atomicAdd(&counts[12], 1);
		else if (result4 == results[4])		atomicAdd(&counts[13], 1);
		else								atomicAdd(&counts[14], 1);
		if (result5 > results[5])			atomicAdd(&counts[15], 1);
		else if (result5 == results[5])		atomicAdd(&counts[16], 1);
		else								atomicAdd(&counts[17], 1);

		dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 1);
		if (result1 > results[8])			atomicAdd(&counts[24], 1);
		else if (result1 == results[8])		atomicAdd(&counts[25], 1);
		else								atomicAdd(&counts[26], 1);
		if (result2 > results[13])			atomicAdd(&counts[39], 1);
		else if (result2 == results[13])	atomicAdd(&counts[40], 1);
		else								atomicAdd(&counts[41], 1);
		dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 2);
		if (result1 > results[9])			atomicAdd(&counts[27], 1);
		else if (result1 == results[9])		atomicAdd(&counts[28], 1);
		else								atomicAdd(&counts[29], 1);
		if (result2 > results[14])			atomicAdd(&counts[42], 1);
		else if (result2 == results[14])	atomicAdd(&counts[43], 1);
		else								atomicAdd(&counts[44], 1);
		dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 8);
		if (result1 > results[10])			atomicAdd(&counts[30], 1);
		else if (result1 == results[10])	atomicAdd(&counts[31], 1);
		else								atomicAdd(&counts[32], 1);
		if (result2 > results[15])			atomicAdd(&counts[45], 1);
		else if (result2 == results[15])	atomicAdd(&counts[46], 1);
		else								atomicAdd(&counts[47], 1);
		dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 16);
		if (result1 > results[11])			atomicAdd(&counts[33], 1);
		else if (result1 == results[11])	atomicAdd(&counts[34], 1);
		else								atomicAdd(&counts[35], 1);
		if (result2 > results[16])			atomicAdd(&counts[48], 1);
		else if (result2 == results[16])	atomicAdd(&counts[49], 1);
		else								atomicAdd(&counts[50], 1);
		dev_test9_and_14(&result1, &result2, Ndata, len, N, tid, 32);
		if (result1 > results[12])			atomicAdd(&counts[36], 1);
		else if (result1 == results[12])	atomicAdd(&counts[37], 1);
		else								atomicAdd(&counts[38], 1);
		if (result2 > results[17])			atomicAdd(&counts[51], 1);
		else if (result2 == results[17])	atomicAdd(&counts[52], 1);
		else								atomicAdd(&counts[53], 1);
	}
}

__global__ void binary_shuffling_kernel(uint8_t *Ndata, uint8_t *bNdata, const uint8_t *data, curandState *dev_curand, 
	const uint32_t len, const uint32_t blen, const uint32_t N)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t i = 0, j = len - 1, random = 0;
	uint8_t tmp = 0;

	for (i = 0; i < len; i++) {
		Ndata[i * N + tid] = data[i];
	}

	while (j > 0) {
		random = curand(&dev_curand[tid]) % j;
		tmp = Ndata[random * N + tid];
		Ndata[random * N + tid] = Ndata[j * N + tid];
		Ndata[j * N + tid] = tmp;
		j--;
	}

	for (i = 0; i < blen; i++) {
		tmp = (Ndata[8 * i * N + tid] & 0x1) << 7;
		tmp ^= (Ndata[(8 * i + 1) * N + tid] & 0x1) << 6;
		tmp ^= (Ndata[(8 * i + 2) * N + tid] & 0x1) << 5;
		tmp ^= (Ndata[(8 * i + 3) * N + tid] & 0x1) << 4;
		tmp ^= (Ndata[(8 * i + 4) * N + tid] & 0x1) << 3;
		tmp ^= (Ndata[(8 * i + 5) * N + tid] & 0x1) << 2;
		tmp ^= (Ndata[(8 * i + 6) * N + tid] & 0x1) << 1;
		tmp ^= (Ndata[(8 * i + 7) * N + tid] & 0x1);
		bNdata[i * N + tid] = tmp;
	}
}

__global__ void binary_b4_statistical_tests_kernel(uint32_t *counts, const double *results, const double mean, const double median,
	const uint8_t *Ndata, const uint8_t *bNdata, const uint32_t size, const uint32_t len, const uint32_t blen, const uint32_t N, const uint32_t num_block)
{
	double result1 = 0, result2 = 0, result3 = 0;
	uint32_t tid = threadIdx.x + (blockIdx.x % num_block)*blockDim.x;

	if ((blockIdx.x / num_block) == 0) {
		dev_test1(&result1, mean, Ndata, len, N, tid);
		if ((float)result1 > (float)results[0])			atomicAdd(&counts[0], 1);
		else if ((float)result1 == (float)results[0])	atomicAdd(&counts[1], 1);
		else											atomicAdd(&counts[2], 1);
	}
	else if ((blockIdx.x / num_block) == 1) {
		dev_test5_6(&result1, &result2, median, Ndata, len, N, tid);
		if (result1 > results[4])			atomicAdd(&counts[12], 1);
		else if (result1 == results[4])		atomicAdd(&counts[13], 1);
		else								atomicAdd(&counts[14], 1);
		if (result2 > results[5])			atomicAdd(&counts[15], 1);
		else if (result2 == results[5])		atomicAdd(&counts[16], 1);
		else								atomicAdd(&counts[17], 1);

		dev_binary_test2_4(&result1, &result2, &result3, bNdata, blen, N, tid);
		if (result1 > results[1])			atomicAdd(&counts[3], 1);
		else if (result1 == results[1])		atomicAdd(&counts[4], 1);
		else								atomicAdd(&counts[5], 1);
		if (result2 > results[2])			atomicAdd(&counts[6], 1);
		else if (result2 == results[2])		atomicAdd(&counts[7], 1);
		else								atomicAdd(&counts[8], 1);
		if (result3 > results[3])			atomicAdd(&counts[9], 1);
		else if (result3 == results[3])		atomicAdd(&counts[10], 1);
		else								atomicAdd(&counts[11], 1);
	}
	else if ((blockIdx.x / num_block) == 2) {
		dev_test7_8(&result1, &result2, bNdata, 8, blen, N, tid);
		if (result1 > results[6])			atomicAdd(&counts[18], 1);
		else if (result1 == results[6])		atomicAdd(&counts[19], 1);
		else								atomicAdd(&counts[20], 1);
		if (result2 > results[7])			atomicAdd(&counts[21], 1);
		else if (result2 == results[7])		atomicAdd(&counts[22], 1);
		else								atomicAdd(&counts[23], 1);
	}
	else if ((blockIdx.x / num_block) == 3) {
		dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 1);
		if (result1 > results[8])			atomicAdd(&counts[24], 1);
		else if (result1 == results[8])		atomicAdd(&counts[25], 1);
		else								atomicAdd(&counts[26], 1);
		if (result2 > results[13])			atomicAdd(&counts[39], 1);
		else if (result2 == results[13])	atomicAdd(&counts[40], 1);
		else								atomicAdd(&counts[41], 1);

		dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 2);
		if (result1 > results[9])			atomicAdd(&counts[27], 1);
		else if (result1 == results[9])		atomicAdd(&counts[28], 1);
		else								atomicAdd(&counts[29], 1);
		if (result2 > results[14])			atomicAdd(&counts[42], 1);
		else if (result2 == results[14])	atomicAdd(&counts[43], 1);
		else								atomicAdd(&counts[44], 1);

		dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 8);
		if (result1 > results[10])			atomicAdd(&counts[30], 1);
		else if (result1 == results[10])	atomicAdd(&counts[31], 1);
		else								atomicAdd(&counts[32], 1);
		if (result2 > results[15])			atomicAdd(&counts[45], 1);
		else if (result2 == results[15])	atomicAdd(&counts[46], 1);
		else								atomicAdd(&counts[47], 1);

		dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 16);
		if (result1 > results[11])			atomicAdd(&counts[33], 1);
		else if (result1 == results[11])	atomicAdd(&counts[34], 1);
		else								atomicAdd(&counts[35], 1);
		if (result2 > results[16])			atomicAdd(&counts[48], 1);
		else if (result2 == results[16])	atomicAdd(&counts[49], 1);
		else								atomicAdd(&counts[50], 1);

		dev_binary_test9_and_14(&result1, &result2, bNdata, blen, N, tid, 32);
		if (result1 > results[12])			atomicAdd(&counts[36], 1);
		else if (result1 == results[12])	atomicAdd(&counts[37], 1);
		else								atomicAdd(&counts[38], 1);
		if (result2 > results[17])			atomicAdd(&counts[51], 1);
		else if (result2 == results[17])	atomicAdd(&counts[52], 1);
		else								atomicAdd(&counts[53], 1);
	}
}










#endif