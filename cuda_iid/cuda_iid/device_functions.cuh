#ifndef _DEVICE_FUNCTIONS_H_
#define _DEVICE_FUNCTIONS_H_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "header.h"

__device__ void dev_test1(double *out_max, const double mean, const uint8_t *data, const uint32_t len,
	const uint32_t N, uint32_t tid)
{
	double max = 0, temp = 0, sum = 0;
	uint32_t i = 0;
	for (i = 0; i < len; i++) {
		sum += data[i * N + tid];
		temp = fabs(sum - ((i + 1)*mean));
		if (max < temp)
			max = temp;
	}
	*out_max = max;
}

__device__ void dev_test2_6(double *out1, double *out2, double *out3, double *out4, double *out5,
	const double median, const uint8_t *data, const uint32_t len, const uint32_t N, uint32_t tid)
{
	uint32_t i;
	*out1 = 1; *out4 = 1;
	uint32_t run1 = 1, run2 = 1;
	uint32_t pos = 0;
	bool f1 = 0, f2 = 0;
	bool fm1 = 0, fm2 = 0;

	if (data[tid] <= data[N + tid])
		f1 = 1;

	if (data[tid] >= median)
		fm1 = 1;

	for (i = 1; i < (len - 1); i++) {
		pos += f1;
		f2 = 0;
		fm2 = 0;

		if (data[i*N + tid] <= data[(i + 1)*N + tid])
			f2 = 1;

		if (data[i*N + tid] >= median)
			fm2 = 1;

		if (f1 == f2)
			run1++;
		else {
			*out1 += 1;
			if (run1 > * out2)
				*out2 = run1;
			run1 = 1;
		}

		if (fm1 == fm2)
			run2++;
		else {
			*out4 += 1;
			if (run2 > * out5)
				*out5 = run2;
			run2 = 1;
		}
		f1 = f2;
		fm1 = fm2;
	}
	pos += f1;

	if (data[(len - 1)*N + tid] >= median)
		fm2 = 1;
	else
		fm2 = 0;

	if (fm1 == fm2)
		run2++;
	else {
		*out4 += 1;
		if (run2 > * out5)
			*out5 = run2;
		run2 = 1;
	}

	if (pos > (len - pos))
		*out3 = pos;
	else
		*out3 = (len - pos);
}

__device__ void dev_test7_8(double *out_average, double *out_max, const uint8_t *data, const uint32_t size,
	const uint32_t len, const uint32_t N, uint32_t tid)
{
	uint32_t i = 0, j = 0, k = 0;
	bool dups[256] = { 0, };
	uint32_t cnt = 0;
	uint32_t max = 0;
	double avg = 0;

	while (i + j < len) {
		for (k = 0; k < (uint32_t)(1 << size); k++) dups[k] = false;

		while (i + j < len) {
			if (dups[data[(i + j)*N + tid]]) {
				avg += j;
				if (j > max)
					max = j;
				cnt++;
				i += j;
				j = 0;
				break;
			}
			else {
				dups[data[(i + j)*N + tid]] = true;
				++j;
			}
		}
		++i;
	}

	*out_average = avg / (double)cnt;
	*out_max = (double)max;
}

__device__ void dev_test9_and_14(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t N, uint32_t tid, const uint32_t lag)
{
	double temp1 = 0, temp2 = 0;
	uint32_t i = 0;
	for (i = 0; i < len - lag; i++) {
		if (data[i*N + tid] == data[(i + lag)*N + tid])
			temp1++;
		temp2 += (data[i*N + tid] * data[(i + lag)*N + tid]);
	}
	*out_num = temp1;
	*out_strength = temp2;
}


// binary
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

__device__ void dev_binary_test2_4(double *out_num, double *out_len, double *out_max, const uint8_t *data,
	const uint32_t len, const uint32_t N, uint32_t tid)
{
	uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0, pos = 0;
	bool bflag1 = 0, bflag2 = 0;
	uint32_t i = 0;
	if (dev_hammingweight(data[tid]) <= dev_hammingweight(data[N + tid]))
		bflag1 = 1;

	for (i = 1; i < len - 1; i++) {
		pos += bflag1;
		bflag2 = 0;

		if (dev_hammingweight(data[i*N + tid]) <= dev_hammingweight(data[(i + 1)*N + tid]))
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

__device__ void dev_binary_test9_and_14(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len,
	const uint32_t N, uint32_t tid, const uint32_t lag)
{
	double temp1 = 0, temp2 = 0;
	uint32_t i = 0;
	for (i = 0; i < len - lag; i++) {
		if (dev_hammingweight(data[i*N + tid]) == dev_hammingweight(data[(i + lag)*N + tid]))
			temp1++;
		temp2 += (dev_hammingweight(data[i*N + tid]) * dev_hammingweight(data[(i + lag)*N + tid]));
	}
	*out_num = temp1;
	*out_strength = temp2;
}

__device__ void dev_test5_6(double *out_num, double *out_len, const double median, const uint8_t *data, const uint32_t len,
	const uint32_t N, uint32_t tid)
{
	uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0;
	bool bflag1 = 0, bflag2 = 0;
	uint32_t i = 0;

	if (data[tid] >= median)
		bflag1 = 1;

	for (i = 1; i < len; i++) {
		bflag2 = 0;

		if (data[i*N + tid] >= median)
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


#endif