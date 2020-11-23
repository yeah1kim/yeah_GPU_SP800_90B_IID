#ifndef _CUDA_IID_PERMUTATION_TESTING_H_
#define _CUDA_IID_PERMUTATION_TESTING_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <mutex>		// std::mutex
#include <omp.h>		// openmp 4.0 with gcc 4.9
using namespace std;
#include "bzip/bzlib.h"
#include "nvidia_help/helper_functions.h"
#include "nvidia_help/helper_cuda.h"

#define VERBOSE					0
#define NUMBER_OF_SAMPLES		1000000
#define PARALLELISM				2500 // in parallel
#define BLOCK					10
#define THREAD					250		

#define CALLOC_ERRORCHK(n)		{if(n==NULL) {printf("calloc error.\n"); return 1;}}
#define CUDA_ERRORCHK(n)		{if(n) {printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n"); goto Error;}}
#define CUDA_CALLOC_ERRORCHK(n) {if(n) {printf("cudaMalloc failed.\n"); goto Error;}}
#define CUDA_MEMCPY_ERRORCHK(n) {if(n) {printf("cudaMemcpy failed.\n"); goto Error;}}

// permutation_testing.cpp
bool permutation_testing(uint8_t *data, uint32_t size, uint32_t len, uint32_t numparallel, uint32_t numblock, uint32_t numthread, bool verbose);

// utils.cpp
uint32_t input_by_user(uint32_t *samplesize, uint32_t *len, uint32_t *numofparallel, uint32_t *numblock, uint32_t *numthread,
	bool *verbose, char *in_file_name);
int read_data_from_file(uint8_t *data, uint32_t size, uint32_t len, char *in_file_name);
void print_original_test_statistics(double *results);
void print_counters(uint32_t *counts);
void seed(uint64_t *xoshiro256starstarState);
void xoshiro_jump(unsigned int jump_count, uint64_t *xoshiro256starstarState);
uint64_t randomRange64(uint64_t s, uint64_t *xoshiro256starstarState);
void FYshuffle(byte data[], const int sample_size, uint64_t *xoshiro256starstarState);


// statistical_test.cpp
void calculate_statistics(double *dmean, double *dmedian, uint8_t *data, uint32_t size, uint32_t len);
int run_tests(double *results, double dmean, double dmedian, uint8_t *data, uint32_t size, uint32_t len);
void excursion_test(double *out, const double dmean, const uint8_t *data, const uint32_t len);
void directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data, const uint32_t len);
void runs_based_on_median(double *out_num, double *out_len, const double dmedian, const uint8_t *data, const uint32_t len);
int collision_test_statistic(double *out_avg, double *out_max, const uint8_t *data, const uint32_t size, const uint32_t len);
void periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len, uint32_t lag);
void compression(double *out, const uint8_t *data, const uint32_t len, const uint32_t size);

void conversion1(uint8_t *bdata, const uint8_t *data, const uint32_t len);
void conversion2(uint8_t *bdata, const uint8_t *data, const uint32_t len);


// gpu_permutation_testing.cu
bool gpu_permutation_testing(double *dgpu_runtime, uint32_t *counts, double *results, double mean, double median,
	uint8_t *data, uint32_t size, uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread);





#endif // !_CUDA_IID_PERMUTATION_TESTING_H_
