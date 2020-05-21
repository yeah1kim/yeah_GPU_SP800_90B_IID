
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "statistical_test_header.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define INFILEPATH		"C:\\Users\\user\\Desktop\\data_p"
#define INFILENAME		"truerand_4bit" 
// 1_RawData_Entropy_0
// truerand_4bit
#define SAMPLESIZE		4
#define NUMSAMPLE		1000000   //90000  // 1000000

#define TESTSHUFF		0		// 2: Permutation, 1: Identity, 0: Random
#define NUMSHUFF		1000	
#define CYCLE			10
#define SF_NUMBLK		10
#define SF_NUMTHD		100
#define T_NUMBLK		10
#define T_NUMTHD		100

size_t GetGPURamUsage();
int  permutation_test(unsigned char *pData, int nNumSample);
cudaError_t shuffling_statisticaltests(double *dkernel_runtime, unsigned int nCount_1[NUMTEST], unsigned int nCount_2[NUMTEST],
	unsigned int nCount_3[NUMTEST], unsigned char *pData, int nNumSample, double dMean, double dMedian, double origin_result[NUMTEST]);
__global__ void setup_curand_kernel(curandState *const state, const unsigned int seed);
__global__ void shuffling_kernel(unsigned char *multidata, unsigned char *data, int nNumSample, curandState *state, int nNumShuffle);
__global__ void test_kernel(unsigned char *multidata, int nNumSample, double dMean, double dMedian, double *dev_result,
	unsigned int *dev_cnt1, unsigned int *dev_cnt2, unsigned int *dev_cnt3, int nNumShuffle);
__global__ void test_shuffling_kernel(unsigned char *multidata, unsigned char *data, int nNumSample, double dMean, double dMedian,
	double *dev_result, curandState *state, unsigned int *dev_cnt1, unsigned int *dev_cnt2, unsigned int *dev_cnt3, int nNumShuffle,
	int ncycle, int nS_numblk, int nS_numthd, int nT_numblk, int nT_numthd);

__device__ void dev_excursion(double *dOut, const double dMean, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff);
__device__ void dev_directionruns_incdec(double *dNum, double *dLen, double *dMax, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff);
__device__ void dev_runsnasedonmedian(double *dNum, double *dLen, const unsigned char *pData, const int nNumSample,
	const double dMedian, int tid, int nNumShuff);
__device__ void dev_collision(double *dAvg, double *dMax, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff);
__device__ void dev_periodicity(double *dPNum, const unsigned char *pData, const int nNumSample, int nLag,
	int tid, int nNumShuff);
__device__ void dev_covariance(double *dCStrength, const unsigned char *pData, const int nNumSample, int nLag,
	int tid, int nNumShuff);

__global__ void setup_curand_kernel(curandState *const state, const unsigned int seed)
{
	unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void shuffling_kernel(unsigned char *multidata, unsigned char *data, int nNumSample, curandState *state, int nNumShuffle)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int i = 0, j = nNumSample - 1, rnd = 0;
	unsigned char tmp = 0;

#if (TESTSHUFF==2)
	for (i = 0; i < nNumSample - tid; i++) {
		multidata[i * nNumShuffle + tid] = data[i + tid];
	}
	for (i = nNumSample - tid; i < nNumSample; i++) {
		multidata[i * nNumShuffle + tid] = data[i - (nNumSample - tid)];
	}
#elif(TESTSHUFF==1)
	for (i = 0; i < nNumSample; i++) {
		multidata[i * nNumShuffle + tid] = data[i];
	}
#elif(TESTSHUFF==0)
	for (i = 0; i < nNumSample; i++) {
		multidata[i * nNumShuffle + tid] = data[i];
	}
	while (j > 0) {
		rnd = curand(&state[tid]) % j;
		tmp = multidata[rnd * nNumShuffle + tid];
		multidata[rnd * nNumShuffle + tid] = multidata[j * nNumShuffle + tid];
		multidata[j * nNumShuffle + tid] = tmp;
		j--;
	}
#endif
}

__global__ void test_kernel(unsigned char *multidata, int nNumSample, double dMean, double dMedian, double *dev_result,
	unsigned int *dev_cnt1, unsigned int *dev_cnt2, unsigned int *dev_cnt3, int nNumShuffle)
{
	/*int k = 0, tid = 0;
	double T1 = 0, T2 = 0, T3 = 0;
	for (k = 0; k < nNumShuffle / blockDim.x; k++) {
		if (blockIdx.x == (0 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_excursion(&T1, dMean, multidata, nNumSample, tid, nNumShuffle);
			if ((float)T1 > (float)dev_result[0])			atomicAdd(&dev_cnt1[0], 1);
			else if ((float)T1 == (float)dev_result[0])		atomicAdd(&dev_cnt2[0], 1);
			else											atomicAdd(&dev_cnt3[0], 1);
		}
		if (blockIdx.x == (1 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_directionruns_incdec(&T1, &T2, &T3, multidata, nNumSample, tid, nNumShuffle);
			if (T1 > dev_result[1])			atomicAdd(&dev_cnt1[1], 1);
			else if (T1 == dev_result[1])	atomicAdd(&dev_cnt2[1], 1);
			else							atomicAdd(&dev_cnt3[1], 1);
			if (T2 > dev_result[2])			atomicAdd(&dev_cnt1[2], 1);
			else if (T2 == dev_result[2])	atomicAdd(&dev_cnt2[2], 1);
			else							atomicAdd(&dev_cnt3[2], 1);
			if (T3 > dev_result[3])			atomicAdd(&dev_cnt1[3], 1);
			else if (T3 == dev_result[3])	atomicAdd(&dev_cnt2[3], 1);
			else							atomicAdd(&dev_cnt3[3], 1);
		}
		if (blockIdx.x == (2 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_runsnasedonmedian(&T1, &T2, multidata, nNumSample, dMedian, tid, nNumShuffle);
			if (T1 > dev_result[4])			atomicAdd(&dev_cnt1[4], 1);
			else if (T1 == dev_result[4])	atomicAdd(&dev_cnt2[4], 1);
			else							atomicAdd(&dev_cnt3[4], 1);
			if (T2 > dev_result[5])			atomicAdd(&dev_cnt1[5], 1);
			else if (T2 == dev_result[5])	atomicAdd(&dev_cnt2[5], 1);
			else							atomicAdd(&dev_cnt3[5], 1);
		}
		if (blockIdx.x == (3 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_collision(&T1, &T2, multidata, nNumSample, tid, nNumShuffle);
			if (T1 > dev_result[6])			atomicAdd(&dev_cnt1[6], 1);
			else if (T1 == dev_result[6])	atomicAdd(&dev_cnt2[6], 1);
			else							atomicAdd(&dev_cnt3[6], 1);
			if (T2 > dev_result[7])			atomicAdd(&dev_cnt1[7], 1);
			else if (T2 == dev_result[7])	atomicAdd(&dev_cnt2[7], 1);
			else							atomicAdd(&dev_cnt3[7], 1);
		}
		if (blockIdx.x == (4 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_periodicity(&T1, multidata, nNumSample, 1, tid, nNumShuffle);
			if (T1 > dev_result[8])			atomicAdd(&dev_cnt1[8], 1);
			else if (T1 == dev_result[8])	atomicAdd(&dev_cnt2[8], 1);
			else							atomicAdd(&dev_cnt3[8], 1);
		}
		if (blockIdx.x == (5 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_periodicity(&T1, multidata, nNumSample, 2, tid, nNumShuffle);
			if (T1 > dev_result[9])			atomicAdd(&dev_cnt1[9], 1);
			else if (T1 == dev_result[9])	atomicAdd(&dev_cnt2[9], 1);
			else							atomicAdd(&dev_cnt3[9], 1);
		}
		if (blockIdx.x == (6 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_periodicity(&T1, multidata, nNumSample, 8, tid, nNumShuffle);
			if (T1 > dev_result[10])			atomicAdd(&dev_cnt1[10], 1);
			else if (T1 == dev_result[10])	atomicAdd(&dev_cnt2[10], 1);
			else							atomicAdd(&dev_cnt3[10], 1);
		}
		if (blockIdx.x == (7 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_periodicity(&T1, multidata, nNumSample, 16, tid, nNumShuffle);
			if (T1 > dev_result[11])			atomicAdd(&dev_cnt1[11], 1);
			else if (T1 == dev_result[11])	atomicAdd(&dev_cnt2[11], 1);
			else							atomicAdd(&dev_cnt3[11], 1);
		}
		if (blockIdx.x == (8 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_periodicity(&T1, multidata, nNumSample, 32, tid, nNumShuffle);
			if (T1 > dev_result[12])			atomicAdd(&dev_cnt1[12], 1);
			else if (T1 == dev_result[12])	atomicAdd(&dev_cnt2[12], 1);
			else							atomicAdd(&dev_cnt3[12], 1);
		}
		if (blockIdx.x == (9 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_covariance(&T1, multidata, nNumSample, 1, tid, nNumShuffle);
			if (T1 > dev_result[13])			atomicAdd(&dev_cnt1[13], 1);
			else if (T1 == dev_result[13])	atomicAdd(&dev_cnt2[13], 1);
			else							atomicAdd(&dev_cnt3[13], 1);
		}
		if (blockIdx.x == (10 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_covariance(&T1, multidata, nNumSample, 2, tid, nNumShuffle);
			if (T1 > dev_result[14])			atomicAdd(&dev_cnt1[14], 1);
			else if (T1 == dev_result[14])	atomicAdd(&dev_cnt2[14], 1);
			else							atomicAdd(&dev_cnt3[14], 1);
		}
		if (blockIdx.x == (11 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_covariance(&T1, multidata, nNumSample, 8, tid, nNumShuffle);
			if (T1 > dev_result[15])			atomicAdd(&dev_cnt1[15], 1);
			else if (T1 == dev_result[15])	atomicAdd(&dev_cnt2[15], 1);
			else							atomicAdd(&dev_cnt3[15], 1);
		}
		if (blockIdx.x == (12 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_covariance(&T1, multidata, nNumSample, 16, tid, nNumShuffle);
			if (T1 > dev_result[16])			atomicAdd(&dev_cnt1[16], 1);
			else if (T1 == dev_result[16])	atomicAdd(&dev_cnt2[16], 1);
			else							atomicAdd(&dev_cnt3[16], 1);
		}
		if (blockIdx.x == (13 + k * 14)) {
			tid = threadIdx.x + k * blockDim.x;
			dev_covariance(&T1, multidata, nNumSample, 32, tid, nNumShuffle);
			if (T1 > dev_result[17])			atomicAdd(&dev_cnt1[17], 1);
			else if (T1 == dev_result[17])	atomicAdd(&dev_cnt2[17], 1);
			else							atomicAdd(&dev_cnt3[17], 1);
		}
	}*/

	int tid = 0;
	char div = 10;
	if ((blockIdx.x / div) == 0) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_excursion(&T1, dMean, multidata, nNumSample, tid, nNumShuffle);
		if ((float)T1 > (float)dev_result[0])			atomicAdd(&dev_cnt1[0], 1);
		else if ((float)T1 == (float)dev_result[0])		atomicAdd(&dev_cnt2[0], 1);
		else											atomicAdd(&dev_cnt3[0], 1);
	}
	else if ((blockIdx.x / div) == 1) {
		double T1 = 0, T2 = 0, T3 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_directionruns_incdec(&T1, &T2, &T3, multidata, nNumSample, tid, nNumShuffle);
		if (T1 > dev_result[1])			atomicAdd(&dev_cnt1[1], 1);
		else if (T1 == dev_result[1])	atomicAdd(&dev_cnt2[1], 1);
		else							atomicAdd(&dev_cnt3[1], 1);
		if (T2 > dev_result[2])			atomicAdd(&dev_cnt1[2], 1);
		else if (T2 == dev_result[2])	atomicAdd(&dev_cnt2[2], 1);
		else							atomicAdd(&dev_cnt3[2], 1);
		if (T3 > dev_result[3])			atomicAdd(&dev_cnt1[3], 1);
		else if (T3 == dev_result[3])	atomicAdd(&dev_cnt2[3], 1);
		else							atomicAdd(&dev_cnt3[3], 1);
	}
	else if ((blockIdx.x / div) == 2) {
		double T1 = 0, T2 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_runsnasedonmedian(&T1, &T2, multidata, nNumSample, dMedian, tid, nNumShuffle);
		if (T1 > dev_result[4])			atomicAdd(&dev_cnt1[4], 1);
		else if (T1 == dev_result[4])	atomicAdd(&dev_cnt2[4], 1);
		else							atomicAdd(&dev_cnt3[4], 1);
		if (T2 > dev_result[5])			atomicAdd(&dev_cnt1[5], 1);
		else if (T2 == dev_result[5])	atomicAdd(&dev_cnt2[5], 1);
		else							atomicAdd(&dev_cnt3[5], 1);
	}
	else if ((blockIdx.x / div) == 3) {
		double T1 = 0, T2 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_collision(&T1, &T2, multidata, nNumSample, tid, nNumShuffle);
		if (T1 > dev_result[6])			atomicAdd(&dev_cnt1[6], 1);
		else if (T1 == dev_result[6])	atomicAdd(&dev_cnt2[6], 1);
		else							atomicAdd(&dev_cnt3[6], 1);
		if (T2 > dev_result[7])			atomicAdd(&dev_cnt1[7], 1);
		else if (T2 == dev_result[7])	atomicAdd(&dev_cnt2[7], 1);
		else							atomicAdd(&dev_cnt3[7], 1);
	}
	else if ((blockIdx.x / div) == 4) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_periodicity(&T1, multidata, nNumSample, 1, tid, nNumShuffle);
		if (T1 > dev_result[8])			atomicAdd(&dev_cnt1[8], 1);
		else if (T1 == dev_result[8])	atomicAdd(&dev_cnt2[8], 1);
		else							atomicAdd(&dev_cnt3[8], 1);
	}
	else if ((blockIdx.x / div) == 5) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_periodicity(&T1, multidata, nNumSample, 2, tid, nNumShuffle);
		if (T1 > dev_result[9])			atomicAdd(&dev_cnt1[9], 1);
		else if (T1 == dev_result[9])	atomicAdd(&dev_cnt2[9], 1);
		else							atomicAdd(&dev_cnt3[9], 1);
	}
	else if ((blockIdx.x / div) == 6) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_periodicity(&T1, multidata, nNumSample, 8, tid, nNumShuffle);
		if (T1 > dev_result[10])			atomicAdd(&dev_cnt1[10], 1);
		else if (T1 == dev_result[10])	atomicAdd(&dev_cnt2[10], 1);
		else							atomicAdd(&dev_cnt3[10], 1);
	}
	else if ((blockIdx.x / div) == 7) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_periodicity(&T1, multidata, nNumSample, 16, tid, nNumShuffle);
		if (T1 > dev_result[11])			atomicAdd(&dev_cnt1[11], 1);
		else if (T1 == dev_result[11])	atomicAdd(&dev_cnt2[11], 1);
		else							atomicAdd(&dev_cnt3[11], 1);
	}
	else if ((blockIdx.x / div) == 8) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_periodicity(&T1, multidata, nNumSample, 32, tid, nNumShuffle);
		if (T1 > dev_result[12])			atomicAdd(&dev_cnt1[12], 1);
		else if (T1 == dev_result[12])	atomicAdd(&dev_cnt2[12], 1);
		else							atomicAdd(&dev_cnt3[12], 1);
	}
	else if ((blockIdx.x / div) == 9) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_covariance(&T1, multidata, nNumSample, 1, tid, nNumShuffle);
		if (T1 > dev_result[13])			atomicAdd(&dev_cnt1[13], 1);
		else if (T1 == dev_result[13])	atomicAdd(&dev_cnt2[13], 1);
		else							atomicAdd(&dev_cnt3[13], 1);
	}
	else if ((blockIdx.x / div) == 10) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_covariance(&T1, multidata, nNumSample, 2, tid, nNumShuffle);
		if (T1 > dev_result[14])			atomicAdd(&dev_cnt1[14], 1);
		else if (T1 == dev_result[14])	atomicAdd(&dev_cnt2[14], 1);
		else							atomicAdd(&dev_cnt3[14], 1);
	}
	else if ((blockIdx.x / div) == 11) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_covariance(&T1, multidata, nNumSample, 8, tid, nNumShuffle);
		if (T1 > dev_result[15])			atomicAdd(&dev_cnt1[15], 1);
		else if (T1 == dev_result[15])	atomicAdd(&dev_cnt2[15], 1);
		else							atomicAdd(&dev_cnt3[15], 1);
	}
	else if ((blockIdx.x / div) == 12) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_covariance(&T1, multidata, nNumSample, 16, tid, nNumShuffle);
		if (T1 > dev_result[16])			atomicAdd(&dev_cnt1[16], 1);
		else if (T1 == dev_result[16])	atomicAdd(&dev_cnt2[16], 1);
		else							atomicAdd(&dev_cnt3[16], 1);
	}
	else if ((blockIdx.x / div) == 13) {
		double T1 = 0;
		tid = threadIdx.x + (blockIdx.x % div) * blockDim.x;
		dev_covariance(&T1, multidata, nNumSample, 32, tid, nNumShuffle);
		if (T1 > dev_result[17])			atomicAdd(&dev_cnt1[17], 1);
		else if (T1 == dev_result[17])	atomicAdd(&dev_cnt2[17], 1);
		else							atomicAdd(&dev_cnt3[17], 1);
	}


}

__global__ void test_shuffling_kernel(unsigned char *multidata, unsigned char *data, int nNumSample, double dMean, double dMedian,
	double *dev_result, curandState *state, unsigned int *dev_cnt1, unsigned int *dev_cnt2, unsigned int *dev_cnt3, int nNumShuffle,
	int ncycle, int nS_numblk, int nS_numthd, int nT_numblk, int nT_numthd)
{
	int i = 0;
	for (i = 0; i < ncycle; i++) {
		if (threadIdx.x == 0)
			test_kernel << <nT_numblk, nT_numthd >> > (multidata + (i % 2)*nNumShuffle*nNumSample, nNumSample, dMean, dMedian, dev_result,
				dev_cnt1, dev_cnt2, dev_cnt3, nNumShuffle);
		else
			shuffling_kernel << < nS_numblk, nS_numthd >> > (multidata + ((i + 1) % 2)*nNumShuffle*nNumSample, data, nNumSample, state, nNumShuffle);

		__syncthreads();
	}
}

int main()
{
	unsigned char pData[NUMSAMPLE] = { 0, };
	int nNumSample = NUMSAMPLE;
	unsigned char nSizeSample = SAMPLESIZE;
	unsigned char mask = (1 << nSizeSample) - 1;

	FILE *fin;
	char infilename[100] = { 0, };
	sprintf(infilename, "%s\\%s.bin", INFILEPATH, INFILENAME);
	if ((fin = fopen(infilename, "rb")) == NULL) {
		printf("file open fail in 'main' 1st fopen. \n");
		return 0;
	}

	unsigned char temp = 0;
	for (int j = 0; j < nNumSample; j++) {
		temp = 0;
		fread(&temp, sizeof(unsigned char), 1, fin);
		pData[j] = (temp&mask);
	}
	fclose(fin);

	clock_t start, end;
	start = clock();
	srand((unsigned int)time(NULL));

	permutation_test(pData, nNumSample);

	end = clock();
	printf("Permutation Test Running Time : %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
	
	return 0;
}

int permutation_test(unsigned char *pData, int nNumSample)
{
	double dMean = 0, dMedian = 0;
	Calc_Stats(&dMean, &dMedian, pData, nNumSample);
	//printf("Data's Mean: %f \n", dMean);
	//printf("Data's Median: %0.0f \n", dMedian);

	double origin_result[NUMTEST] = { 0, };
	printf("Calculating statistics on original sequence. \n");
	StatisticalTests(origin_result, pData, nNumSample, dMean, dMedian);
	Print_OriginResult(origin_result);

	unsigned int nCount_1[NUMTEST] = { 0, };
	unsigned int nCount_2[NUMTEST] = { 0, };
	unsigned int nCount_3[NUMTEST] = { 0, };
	printf("Calculating statistics on permuted sequences. \n");

	//parallel
	double dkernel_runtime = 0;
	cudaError_t cudaStatus = shuffling_statisticaltests(&dkernel_runtime, nCount_1, nCount_2, nCount_3, pData, nNumSample, dMean,
		dMedian, origin_result);
	Print_ShuffleResult(nCount_1, nCount_2, nCount_3);
	printf("Permutation Test Running Time(CUDA) : %.3f sec\n", dkernel_runtime / (double)CLOCKS_PER_SEC);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}

size_t GetGPURamUsage()
{
	//cudaSetDevice(_NumGPU);

	size_t l_free = 0;
	size_t l_Total = 0;
	cudaError_t error_id = cudaMemGetInfo(&l_free, &l_Total);

	return (l_Total - l_free);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t shuffling_statisticaltests(double *dkernel_runtime, unsigned int nCount_1[NUMTEST], unsigned int nCount_2[NUMTEST],
	unsigned int nCount_3[NUMTEST], unsigned char *pData, int nNumSample, double dMean, double dMedian, double origin_result[NUMTEST])
{
	unsigned char *dev_data = 0;
	unsigned char *dev_multidata = 0;
	double *dev_result = 0;
	unsigned int *dev_cnt1 = 0;
	unsigned int *dev_cnt2 = 0;
	unsigned int *dev_cnt3 = 0;
	curandState *dev_curand;
	cudaError_t cudaStatus;
	cudaEvent_t cuda_st, cuda_end;
	cudaEvent_t cuda_rnd_init;
	cudaEventCreate(&cuda_st);
	cudaEventCreate(&cuda_end);
	cudaEventCreate(&cuda_rnd_init);

	/*size_t freebyte, totalbyte;
	cudaStatus = cudaMemGetInfo(&freebyte, &totalbyte);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shuffling_statisticaltests!\n", cudaStatus);
		goto Error;
	}
	double free_db = (double)freebyte;
	double total_db = (double)totalbyte;
	double used_db = total_db - free_db;
	printf("GPU memeory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0,
		free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);*/

	//size_t before = GetGPURamUsage();
	//size_t after = GetGPURamUsage();
	//printf("GPU memeory usage: used = %f MB\n", (after - before) / 1024.0 / 1024.0);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_data, nNumSample * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_data)!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_multidata, nNumSample * NUMSHUFF * 2 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_multidata)!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_result, NUMTEST * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_result)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cnt1, NUMTEST * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_cnt1)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cnt2, NUMTEST * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_cnt2)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cnt3, NUMTEST * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_cnt3)!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_curand, NUMSHUFF * sizeof(curandState));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed(dev_curand)!");
		goto Error;
	}
	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data, pData, nNumSample * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(dev_data)!");
		goto Error;
	}	

	cudaStatus = cudaMemcpy(dev_result, origin_result, NUMTEST * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(dev_result)!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cnt1, nCount_1, NUMTEST * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(dev_cnt1)!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cnt2, nCount_2, NUMTEST * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(dev_cnt2)!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_cnt3, nCount_3, NUMTEST * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed(dev_cnt3)!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int nS_numblk = SF_NUMBLK, nS_numthd = SF_NUMTHD;
	int nT_numblk = 14 * T_NUMBLK, nT_numthd = T_NUMTHD;
	int nCycle = CYCLE - 1;
	int nNumShuffle = NUMSHUFF;

	cudaEventRecord(cuda_st, 0);
	 
	setup_curand_kernel << < nS_numblk, nS_numthd >> > (dev_curand, (unsigned int)time(NULL));

	cudaEventRecord(cuda_rnd_init, 0);

	shuffling_kernel << < nS_numblk, nS_numthd >> > (dev_multidata, dev_data, nNumSample, dev_curand, nNumShuffle);

	test_shuffling_kernel << < 1, 2 >> > (dev_multidata, dev_data, nNumSample, dMean, dMedian, dev_result, dev_curand, dev_cnt1,
		dev_cnt2, dev_cnt3, nNumShuffle, nCycle, nS_numblk, nS_numthd, nT_numblk, nT_numthd);

	test_kernel << < nT_numblk, nT_numthd >> > (dev_multidata + (nCycle % 2)*nNumShuffle*nNumSample, nNumSample, dMean, dMedian, dev_result,
		dev_cnt1, dev_cnt2, dev_cnt3, nNumShuffle);

	cudaEventRecord(cuda_end, 0);
	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shuffling_statisticaltests launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching shuffling_statisticaltests!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(nCount_1, dev_cnt1, NUMTEST * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(nCount_2, dev_cnt2, NUMTEST * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(nCount_3, dev_cnt3, NUMTEST * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	float cuda_time;
	cudaEventElapsedTime(&cuda_time, cuda_st, cuda_end);
	*dkernel_runtime = cuda_time;

	float cuda_init;
	cudaEventElapsedTime(&cuda_init, cuda_st, cuda_rnd_init);
	printf("CURAND init Time(CUDA) : %.3f sec\n", cuda_init / (float)CLOCKS_PER_SEC);
	
	cudaEventDestroy(cuda_st);
	cudaEventDestroy(cuda_end);
	cudaEventDestroy(cuda_rnd_init);

Error:
	cudaFree(dev_data);
	cudaFree(dev_multidata);
	cudaFree(dev_result);
	cudaFree(dev_cnt1);
	cudaFree(dev_cnt2);
	cudaFree(dev_cnt3);
	cudaFree(dev_curand);
	
	return cudaStatus;
}


__device__ void dev_excursion(double *dOut, const double dMean, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff)
{
	double dMax = 0;
	double d_i = 0;
	double running_sum = 0;
	int i = 0;

	for (i = 0; i < nNumSample; i++) {
		running_sum += pData[i*nNumShuff + tid];
		d_i = fabs(running_sum - ((i + 1)*dMean));
		if (dMax < d_i)
			dMax = d_i;
	}

	*dOut = dMax;
}

__device__ void dev_directionruns_incdec(double *dNum, double *dLen, double *dMax, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff)
{
	unsigned int num_runs = 1;
	unsigned int len_runs = 1;
	unsigned int max_len_runs = 0;
	unsigned int pos = 0;
	bool bflag1 = 0;
	bool bflag2 = 0;
	int i = 0;

	if (pData[tid] <= pData[nNumShuff + tid])
		bflag1 = 1;

	for (i = 1; i < nNumSample - 1; i++) {
		pos += bflag1;
		bflag2 = 0;

		if (pData[i*nNumShuff + tid] <= pData[(i + 1)*nNumShuff + tid])
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
	*dNum = num_runs;
	*dLen = max_len_runs;
	*dMax = max(pos, nNumSample - pos);
}

__device__ void dev_runsnasedonmedian(double *dNum, double *dLen, const unsigned char *pData, const int nNumSample,
	const double dMedian, int tid, int nNumShuff)
{
	unsigned int num_runs = 1;
	unsigned int len_runs = 1;
	unsigned int max_len_runs = 0;
	bool bflag1 = 0;
	bool bflag2 = 0;
	int i = 0;

	if (pData[tid] >= dMedian)
		bflag1 = 1;

	for (i = 1; i < nNumSample; i++) {
		bflag2 = 0;

		if (pData[i*nNumShuff + tid] >= dMedian)
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

	*dNum = num_runs;
	*dLen = max_len_runs;
}

__device__ void dev_collision(double *dAvg, double *dMax, const unsigned char *pData, const int nNumSample,
	int tid, int nNumShuff)
{
	unsigned int nSum = 0, nSt = 0, nCnt = 1, nIdx = 0, col_max = 0;
	int i = 0, j = 0;
	double col_avg = 0;
	unsigned char c1 = 0, c2 = 0;

	while (nCnt) {
		nSt = i;
		nCnt = 0;
		for (i = nSt + 1; ((i < nNumSample) && (nCnt == 0)); i++) {
			c1 = pData[i*nNumShuff + tid];
			for (j = nSt; j < i; j++) {
				c2 = pData[j*nNumShuff + tid];
				if (c1 == c2) {
					nCnt = i - nSt + 1;
					break;
				}
			}
			if (nCnt) {
				nSum += nCnt;
				nIdx++;
				if (col_max < nCnt)
					col_max = nCnt;
			}
		}
	}
	col_avg = nSum / (double)nIdx;
	*dAvg = col_avg;
	*dMax = col_max;
}

__device__ void dev_periodicity(double *dPNum, const unsigned char *pData, const int nNumSample, int nLag,
	int tid, int nNumShuff)
{
	double T = 0;
	int i = 0;
	for (i = 0; i < nNumSample - nLag; i++)
		if (pData[i*nNumShuff + tid] == pData[(i + nLag)*nNumShuff + tid])
			T++;
	*dPNum = T;
}

__device__ void dev_covariance(double *dCStrength, const unsigned char *pData, const int nNumSample, int nLag,
	int tid, int nNumShuff)
{
	double T = 0;
	int i = 0;
	for (i = 0; i < nNumSample - nLag; i++)
		T += (pData[i*nNumShuff + tid] * pData[(i + nLag)*nNumShuff + tid]);
	*dCStrength = T;
}

