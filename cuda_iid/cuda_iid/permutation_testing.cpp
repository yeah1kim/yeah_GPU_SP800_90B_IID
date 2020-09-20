#include "header.h"

uint32_t permutation_testing(uint8_t *data, uint32_t size, uint32_t len, uint32_t numofparallel, uint32_t numblock, uint32_t numthread, bool verbose)
{
	double dmean = 0, dmedian = 0;
	double results[18] = { 0, };
	uint32_t counts[3 * 18] = { 0, };
	uint8_t iidcheck = 0;
	uint32_t permutation_testing_result = 0;

	calculate_statistics(&dmean, &dmedian, data, size, len);
	if (verbose) {
		printf(">---- Mean value of the original data(input): %f \n", dmean);
		printf(">---- Median value of the original data(input): %f \n\n", dmedian);
	}

	/* perform 18 Statisitcal tests on the original data(input). */
	printf("Performing 18 Statisitcal tests on the original data. \n");
	run_tests(results, dmean, dmedian, data, size, len);
	if (verbose)
		print_original_test_statistics(results);

	/* perform 10,000 iterations in parallel on the GPU. */
	printf("Performing 10,000 iterations in parallel on the GPU. \n");
	double dGPU_runtime = 0; /* runtime of 10,000 iterations measured by CUDA timer */
	gpu_permutation_testing(&dGPU_runtime, counts, results, dmean, dmedian, data, size, len, numofparallel, numblock, numthread);

	if (verbose) {
		print_counters(counts);
		printf("Run-time of the permutation testing processed in the GPU (measured by CUDA timer) : %.3f sec\n", dGPU_runtime / (double)CLOCKS_PER_SEC);
	}

	for (int t = 0; t < 18; t++) {
		if (((counts[3 * t] + counts[3 * t + 1]) > 5) && ((counts[3 * t + 1] + counts[3 * t + 2]) > 5))
			iidcheck++;
	}
	if (iidcheck == 18)
		permutation_testing_result = 1;

	return permutation_testing_result;
}






