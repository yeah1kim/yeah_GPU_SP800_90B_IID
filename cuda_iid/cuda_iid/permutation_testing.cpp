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

#include "header.h"


 /**
  * @brief Identify evidence against the null hypothesis that the noise source is IID.
  *  - The process follows the process of the algorithm in NIST SP 800-90B.
  * @param uint8_t $data[]: The original data(input), which consists of (noise) samples
  * @param uint32_t $size: The size of sample in bits (1~8)
  * @param uint32_t $len: The number of samples in the original data
  * @param uint32_t $numparallel: The number of iterations processing in parallel on the GPU
  * @param uint32_t $num_block: The number of CUDA blocks
  * @param uint32_t $num_thread: The number of CUDA threads per block
  * @param bool $verbose: Verbosity flag for more output
  * @return bool $iidchk
  */
bool permutation_testing(uint8_t *data, uint32_t size, uint32_t len, uint32_t numparallel, uint32_t numblock, uint32_t numthread, bool verbose)
{
	printf("Start the permutation testing. \n");
	bool iidchk = true;
	double dmean = 0, dmedian = 0;
	double results[19] = { 0, };
	uint32_t counts[3 * 19] = { 0, };

	calculate_statistics(&dmean, &dmedian, data, size, len);
	if (verbose) {
		printf(">---- Mean value of the original data(input): %f \n", dmean);
		printf(">---- Median value of the original data(input): %f \n\n", dmedian);
	}

	/* perform 18 Statisitcal tests on the original data(input). */
	printf("Performing 19 statisitcal tests on the original data. \n");
	run_tests(results, dmean, dmedian, data, size, len);
	if (verbose)
		print_original_test_statistics(results);


	/* perform 10,000 iterations in parallel on the GPU. */
	printf("Performing 10,000 iterations(shuffling + 18 statisitcal tests) in parallel on the GPU. \n");
	double dgpu_runtime = 0; /* runtime of 10,000 iterations measured by CUDA timer */
	iidchk = gpu_permutation_testing(&dgpu_runtime, counts, results, dmean, dmedian, data, size, len, numparallel, numblock, numthread);

	if (iidchk == true) {// IID -> compression test
		printf("Performing 10,000 iterations(shuffling + compression test) using OpenMP on the CPU. \n");
		if (verbose) {
			uint32_t num_cpus = omp_get_max_threads();
			printf(">---- %d CPU threads. \n\n", num_cpus);
		}
		uint64_t xoshiro256starstarMainSeed[4];
		seed(xoshiro256starstarMainSeed);
		size_t completed = 0;
		iidchk = false;

		#pragma omp parallel
		{
			bool test_status = true;
			uint8_t *shuffled_data;
			shuffled_data = new uint8_t[len];
			double comp_result;

			uint64_t xoshiro256starstarSeed[4];
			memcpy(shuffled_data, data, sizeof(uint8_t) * len);
			memcpy(xoshiro256starstarSeed, xoshiro256starstarMainSeed, sizeof(xoshiro256starstarMainSeed));
			xoshiro_jump(omp_get_thread_num(), xoshiro256starstarSeed);

			#pragma omp for
			for (int i = 0; i < 10000; ++i) {
				if (test_status == true) {
					char statusMessage[1024];
					size_t statusMessageLength = 0;

					FYshuffle(shuffled_data, len, xoshiro256starstarSeed); // shuffling
					comp_result = 0;
					compression(&comp_result, shuffled_data, len, size); // compression test

					#pragma omp critical(resultUpdate)
					{
						if (comp_result > results[18]) {
							counts[54]++;
						}
						else if (comp_result == results[18]) {
							counts[55]++;
						}
						else {
							counts[56]++;
						}
						if ((counts[54] + counts[55] > 5) && (counts[55] + counts[56] > 5)) {
							test_status = false;
						}
					}
				}
				else {
					iidchk = true;
					completed++;
				}
			}
			delete[](shuffled_data);
		}
	}
	else {
		printf("No need to perform 10,000 compression tests using OpenMP on the CPU. \n");
	}

	printf("End the permutation testing. \n\n");
	if (verbose) {
		print_counters(counts);
		printf("Run-time of the permutation testing processed in the GPU (measured by CUDA timer) : %.3f sec\n", dgpu_runtime / (double)CLOCKS_PER_SEC);
	}

	return iidchk;
}






