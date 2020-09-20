#include "header.h"

int main(void)
{
	clock_t start, end;
	uint32_t permutation_testing_result = 0;
	uint32_t samplesize = SAMPLE_SIZE;
	uint32_t len = NUMBER_OF_SAMPLES;
	uint32_t numofparallel = PARALLELISM;
	uint32_t numblock = BLOCK;
	uint32_t numthread = THREAD;
	bool verbose = VERBOSE;
	char in_file_name[200];
	uint8_t *data;

	/* set the parameter and read the data from the input file */
	input_by_user(&samplesize, &len, &numofparallel, &numblock, &numthread, &verbose, in_file_name);
	CALLOC_ERRORCHK((data = (uint8_t*)calloc(len, sizeof(uint8_t))));
	read_data_from_file(data, samplesize, len, in_file_name);

	/* start the permutation testing */
	printf("Start the permutation testing. \n\n");
	start = clock();
	permutation_testing_result = permutation_testing(data, samplesize, len, numofparallel, numblock, numthread, verbose);
	end = clock();
	printf("Run-time of the permutation testing : %.3f sec\n\n", (double)(end - start) / CLOCKS_PER_SEC);

	/* print the result of the permutation testing. */
	if (permutation_testing_result)
		printf("==> Assume that the noise source outputs are IID! \n\n");
	else
		printf("==> Reject the IID assumption! \n\n");

	free(data);

	return 0;
}


