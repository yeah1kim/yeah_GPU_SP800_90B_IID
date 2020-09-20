#include "header.h"

int read_data_from_file(uint8_t *data, uint32_t size, uint32_t len, char *in_file_name)
{
	FILE *fin;
	uint8_t temp = 0;
	uint8_t mask = (1 << size) - 1;
	uint32_t i = 0;

	char filename[200];
	sprintf(filename, "%s", in_file_name);
	if ((fin = fopen(in_file_name, "rb")) == NULL) {
		printf("File open fails. \n");
		return 1;
	}
	for (i = 0; i < len; i++) {
		temp = 0;
		fread(&temp, sizeof(unsigned char), 1, fin);
		data[i] = (temp&mask);
	}
	fclose(fin);

	return 0;
}

void print_original_test_statistics(double *results)
{
	printf(">---- Origianl test statistics: \n");
	printf("                        Excursion test = %0.4f \n", results[0]);
	printf("            Number of directional runs = %0.0f \n", results[1]);
	printf("            Length of directional runs = %0.0f \n", results[2]);
	printf("    Numbers of increases and decreases = %0.0f \n", results[3]);
	printf("        Number of runs based on median = %0.0f \n", results[4]);
	printf("        Length of runs based on median = %0.0f \n", results[5]);
	printf("      Average collision test statistic = %0.4f \n", results[6]);
	printf("      Maximum collision test statistic = %0.0f \n", results[7]);
	printf("           Periodicity test (lag =  1) = %0.0f \n", results[8]);
	printf("           Periodicity test (lag =  2) = %0.0f \n", results[9]);
	printf("           Periodicity test (lag =  8) = %0.0f \n", results[10]);
	printf("           Periodicity test (lag = 16) = %0.0f \n", results[11]);
	printf("           Periodicity test (lag = 32) = %0.0f \n", results[12]);
	printf("            Covariance test (lag =  1) = %0.0f \n", results[13]);
	printf("            Covariance test (lag =  2) = %0.0f \n", results[14]);
	printf("            Covariance test (lag =  8) = %0.0f \n", results[15]);
	printf("            Covariance test (lag = 16) = %0.0f \n", results[16]);
	printf("            Covariance test (lag = 32) = %0.0f \n", results[17]);
	printf(">------------------------------------------------------< \n\n");
}

void print_counters(uint32_t *counts)
{
	printf(">---- The ranking(count) of the original test statistics: \n");
	printf("	#Test	 Counter 1	Counter 2	Counter 3\n");
	for (uint32_t i = 0; i < 18; i++)
		printf("	  %2d	%7d 	%7d 	%7d \n", i + 1, counts[3 * i], counts[3 * i + 1], counts[3 * i + 2]);
	printf(">------------------------------------------------------< \n\n");
}


uint32_t input_by_user(uint32_t *samplesize, uint32_t *len, uint32_t *numofparallel, uint32_t *numblock, uint32_t *numthread,
	bool *verbose, char *in_file_name)
{
	FILE *fin;
	uint32_t user_len = 0;						/* the number of samples in the input data (= len) */
	uint32_t file_size = 1000000;				/* the size of the input file */
	uint32_t user_num_iteration_in_parallel;	/* the number of iterations processing in parallel on the GPU */
	uint32_t user_num_block;					/* the number of CUDA blocks using in the CUDA kernel */
	uint32_t user_num_thread;					/* the number of CUDA threads using in the CUDA kernel */
	uint32_t verbose_flag;						/* optional verbosity flag for more output */

	printf("<file_name>: Must be relative path to a binary file with 1 million entries (samples).\n");
	printf("	     ex) C:\\Users\\user\\Desktop\\test_data\\truerand_1bit.bin \n");
INPUT_FILE_NAME:
	scanf("%s", in_file_name);
	if ((fin = fopen(in_file_name, "rb")) == NULL) {
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
		*len = user_len;
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
	scanf("%d", samplesize);
	if ((*samplesize < 1) || (*samplesize > 8)) {
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
		*numofparallel = user_num_iteration_in_parallel;
		*numblock = user_num_block;
		*numthread = user_num_thread;
	}
	printf("\n[verbose] Optional verbosity flag(0/1) for more output. 0(false) is the default. \n");
	scanf("%d", &verbose_flag);
	if (verbose_flag)
		*verbose = true;
	printf("\n");

	return 0;
}