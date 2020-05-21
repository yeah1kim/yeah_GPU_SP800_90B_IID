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

#include "gpu_permutation_testing_header.cuh"

int main(void)
{
	DATA data_comp = { 0x00, };
	USER_INPUT_GPU_COMP user_gpu_comp = { 0x00, };
	uint32_t permutation_testing_result = 0;
	clock_t start, end;

	/* set the parameter and read the data from the input file */
	set_parameter_by_default(&data_comp, &user_gpu_comp);
	input_by_user(&data_comp, &user_gpu_comp);
	read_data_from_file(&data_comp);

	start = clock();

	/* start the permutation testing */
	printf("Start the permutation testing. \n\n");
	permutation_testing(&permutation_testing_result, &data_comp, &user_gpu_comp);

	end = clock();
	printf("Run-time of the permutation testing : %.3f sec\n\n", (double)(end - start) / CLOCKS_PER_SEC);

	/* print the result of the permutation testing. */
	if (permutation_testing_result)
		printf("==> Assume that the noise source outputs are IID! \n\n");
	else
		printf("==> Reject the IID assumption! \n\n");

	/* free */
	free(data_comp.pData);
	return 0;
}