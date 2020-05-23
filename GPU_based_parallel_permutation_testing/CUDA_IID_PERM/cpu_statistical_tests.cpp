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

#include "cpu_statistical_test_header.h"

 /*******
  * Conversion I/II that using when the input data is binary (that is, the sample size is 2)
 *******/
/**
 * @brief Divide the input data into 8-bit non-overlapping blocks and counts the number of 1s in each block.
 * @param DATA $conversion_bdata_comp Data structure for the original data(input) after conversion
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void conversion1(DATA *conversion_bdata_comp, const DATA *data_comp)
{
	uint32_t i = 0, j = 0;
	for (i = 0; i < (data_comp->nLen / 8); i++)
		for (j = 0; j < 8; j++)
			conversion_bdata_comp->pData[i] += data_comp->pData[8 * i + j];

	if ((data_comp->nLen % 8) > 0) 
		for (j = 0; j < (data_comp->nLen % 8); j++)
			conversion_bdata_comp->pData[i] += data_comp->pData[data_comp->nLen - j - 1];
}

/**
 * @brief Divide the input data into 8-bit non-overlapping blocks and calculates the integer value of each block.
 * @param DATA $conversion_bdata_comp Data structure for the original data(input) after conversion
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void conversion2(DATA *conversion_bdata_comp, const DATA *data_comp)
{
	uint32_t i = 0, j = 0;
	for (i = 0; i < (data_comp->nLen / 8); i++) {
		conversion_bdata_comp->pData[i] = 0;
		for (j = 0; j < 8; j++)
			conversion_bdata_comp->pData[i] |= ((data_comp->pData[8 * i + j] & 0x1) << (7 - j));
	}
	
	if ((data_comp->nLen % 8) > 0) 
		for (j = (data_comp->nLen % 8); j > 0; j--)
			conversion_bdata_comp->pData[i] |= data_comp->pData[data_comp->nLen - j] << (7 + j - (data_comp->nLen % 8));
	
}


/*******
 * Functions to perform 18 statistical tests on the original data
*******/
/**
 * @brief Calculate mean and median for the original data(input).
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void calculate_statsistics(TEST_COMP *test_comp, const DATA *data_comp)
{
	test_comp->dMean = 0;
	for (uint32_t i = 0; i < data_comp->nLen; i++)
		test_comp->dMean += data_comp->pData[i];
	test_comp->dMean /= (double)data_comp->nLen;

	vector<unsigned char> Data(data_comp->pData, data_comp->pData + data_comp->nLen);
	sort(Data.begin(), Data.end());

	uint32_t half = data_comp->nLen / 2;

	if (data_comp->nSample_size == 1)
		test_comp->dMedian = 0.5;
	else {
		if (data_comp->nLen % 2 == 0)
			test_comp->dMedian = (Data[half] + Data[half - 1]) / 2.0;
		else
			test_comp->dMedian = Data[half];
	}

	Data.clear();
}

/**
 * @brief Perform 18 Statisitcal tests on the original data(input).
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return uint32_t $calloc_error
 */
uint32_t statistical_tests(TEST_COMP *test_comp, const DATA *data_comp)
{
	if (data_comp->nSample_size == 1) {

		excursion_test(&test_comp->dOriginal_test_statistics[0], test_comp, data_comp);
		runs_based_on_median(&test_comp->dOriginal_test_statistics[4], &test_comp->dOriginal_test_statistics[5], test_comp, data_comp);

		DATA conversion_bdata_comp = { 0x00, };
		conversion_bdata_comp.nLen = data_comp->nLen / 8;
		if ((data_comp->nLen % 8) > 0) conversion_bdata_comp.nLen++;
		conversion_bdata_comp.nSample_size = data_comp->nSample_size;

		if((conversion_bdata_comp.pData = (uint8_t*)calloc(conversion_bdata_comp.nLen, sizeof(uint8_t)))==NULL) {
			printf("calloc error \n");
			return 0;
		}

		conversion1(&conversion_bdata_comp, data_comp);
		directional_runs_and_number_of_inc_dec(&test_comp->dOriginal_test_statistics[1], &test_comp->dOriginal_test_statistics[2], &test_comp->dOriginal_test_statistics[3], &conversion_bdata_comp);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[8], &test_comp->dOriginal_test_statistics[13], &conversion_bdata_comp, 1);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[9], &test_comp->dOriginal_test_statistics[14], &conversion_bdata_comp, 2);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[10], &test_comp->dOriginal_test_statistics[15], &conversion_bdata_comp, 8);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[11], &test_comp->dOriginal_test_statistics[16], &conversion_bdata_comp, 16);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[12], &test_comp->dOriginal_test_statistics[17], &conversion_bdata_comp, 32);

		conversion2(&conversion_bdata_comp, data_comp);
		collision_test_statistic(&test_comp->dOriginal_test_statistics[6], &test_comp->dOriginal_test_statistics[7], &conversion_bdata_comp);
		
		free(conversion_bdata_comp.pData);
	}
	else {
		excursion_test(&test_comp->dOriginal_test_statistics[0], test_comp, data_comp);
		directional_runs_and_number_of_inc_dec(&test_comp->dOriginal_test_statistics[1], &test_comp->dOriginal_test_statistics[2], &test_comp->dOriginal_test_statistics[3], data_comp);
		runs_based_on_median(&test_comp->dOriginal_test_statistics[4], &test_comp->dOriginal_test_statistics[5], test_comp, data_comp);
		collision_test_statistic(&test_comp->dOriginal_test_statistics[6], &test_comp->dOriginal_test_statistics[7], data_comp);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[8], &test_comp->dOriginal_test_statistics[13], data_comp, 1);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[9], &test_comp->dOriginal_test_statistics[14], data_comp, 2);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[10], &test_comp->dOriginal_test_statistics[15], data_comp, 8);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[11], &test_comp->dOriginal_test_statistics[16], data_comp, 16);
		periodicity_covariance_test(&test_comp->dOriginal_test_statistics[12], &test_comp->dOriginal_test_statistics[17], data_comp, 32);
	}
	return 0;
}


/*******
 * Statistical tests that are merged
 ** Directional runs and number of inc/dec = number of directional runs + length of directional runs + numbers of increases and decreases
 ** Runs based on median = number of runs based on median + length of runs based on median
 ** Collision test statistic = average collision test statistic + maximum collision test statistic
 ** Periodicity/covariance_test = periodicity test + covariance test
*******/
void excursion_test(double *out_dMax, const TEST_COMP *test_comp, const DATA *data_comp)
{
	double dTemp_max = 0;
	double dTemp = 0;
	double dRunning_sum = 0;
	uint32_t i = 0;

	for (i = 0; i < data_comp->nLen; i++) {
		dRunning_sum += data_comp->pData[i];
		dTemp = fabs(dRunning_sum - ((i + 1)*test_comp->dMean));
		if (dTemp_max < dTemp)
			dTemp_max = dTemp;
	}

	*out_dMax = dTemp_max;
}

void directional_runs_and_number_of_inc_dec(double *out_dNum, double *out_dLen, double *out_dMax, const DATA *data_comp)
{
	uint32_t nNum_runs = 1, nLen_runs = 1, nMax_len_runs = 0, nPos = 0;
	bool bFlag1 = 0, bFlag2 = 0;
	uint32_t i = 0;

	if (data_comp->pData[0] <= data_comp->pData[1])
		bFlag1 = 1;

	for (i = 1; i < data_comp->nLen - 1; i++) {
		nPos += bFlag1;
		bFlag2 = 0;

		if (data_comp->pData[i] <= data_comp->pData[i + 1])
			bFlag2 = 1;
		if (bFlag1 == bFlag2)
			nLen_runs++;
		else {
			nNum_runs++;
			if (nLen_runs > nMax_len_runs)
				nMax_len_runs = nLen_runs;
			nLen_runs = 1;
		}
		bFlag1 = bFlag2;
	}
	nPos += bFlag1;
	*out_dNum = (double)nNum_runs;
	*out_dLen = (double)nMax_len_runs;
	*out_dMax = (double)max(nPos, data_comp->nLen - nPos);
}

void runs_based_on_median(double *out_dNum, double *out_dLen, const TEST_COMP *test_comp, const DATA *data_comp)
{
	uint32_t nNum_runs = 1, nLen_runs = 1, nMax_len_runs = 0;
	bool bFlag1 = 0, bFlag2 = 0;
	uint32_t i = 0;

	if (data_comp->pData[0] >= test_comp->dMedian)
		bFlag1 = 1;

	for (i = 1; i < data_comp->nLen; i++) {
		bFlag2 = 0;

		if (data_comp->pData[i] >= test_comp->dMedian)
			bFlag2 = 1;
		if (bFlag1 == bFlag2)
			nLen_runs++;
		else {
			nNum_runs++;
			if (nLen_runs > nMax_len_runs)
				nMax_len_runs = nLen_runs;
			nLen_runs = 1;
		}
		bFlag1 = bFlag2;
	}

	*out_dNum = (double)nNum_runs;
	*out_dLen = (double)nMax_len_runs;
}

void collision_test_statistic(double *out_dAvg, double *out_dMax, const DATA *data_comp)
{
	uint32_t nSum = 0, nStart = 0, nCollision = 1, nIdxex = 0, nCollision_max = 0;
	uint32_t i = 0, j = 0;
	double dCollision_avg = 0;
	uint8_t cTemp1 = 0, cTemp2 = 0;

	while (nCollision) {
		nStart = i;
		nCollision = 0;
		for (i = nStart + 1; ((i < data_comp->nLen) && (nCollision == 0)); i++) {
			cTemp1 = data_comp->pData[i];
			for (j = nStart; j < i; j++) {
				cTemp2 = data_comp->pData[j];
				if (cTemp1 == cTemp2) {
					nCollision = i - nStart + 1;
					break;
				}
			}
			if (nCollision) {
				nSum += nCollision;
				nIdxex++;
				if (nCollision_max < nCollision)
					nCollision_max = nCollision;
			}
		}
	}
	dCollision_avg = nSum / (double)nIdxex;
	*out_dAvg = dCollision_avg - 1;
	*out_dMax = (double)nCollision_max - 1;
}

void periodicity_covariance_test(double *out_dNum, double *out_dStrength, const DATA *data_comp, uint32_t nLag)
{
	double dT1 = 0, dT2 = 0;
	uint32_t i = 0;
	for (i = 0; i < data_comp->nLen - nLag; i++) {
		if (data_comp->pData[i] == data_comp->pData[i + nLag])
			dT1++;
		dT2+= (data_comp->pData[i] * data_comp->pData[i + nLag]);
	}

	*out_dNum = dT1;
	*out_dStrength = dT2;
}


/*******
 * Functions for print the data
*******/
/**
 * @brief Print the results of 18 Statisitcal tests on the original data(input).
 * @param double $dOriginal_test_statistics[NUM_TEST] Results of 18 Statisitcal tests on the original data(input)
 * @return void
 */
void print_original_test_statistics(double dOriginal_test_statistics[NUM_TEST])
{
	printf(">---- Origianl test statistics: \n");
	printf("                        Excursion test = %0.4f \n", dOriginal_test_statistics[0]);
	printf("            Number of directional runs = %0.0f \n", dOriginal_test_statistics[1]);
	printf("            Length of directional runs = %0.0f \n", dOriginal_test_statistics[2]);
	printf("    Numbers of increases and decreases = %0.0f \n", dOriginal_test_statistics[3]);
	printf("        Number of runs based on median = %0.0f \n", dOriginal_test_statistics[4]);
	printf("        Length of runs based on median = %0.0f \n", dOriginal_test_statistics[5]);
	printf("      Average collision test statistic = %0.4f \n", dOriginal_test_statistics[6]);
	printf("      Maximum collision test statistic = %0.0f \n", dOriginal_test_statistics[7]);
	printf("           Periodicity test (lag =  1) = %0.0f \n", dOriginal_test_statistics[8]);
	printf("           Periodicity test (lag =  2) = %0.0f \n", dOriginal_test_statistics[9]);
	printf("           Periodicity test (lag =  8) = %0.0f \n", dOriginal_test_statistics[10]);
	printf("           Periodicity test (lag = 16) = %0.0f \n", dOriginal_test_statistics[11]);
	printf("           Periodicity test (lag = 32) = %0.0f \n", dOriginal_test_statistics[12]);
	printf("            Covariance test (lag =  1) = %0.0f \n", dOriginal_test_statistics[13]);
	printf("            Covariance test (lag =  2) = %0.0f \n", dOriginal_test_statistics[14]);
	printf("            Covariance test (lag =  8) = %0.0f \n", dOriginal_test_statistics[15]);
	printf("            Covariance test (lag = 16) = %0.0f \n", dOriginal_test_statistics[16]);
	printf("            Covariance test (lag = 32) = %0.0f \n", dOriginal_test_statistics[17]);
	printf(">------------------------------------------------------< \n\n");
}

/**
 * @brief Print the counters that are the ranking of the original test statistics.
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @return void
 */
void print_counters(TEST_COMP *test_comp)
{
	printf(">---- The ranking(count) of the original test statistics: \n");
	printf("	#Test	 Counter 1	Counter 2	Counter 3\n");
	for (uint32_t i = 0; i < NUM_TEST; i++)
		printf("	  %2d	%7d 	%7d 	%7d \n", i + 1, test_comp->nCount[0][i], test_comp->nCount[1][i], test_comp->nCount[2][i]);
	printf(">------------------------------------------------------< \n\n");
}






