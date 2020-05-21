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

#ifndef __CPU_STATISTICAL_TEST_H__
#define __CPU_STATISTICAL_TEST_H__

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdint.h>
using namespace std;

#define NUM_TEST	18

/* data structure for the original data(input) */
typedef struct _ORIGINAL_DATA {
	/* sample: data ovtained from one output of the (digitized) noise source */
	uint8_t *pData;			/* the original data(input), which consists of (noise) samples */
	uint32_t nSample_size;	/* the size of sample in bits (1~8)*/
	uint32_t nLen;			/* the number of samples in the original data */
	uint32_t nBlen;			/* the number of samples after conversion I/II (= nLen/8) */

	char in_file_name[100];					/* file_path\\file_name.bin */
}DATA;

/* data structure for the permutation testing */
typedef struct _TEST_COMP {
	double dMean;								/* mean value of the original data(input) */
	double dMedian;								/* median value of the original data(input) */
	double dOriginal_test_statistics[NUM_TEST]; /* the results of 18 statistical tests on the original data */
	uint32_t nCount[3][NUM_TEST] = { {0,}, };	/* the counters, that is original test statistics's rankings */
}TEST_COMP;


/*******
 * Conversion I/II that using when the input data is binary (that is, the sample size is 2)
*******/
/**
 * @brief Divide the input data into 8-bit non-overlapping blocks and counts the number of 1s in each block.
 * @param DATA $conversion_bdata_comp Data structure for the original data(input) after conversion
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void conversion1(DATA *conversion_bdata_comp, const DATA *data_comp);

/**
 * @brief Divide the input data into 8-bit non-overlapping blocks and calculates the integer value of each block.
 * @param DATA $conversion_bdata_comp Data structure for the original data(input) after conversion
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void conversion2(DATA *conversion_bdata_comp, const DATA *data_comp);


/*******
 * Functions to perform 18 statistical tests on the original data
*******/
/**
 * @brief Calculate mean and median for the original data(input).
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return void
 */
void calculate_statsistics(TEST_COMP *test_comp, const DATA *data_comp);

/**
 * @brief Perform 18 Statisitcal tests on the original data(input).
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @param DATA $data_comp Data structure for the original data(input)
 * @return uint32_t $calloc_error
 */
uint32_t statistical_tests(TEST_COMP *test_comp, const DATA *data_comp);


/*******
 * Statistical tests that are merged
 ** Directional runs and number of inc/dec = number of directional runs + length of directional runs + numbers of increases and decreases
 ** Runs based on median = number of runs based on median + length of runs based on median
 ** Collision test statistic = average collision test statistic + maximum collision test statistic
 ** Periodicity/covariance_test = periodicity test + covariance test
*******/
void excursion_test(double *out_dMax, const TEST_COMP *test_comp, const DATA *data_comp);
void directional_runs_and_number_of_inc_dec(double *out_dNum, double *out_dLen, double *out_dMax, const DATA *data_comp);
void runs_based_on_median(double *out_dNum, double *out_dLen, const TEST_COMP *test_comp, const DATA *data_comp);
void collision_test_statistic(double *out_dAvg, double *out_dMax, const DATA *data_comp);
void periodicity_covariance_test(double *out_dNum, double *out_dStrength, const DATA *data_comp, uint32_t nLag);


/*******
 * Functions for print the data
*******/
/**
 * @brief Print the results of 18 Statisitcal tests on the original data(input).
 * @param double $dOriginal_test_statistics[NUM_TEST] Results of 18 Statisitcal tests on the original data(input)
 * @return void
 */
void print_original_test_statistics(double dOriginal_test_statistics[NUM_TEST]);

/**
 * @brief Print the counters that are the ranking of the original test statistics.
 * @param TEST_COMP $test_comp Data structure for the permutation testing
 * @return void
 */
void print_counters(TEST_COMP *test_comp);

#endif
#pragma once
