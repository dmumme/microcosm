/*
 * simple_test_patterns.h
 *
 *  Created on: Jan 1, 2015
 *      Author: dean
 */

#ifndef SIMPLE_TEST_PATTERNS_H_
#define SIMPLE_TEST_PATTERNS_H_


namespace cublas_bp_nnet {

class BPNeuralNetwork;

static unsigned const NUM_FOUR_PATTERNS = 4;
static unsigned const NUM_EIGHT_PATTERNS = 8;

/**
 * This is a test over the simple encoder / decoder problem
 * in which exactly one unit is turned on at the input layer
 * and the nnet learns to turn on the corresponding unit in
 * the output layer
 */
void test_nnet_on_simple_patterns(
		unsigned num_patterns,
		int epochs,
		BPNeuralNetwork& nnet);
}


#endif /* SIMPLE_TEST_PATTERNS_H_ */
