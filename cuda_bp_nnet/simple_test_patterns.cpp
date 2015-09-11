/*
 * simple_test_patterns.cpp
 *
 *  Created on: Jan 1, 2015
 *      Author: dean
 */

#include <iostream>
#include <cstring>
#include <vector>
using namespace std;

#include "BPNeuralNetwork.h"
#include "simple_test_patterns.h"


namespace cublas_bp_nnet {

// very simple patterns for testing the neural network
static float four_dim_input_patterns[][NUM_FOUR_PATTERNS] = {
		{ 1, 0, 0, 0, },
		{ 0, 1, 0, 0, },
		{ 0, 0, 1, 0, },
		{ 0, 0, 0, 1, },
};

static float four_dim_target_patterns[][NUM_FOUR_PATTERNS] = {
		{ 1, 0, 0, 0, },
		{ 0, 1, 0, 0, },
		{ 0, 0, 1, 0, },
		{ 0, 0, 0, 1, },
};


static float eight_dim_input_patterns[][NUM_EIGHT_PATTERNS] = {
		{ 1, 0, 0, 0, 0, 0, 0, 0, },
		{ 0, 1, 0, 0, 0, 0, 0, 0, },
		{ 0, 0, 1, 0, 0, 0, 0, 0, },
		{ 0, 0, 0, 1, 0, 0, 0, 0, },
		{ 0, 0, 0, 0, 1, 0, 0, 0, },
		{ 0, 0, 0, 0, 0, 1, 0, 0, },
		{ 0, 0, 0, 0, 0, 0, 1, 0, },
		{ 0, 0, 0, 0, 0, 0, 0, 1, },
};

static float eight_dim_target_patterns[][NUM_EIGHT_PATTERNS] = {
		{ 1, 0, 0, 0, 0, 0, 0, 0, },
		{ 0, 1, 0, 0, 0, 0, 0, 0, },
		{ 0, 0, 1, 0, 0, 0, 0, 0, },
		{ 0, 0, 0, 1, 0, 0, 0, 0, },
		{ 0, 0, 0, 0, 1, 0, 0, 0, },
		{ 0, 0, 0, 0, 0, 1, 0, 0, },
		{ 0, 0, 0, 0, 0, 0, 1, 0, },
		{ 0, 0, 0, 0, 0, 0, 0, 1, },
};

// hard threshold function
float slam(float value) {
	if (fabs(1.0 - value) < 0.2) return 1.0;
	if (fabs(value) < 0.2) return 0.0;
}

/** originally this used the patterns above to test. (see commented out
 * code below.)  However, now the number of patterns is passed in.
 * For the encoder / decoder problem this also happens to be the
 * dimension of both the input and output layers
 *
 * Currently using the GeForce GTX 970 a 2 layer net with over 1000 units
 * in the input and output layers can be run.  Even larger networks can
 * also be run, but once the dimension exceeds a certain limit
 * (128 units? I forget...) the training fails.
 *
 * I believe this is due to the fact that only one unit is turned on
 * at a time in the input and output layers so the training signal is
 * swamped by the training noise.
 */
void test_nnet_on_simple_patterns(
		unsigned num_patterns, int epochs, BPNeuralNetwork& nnet
		)
{
#if 0
	vector<float const*> inputList;
	for (unsigned i = 0; i < NUM_FOUR_PATTERNS; i++) {
		inputList.push_back(four_dim_input_patterns[i]);
	}

	vector<float const*> targetList;
	for (unsigned i = 0; i < NUM_FOUR_PATTERNS; i++) {
		targetList.push_back(four_dim_target_patterns[i]);
	}
	for (unsigned i = 0; i < NUM_EIGHT_PATTERNS; i++) {
		inputList.push_back(eight_dim_input_patterns[i]);
	}

	vector<float const*> targetList;
	for (unsigned i = 0; i < NUM_EIGHT_PATTERNS; i++) {
		targetList.push_back(eight_dim_target_patterns[i]);
	}
#endif
	vector<float const*> inputList;
	vector<float const*> targetList;

	unsigned dim = numPatterns;
	for (unsigned i = 0; i < numPatterns; i++) {
		float* vec = new float[dim];
		for (unsigned j = 0; j < dim; j++) {
			if (i == j) vec[j] = 1.0;
			else vec[j] = 0.0;
		}
		inputList.push_back(vec);
		targetList.push_back(vec);
	}

	for (unsigned i = 0; i < epochs; i++) {
		nnet.operator ()(0.25, 0.8, inputList, targetList);
		float armse = nnet.armse(inputList, targetList);
		if (i % 10 == 0) cout << i << ": armse: " << armse << endl;
	}

	float output[numPatterns];

	for (unsigned i = 0; i < 16; i++) {
		cout << "Output for input[" << i << "]" << endl;
		float const* input = &(inputList[i][0]);
		for (unsigned j = 0; j < numPatterns; j++) {
			cout << "  " << input[j];
			if (j >= 15) break;
		}
		cout << "  ::    ";
		//nnet.operator ()(input, output);
		//int numLayers = nnet.layerList.size() - 1;
		//nnet.operator ()(input);
		//float* dev_output = nnet.layerList[numLayers];
		//cudaMemcpy(output, dev_output, nnet.layerDimensionList[numLayers], cudaMemcpyDeviceToHost);
		nnet.operator ()(inputList[i], output);
		for (unsigned j = 0; j < numPatterns; j++) {
			cout.precision(2);
			cout << "  " << slam(output[j]);
			if (j >= 15) break;
		}
		cout << endl;
	}
	float armse = nnet.armse(inputList, targetList);
	cout << "armse = " << armse << endl;
}


} /* cublas_bp_nnet */
