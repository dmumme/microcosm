/*
 * BPNeuralNetwork.h
 *
 *  Created on: Dec 26, 2014
 *      Author: dean
 */

#ifndef BPNEURALNETWORK_H_
#define BPNEURALNETWORK_H_

#include <cassert>


#include "cublas_v2.h"

//#include <thrust/device_vector.h>
//#include <thrust/device_vector.h>


namespace cublas_bp_nnet {

struct BPNeuralNetwork {


	BPNeuralNetwork(cublasHandle_t& handle, vector<int> const& layerDimensions,
			int maxThreadsPerBlock, float noiseInitLevel=0);
	BPNeuralNetwork(vector<int> const& layerDimensions, vector<float*> const& biasList,
			vector<float*> const& weightList);
	~BPNeuralNetwork();

	// calculate root-mean-squared error for a single input-target pair
	float rmse(float const* input, float const* target) const;

	// calculate average root-mean-squared error over a set of input-target pairs
	float armse(vector<float const*> inputList,	vector<float const*> targetList) const;

	// training operators
	void operator()(float rate, float momentum, float const* input, float const* target);
	void operator()(float rate, float momentum, vector<float const*> inputList, vector<float const*> targetList);

	// excitation operator
	void operator()(float const* input) const;
	void operator()(float const* input, float* output) const;

	static void readFile(ifstream& input, vector<BPNeuralNetwork*>& nnetList);
	static void writeFile(vector<BPNeuralNetwork*>& nnetList, ofstream& output);

	vector<int>    layerDimensionList;
	vector<float*> layerList;  // these pointers are allocated on the device
	vector<float*> biasWeightList;       // device
	vector<float*> deltaBiasWeightList;  // device
	vector<float*> matrixList;        // device
	vector<float*> deltaMatrixList;   // device
	cublasHandle_t* pHandle;   // cublas context handle
	int maxThreadsPerBlock;
};

} /* namespace cublas_bp_nnet */
#endif /* BPNEURALNETWORK_H_ */
