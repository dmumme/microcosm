/*
 * BPNeuralNetwork.cpp
 *
 *  Created on: Dec 26, 2014
 *      Author: dean
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include "neuro_vision_utility.h"
#include "BPNeuralNetwork.h"
#include "simple_test_patterns.h"


//#include <thrust/transform.h>
//#include <thrust/sequence.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }



#define CUBLAS_CHECK_RETURN(value) {											\
		cublasStatus_t _m_cublasStat = value;										\
		if (_m_cublasStat != CUBLAS_STATUS_SUCCESS) {										\
			fprintf(stderr, "Error %d at line %d in file %s\n",					\
					_m_cublasStat, __LINE__, __FILE__);		\
					exit(1);															\
		} }

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void sv(int dim, float const* v) {
	for (unsigned i = 0; i < dim; i++) {
		cout << "  " << v[i];
	}
}

void sm(int rows, int cols, float const* m) {
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			cout << "  " << m[IDX2C(i,j, rows)];
		}
		cout << " ::";
	}
}


namespace cublas_bp_nnet {

struct sigmoid_functor
{
	__host__ __device__
	float operator()(const float x) const {
		return 1.0 / (1.0 + expf(- x));
	}
};

__global__
void vector_add(int dim, float const* a, float const* b, float* c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < dim) {
		c[i] = a[i] + b[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__
void vector_sub(int dim, float const* a, float const* b, float* c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < dim) {
		c[i] = a[i] - b[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__
void vector_mpy(int dim, float const* a, float const* b, float* c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < dim) {
		c[i] = a[i] * b[i];
		i += blockDim.x * gridDim.x;
	}
}

__global__
void vector_sigmoid(int dim, float const* a, float* b) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < dim) {
		b[i] = 1.0 / (1.0 + expf(- a[i]));
		i += blockDim.x * gridDim.x;
	}
}


BPNeuralNetwork::BPNeuralNetwork(cublasHandle_t& handle,
		vector<int> const& layerDimensions, int maxThreadsPerBlock,
		float noiseInitLevel) // dflt == 0
: layerDimensionList(layerDimensions),
  pHandle(&handle),
  maxThreadsPerBlock(maxThreadsPerBlock) // change this to be a query of the device
{
	int numLayers = layerDimensionList.size();

	assert(numLayers > 1);

	float* dev_layerPtr;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_layerPtr, layerDimensionList[0] * sizeof(float)));
	layerList.push_back(dev_layerPtr);

	for (unsigned layerNum = 1; layerNum < numLayers; layerNum++) {

		int inDim  = layerDimensionList[layerNum - 1];
		int outDim = layerDimensionList[layerNum];

		CUDA_CHECK_RETURN(cudaMalloc(&dev_layerPtr, outDim  * sizeof(float)));
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		layerList.push_back(dev_layerPtr);

		float* dev_biasPtr;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_biasPtr, outDim * sizeof(float)));
		if (noiseInitLevel > 0) {
			float* temp = new float[outDim];
			// fill matrix with random noise in [-noiseInitLevel, noiseInitLevel]
			for (unsigned i = 0; i < outDim; i++) {
				temp[i] = -noiseInitLevel + (2 * noiseInitLevel * rand()) / ((float)RAND_MAX);
			}
			CUBLAS_CHECK_RETURN(cublasSetVector(outDim, sizeof(float), temp, 1, dev_biasPtr, 1));
			delete [] temp;
		} else {
			CUDA_CHECK_RETURN(cudaMemset(dev_biasPtr, 0, outDim * sizeof(float)));
			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		biasWeightList.push_back(dev_biasPtr);

		float* dev_deltaBias;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_deltaBias, outDim * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemset(dev_deltaBias, 0, outDim * sizeof(float)));
		deltaBiasWeightList.push_back(dev_deltaBias);

		float* dev_matrix;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_matrix, inDim * outDim * sizeof(float)));

		if (noiseInitLevel > 0) {
			float* temp = new float[inDim * outDim];
			// fill matrix with random noise in [-noiseInitLevel, noiseInitLevel]
			for (unsigned i = 0; i < inDim * outDim; i++) {
				temp[i] = -noiseInitLevel + (2 * noiseInitLevel * rand()) / ((float)RAND_MAX);
			}
			CUBLAS_CHECK_RETURN(cublasSetVector(inDim * outDim, sizeof(float), temp, 1, dev_matrix, 1));
			delete [] temp;
		} else {
			CUDA_CHECK_RETURN(cudaMemset(dev_matrix, 0, inDim * outDim * sizeof(float)));
		}
		matrixList.push_back(dev_matrix);

		float* dev_deltaMatrix;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_deltaMatrix, inDim * outDim * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemset(dev_deltaMatrix, 0, inDim * outDim * sizeof(float)));
		deltaMatrixList.push_back(dev_deltaMatrix);
	}
}

BPNeuralNetwork::BPNeuralNetwork(vector<int> const& layerDimensions,
		vector<float*> const& biasList,	vector<float*> const& weightList)
: layerDimensionList(layerDimensions),
  pHandle(NULL),
  maxThreadsPerBlock(0) // change this to be a query of the device
{
	int numLayers = layerDimensionList.size();

	assert(numLayers > 1);

	float* dev_layerPtr;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_layerPtr, layerDimensionList[0] * sizeof(float)));
	layerList.push_back(dev_layerPtr);

	for (unsigned layerNum = 1; layerNum < numLayers; layerNum++) {

		int inDim  = layerDimensionList[layerNum - 1];
		int outDim = layerDimensionList[layerNum];

		CUDA_CHECK_RETURN(cudaMalloc(&dev_layerPtr, outDim  * sizeof(float)));
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		layerList.push_back(dev_layerPtr);

		float* dev_biasPtr;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_biasPtr, outDim * sizeof(float)));
		// todo  do a set vector using the passed-in bias weights
		biasWeightList.push_back(dev_biasPtr);

		float* dev_deltaBias;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_deltaBias, outDim * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemset(dev_deltaBias, 0, outDim * sizeof(float)));
		deltaBiasWeightList.push_back(dev_deltaBias);

		float* dev_matrix;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_matrix, inDim * outDim * sizeof(float)));
		// todo  do a set vector using the passed-in weights
		matrixList.push_back(dev_matrix);

		float* dev_deltaMatrix;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_deltaMatrix, inDim * outDim * sizeof(float)));
		CUDA_CHECK_RETURN(cudaMemset(dev_deltaMatrix, 0, inDim * outDim * sizeof(float)));
		deltaMatrixList.push_back(dev_deltaMatrix);
	}
}


BPNeuralNetwork::~BPNeuralNetwork() {
	float* dev_ptr = NULL;
	for (unsigned i = 1; i < biasWeightList.size(); i++) {
		dev_ptr = biasWeightList[i - 1];
		CUDA_CHECK_RETURN(cudaFree(dev_ptr));
	}
	for (unsigned i = 0; i < layerList.size(); i++) {
		dev_ptr = layerList[i];
		CUDA_CHECK_RETURN(cudaFree(dev_ptr));
	}
	for (unsigned i = 0; i < matrixList.size(); i++) {
		dev_ptr = matrixList[i];
		CUDA_CHECK_RETURN(cudaFree(dev_ptr));
	}
	for (unsigned i = 0; i < deltaMatrixList.size(); i++) {
		dev_ptr = deltaMatrixList[i];
		CUDA_CHECK_RETURN(cudaFree(dev_ptr));
	}
}

float BPNeuralNetwork::rmse(float const* input,	float const* target) const {
	float result;

	operator ()(input);
	int outputLayer = layerList.size() - 1;
	int outputDim   = layerDimensionList[outputLayer];

	float* dev_output = layerList[outputLayer];

	float* dev_target;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_target, outputDim * sizeof(float)));
	CUBLAS_CHECK_RETURN(cublasSetVector(outputDim, sizeof(float), target, 1, dev_target, 1));

	float* dev_difference;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_difference, outputDim * sizeof(float)));
	CUBLAS_CHECK_RETURN(cublasScopy(*pHandle, outputDim, dev_target, 1, dev_difference, 1));

	float alpha = -1.0;
	CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &alpha, dev_output, 1, dev_difference, 1));
	CUBLAS_CHECK_RETURN(cublasSnrm2(*pHandle, outputDim, dev_difference, 1, &result));

	CUDA_CHECK_RETURN(cudaFree(dev_difference));
	CUDA_CHECK_RETURN(cudaFree(dev_target));

	return result;
}

float BPNeuralNetwork::armse(vector<float const*> inputList,	vector<float const*> targetList) const {
	assert(! inputList.empty() && inputList.size() == targetList.size());
	float sum;
	for (unsigned i = 0; i < inputList.size(); i++) {
		sum += rmse(inputList[i], targetList[i]);
	}
	return sum / inputList.size();
}

// training operators

void BPNeuralNetwork::operator()(float rate, float momentum,
		float const* input, float const* target)
{
	assert(rate >= 0);
	int numLayers = layerDimensionList.size();
	int outputDim = layerDimensionList[numLayers - 1];

	// compute the activations
	operator()(input);

	float const* dev_output = layerList[numLayers - 1]; // dev_output = o  (output)

	float* dev_deltas;
	CUDA_CHECK_RETURN(cudaMalloc(&dev_deltas, outputDim * sizeof(float)));
	CUBLAS_CHECK_RETURN(cublasSetVector(outputDim, sizeof(float), target, 1, dev_deltas, 1)); // deltas = t

	float alpha = -1.0;
	CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &alpha,
			layerList[numLayers - 1], 1, dev_deltas, 1)); // deltas = (t - o)

	int previousLayerNum;
	int layerNum;

	//////////////////////////////////////////////////////////////////
	// Start iterations for weight corrections at each layer
	for (unsigned i = 1; i < numLayers; i++) {
		layerNum = numLayers - i; // start at output layer and work backwards
		previousLayerNum = layerNum - 1;

		int inputDim = layerDimensionList[previousLayerNum];

		// calculate fPrime
		float* ones = new float[outputDim];
		for (unsigned i = 0; i < outputDim; i++) {
			ones[i] = 1.0;
		}

		float* dev_fPrime;
		CUDA_CHECK_RETURN(cudaMalloc(&dev_fPrime, outputDim * sizeof(float)));
		CUBLAS_CHECK_RETURN(cublasSetVector(outputDim, sizeof(float), ones, 1, dev_fPrime, 1)); // fPrime = (1)

		alpha = -1.0;
		CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &alpha, layerList[layerNum], 1, dev_fPrime, 1)); // fPrime = (1 - o)

		int numBlocks = (outputDim + (maxThreadsPerBlock - 1)) / maxThreadsPerBlock;
		vector_mpy<<<numBlocks, maxThreadsPerBlock>>>(outputDim, layerList[layerNum], dev_fPrime, dev_fPrime);// fPrime = o(1-o)
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());


		// multipy deltas by fPrime
		vector_mpy<<<numBlocks, maxThreadsPerBlock>>>(outputDim, dev_deltas, dev_fPrime, dev_deltas);// deltas = eo(1-o)
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaFree(dev_fPrime));

		float* dev_previousLayer = layerList[previousLayerNum];

		float* dev_matrix = matrixList[previousLayerNum];

		////////////////////// Update the weight matrix

		// scale down the delta-weight-matrix by the momentum term

		float* dev_deltaMatrix = deltaMatrixList[previousLayerNum];
		CUBLAS_CHECK_RETURN(cublasSscal(*pHandle, inputDim * outputDim,
				&momentum, dev_deltaMatrix, 1));

		// compute rate * outer-product matrix and add it to the delta-weight-matrix
		CUBLAS_CHECK_RETURN(cublasSger(*pHandle, outputDim, inputDim, &rate,
				dev_deltas, 1, dev_previousLayer, 1, dev_deltaMatrix, outputDim));

		// update the weight-matrix = wm + Dwm
		alpha = 1.0;
		CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, inputDim * outputDim, &alpha, dev_deltaMatrix, 1, dev_matrix, 1));

		////////////////////// Update the biases

		// scale down the delta-biases by the momentum term
		float* dev_deltaBias = deltaBiasWeightList[previousLayerNum];
		CUBLAS_CHECK_RETURN(cublasSscal(*pHandle, outputDim, &momentum, dev_deltaBias, 1));

		// no need compute the new corrections: they are equal to dev_deltas. just add them
		// into the delta-biases
		CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &rate, dev_deltas, 1, dev_deltaBias, 1));

		// update the biases
		float* dev_biasPtr = biasWeightList[previousLayerNum];

		alpha = 1.0;
		CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &alpha, dev_deltaBias, 1, dev_biasPtr, 1));

		// no need to compute deltas beyond layer one
		if (previousLayerNum == 0) {
			CUDA_CHECK_RETURN(cudaFree(dev_deltas));
			break;
		}

		float* dev_previousLayerDeltas; // deltas for previous layer
		CUDA_CHECK_RETURN(cudaMalloc(&dev_previousLayerDeltas, inputDim * sizeof(float)));

		float beta  = 0.0;

		// mpy by the matrix transpose to go back down the layers
		CUBLAS_CHECK_RETURN(cublasSgemv(*pHandle,
				CUBLAS_OP_T, // use transpose of weight matrix
				outputDim,
				inputDim,
				&alpha,
				dev_matrix,
				outputDim,// leading dim of column-major matrix
				dev_deltas,// input
				1,
				&beta,
				dev_previousLayerDeltas,  // output: priorDeltas = e
				1));

		// point past bias unit
		//		dev_output = layerList[previousLayerNum];
		outputDim  = inputDim;

		CUDA_CHECK_RETURN(cudaFree(dev_deltas));
		dev_deltas = dev_previousLayerDeltas; // deltas = e
	} // for
}

// train a list of input-target pairs
void BPNeuralNetwork::operator()(float rate, float momentum,
		vector<float const*> inputList, vector<float const*> targetList)
{
	assert(inputList.size() > 0 && inputList.size() == targetList.size());

	int numPairs = inputList.size();

	vector<int> indices(numPairs);
	sequence_fill(0, 1, indices);
	in_place_shuffle(indices);

	for (unsigned i = 0; i < numPairs; i++) {

		float const* input = inputList[indices[i]];
		float const* target = targetList[indices[i]];

		operator()(rate, momentum, input, target);
	}
}

//////////////////////////////////////////////////////
// calculate activiations for the given input
void BPNeuralNetwork::operator()(float const* input) const
{
	int numLayers = layerDimensionList.size();
	int inputDim  = layerDimensionList[0];

	float* dev_input = layerList[0];
	CUBLAS_CHECK_RETURN(cublasSetVector(inputDim, sizeof(float), input, 1, dev_input, 1));

	float alpha = 1.0;
	float beta  = 0.0;


	for (unsigned layerNum = 1; layerNum < numLayers; layerNum++) {

		int outputDim = layerDimensionList[layerNum];
		//float* dev_output = layerList[layerNum];

		float* dev_matrix = matrixList[layerNum - 1];

		// do output = matrix * input
		CUBLAS_CHECK_RETURN(cublasSgemv(*pHandle,
				CUBLAS_OP_N,
				outputDim,
				inputDim,
				&alpha,
				matrixList[layerNum - 1],
				//				dev_matrix,
				outputDim,// leading dim of matrix
				dev_input,
				1,
				&beta,
				layerList[layerNum],// output
				//				dev_output,
				1));

		// add biases
		float* dev_biasWeighs = biasWeightList[layerNum - 1];
		CUBLAS_CHECK_RETURN(cublasSaxpy(*pHandle, outputDim, &alpha,
				biasWeightList[layerNum - 1], 1, layerList[layerNum], 1)); // add biases to outputs

		int numBlocks = (outputDim + (maxThreadsPerBlock - 1)) / maxThreadsPerBlock;
		vector_sigmoid<<<numBlocks, maxThreadsPerBlock>>>(outputDim, layerList[layerNum], layerList[layerNum]);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		inputDim  = layerDimensionList[layerNum];

		//dev_input = dev_output;//layerList[layerNum];
		dev_input = layerList[layerNum];
	}
}

// calculate activiations for the given input and
// return the output activiations
void BPNeuralNetwork::operator()(float const* input, float* output) const {
	int numLayers = layerDimensionList.size();

	operator()(input);
	int outputDim = layerDimensionList[numLayers - 1];
	float* dev_output = layerList[numLayers - 1];

	CUBLAS_CHECK_RETURN(cublasGetVector(outputDim, sizeof(float), dev_output, 1, output, 1));
	//	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

////////////////////////// TODO Read and write NOT READY.  Needs completion of related ctor
// static methods
void BPNeuralNetwork::readFile(ifstream& input, vector<BPNeuralNetwork*>& nnetList) {
	float* biasWeights = NULL;
	float* weights = NULL;

	string line;
	while(input.good()) {
		getline(input, line);
		line = line.substr(0, line.find_first_of('#'));
		if (line.empty()) continue;
		stringstream dimStream(line);
		int numLayers;
		dimStream >> numLayers;
		if (dimStream.fail()) break;
		vector<int> dimensionList;
		for (unsigned i = 0; i < numLayers; i++) {
			int dim;
			dimStream >> dim;
			if (dimStream.fail()) break;
			dimensionList.push_back(dim);
		}
		vector<float*> biasList;
		vector<float*> weightList;
		// biases and weight matrices apply to the non-input layers
		for (unsigned i = 1; i < numLayers; i++) {

			// first get the biases for each layer
			getline(input, line);
			line = line.substr(0, line.find_first_of('#'));
			if (line.empty()) continue;

			stringstream biasStream(line);

			// cant delete this pointer unless the read is unsuccessful
			// it is stored in the new objects bias weight list
			biasWeights = new float[dimensionList[i]];
			for (unsigned j = 0; j < dimensionList[i]; j++) {
				biasStream >> biasWeights[j];
				if (biasStream.fail()) {
					delete [] biasWeights;
					break;
				}
				biasList.push_back(biasWeights);
			}
			for (unsigned i = 1; i < numLayers; i++) {
				getline(input, line);
				line = line.substr(0, line.find_first_of('#'));
				if (line.empty()) continue;
				stringstream weightStream(line);

				int numWeights = dimensionList[i - 1] * dimensionList[i];
				// cant delete this pointer unless the read is unsuccessful
				// it is stored in the new objects bias weight list
				weights = new float[numWeights];
				for (unsigned j = 0; j < numWeights; j++) {
					weightStream >> weights[j];
					if (weightStream.fail()) {
						delete [] biasWeights;
						delete [] weights;
						break;
					}
					weightList.push_back(weights);
				}
			}
		}
		BPNeuralNetwork* pNNet = new BPNeuralNetwork(dimensionList, biasList, weightList);
	}
}

void BPNeuralNetwork::writeFile(vector<BPNeuralNetwork*>& nnetList, ofstream& output) {

}


} /* namespace cublas_bp_nnet */
