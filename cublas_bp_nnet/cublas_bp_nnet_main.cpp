/*
 * neuro_vision.cpp
 *
 *  Created on: Dec 26, 2014
 *      Author: dean
 */

#include <cassert>
#include <cstdlib>


#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
using namespace std;

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "simple_test_patterns.h"
#include "simple_test_weights.h"

#include "cublas_bp_nnet_utility.h"
#include "BPNeuralNetwork.h"
#include "ImageHandle.h"
using namespace cublas_bp_nnet;


#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }



void query_cuda_device(int dev, cudaDeviceProp& gpu_properties) {
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  CUDA_CHECK_RETURN(cudaGetDeviceProperties(&gpu_properties, dev));
}


int main(int argc, char* argv[]) {

	if (argc < 3) { //4) {
		cout << "*** Got " << argc - 1 << " arguments" << endl;
		cout << argv[0]
		<< " requires: (1) num epochs, (2) filename for neuron spec,"
		<< " (3) filename for input images, (4) filename for target images" << endl;
		exit(0);
	}

	vector<ImageHandle> inputImageList;
	vector<ImageHandle> targetImageList;

	vector<vector<int> > neuralNetworkSpecList;

	stringstream stream(argv[1]);
	int epochs;
	stream >> epochs;
	assert(epochs > 0);

	if (! readNeuronFile(argv[2], neuralNetworkSpecList)) {
		cout << "***** Cant read neuron file" << endl;
		exit(1);
	}

	/*
	vector<ImageHandle> inputImageHandleList;
	if (! readImageFile(argv[2], inputImageHandleList)) {
		cout << "***** Cant read image file" << endl;
		exit(1);
	}
	vector<ImageHandle> targetImageHandleList;
	if (! readImageFile(argv[3], targetImageHandleList)) {
		cout << "***** Cant read image file" << endl;
		exit(1);
	}
*/
	int numDevices;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&numDevices));
	if (numDevices == 0) {
		cout << "No CUDA devices available, Terminating" << endl;
		exit(1);
	}

	// just use the first device
	int deviceNum = 0;
	CUDA_CHECK_RETURN(cudaSetDevice(deviceNum));

	cublasHandle_t handle;
	cublasCreate(&handle);

	// query the GPU properties
	cudaDeviceProp gpu_properties;
	query_cuda_device(deviceNum, gpu_properties);

	float noiseLevel = 0.01;

	// create the neural networks from the specification list and the
	// cublas handle
	vector<BPNeuralNetwork*> neuralNetworkList;
	BPNeuralNetwork* pNet;
	for (unsigned i = 0; i < neuralNetworkSpecList.size(); i++) {
		pNet = new BPNeuralNetwork(handle, neuralNetworkSpecList[i],
				gpu_properties.maxThreadsPerBlock, noiseLevel);
		neuralNetworkList.push_back(pNet);
	}

	// test nnet on simple patterns
	test_nnet_on_simple_patterns(epochs, *pNet);
//	test_weights(*pNet);


	////////////////////////////////////////////////////////
	// Cleanup
	for (unsigned i = 0; i < neuralNetworkList.size(); i++) {
		delete neuralNetworkList[i];
	}

	cublasDestroy(handle);
	return 0;
}


