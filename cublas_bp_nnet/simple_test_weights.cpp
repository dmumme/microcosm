/*
 * simple_test_weights.cpp
 *
 *  Created on: Jan 2, 2015
 *      Author: dean
 */

#include <iostream>
using namespace std;

#include "cublas_v2.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

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





#include <vector>
using namespace std;

#include "BPNeuralNetwork.h"
#include "simple_test_patterns.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


namespace cublas_bp_nnet {

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

static float four_dim_test_inputs[][NUM_FOUR_PATTERNS] = {
		{ 0, 0, 0, 0, }, // 0
		{ 1, 0, 0, 0, },
		{ 0, 1, 0, 0, },
		{ 0, 0, 1, 0, },
		{ 0, 0, 0, 1, }, // 4
		{ 1, 1, 0, 0, },
		{ 0, 1, 1, 0, },
		{ 0, 0, 1, 1, },
		{ 1, 0, 0, 1, }, // 8
		{ 1, 0, 1, 0, },
		{ 0, 1, 0, 1, },
		{ 0, 1, 1, 1, },
		{ 1, 0, 1, 1, }, // 12
		{ 1, 1, 0, 1, },
		{ 1, 1, 1, 0, },
		{ 1, 1, 1, 1, },
};

static float input_weight_matrices[][8] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0, },
		{ 1, 0, 0, 0, 0, 0, 0, 0, },
		{ 0, 1, 0, 0, 0, 0, 0, 0, },
		{ 0, 0, 1, 0, 0, 0, 0, 0, },
		{ 0, 0, 0, 1, 0, 0, 0, 0, },
		{ 0, 0, 0, 0, 1, 0, 0, 0, },
		{ 0, 0, 0, 0, 0, 1, 0, 0, },
		{ 0, 0, 0, 0, 0, 0, 1, 0, },
		{ 0, 0, 0, 0, 0, 0, 0, 1, },
};

static float hidden_weight_matrices[][8] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0, },
		{ 1, 0, 0, 0, 0, 0, 0, 0, },
		{ 0, 1, 0, 0, 0, 0, 0, 0, },
		{ 0, 0, 1, 0, 0, 0, 0, 0, },
		{ 0, 0, 0, 1, 0, 0, 0, 0, },
		{ 0, 0, 0, 0, 1, 0, 0, 0, },
		{ 0, 0, 0, 0, 0, 1, 0, 0, },
		{ 0, 0, 0, 0, 0, 0, 1, 0, },
		{ 0, 0, 0, 0, 0, 0, 0, 1, },
};

static float hidden_biases[][2] = {
		{ 0, 0 },
		{ 1, 0 },
		{ 0, 1 },
		{ 1, 1 },
};

static float output_biases[][4] = {
		{ 0, 0, 0, 0, },
		{ 1, 0, 0, 0, },
		{ 0, 1, 0, 0, },
		{ 0, 0, 1, 0, },
		{ 0, 0, 0, 1, },
		{ 1, 1, 0, 0, },
		{ 0, 1, 1, 0, },
		{ 0, 0, 1, 1, },
		{ 1, 0, 0, 1, },
		{ 1, 0, 1, 0, },
		{ 0, 1, 1, 1, },
		{ 1, 0, 1, 1, },
		{ 1, 1, 0, 1, },
		{ 1, 1, 1, 0, },
		{ 1, 1, 1, 1, },
};

static void showVector(int dim, float* v) {
	for (unsigned j = 0; j < dim; j++) {
		float value = fabs(v[j]) < 1e-2 ? 0 : v[j];
		cout << "  " << value;
	}
}

static void showMatrix(int rows, int cols, float* matrix) {
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			cout.precision(3);
			float value = fabs(matrix[IDX2C(i,j,rows)]) < 1e-2 ? 0 : matrix[IDX2C(i,j,rows)];
			cout << "  " << value;
		}
		cout << "::";
	}
}

#if 0
// this assumes a 3 layer network
static void twiddleWeights(BPNeuralNetwork& nnet) {
	int inputDim = nnet.layerDimensionList[0];
	int hiddenDim = nnet.layerDimensionList[1];
	int outputDim = nnet.layerDimensionList[2];

	float* dev_input_layer = nnet.layerList[0];
	float* dev_hidden_layer = nnet.layerList[1];
	float* dev_output_layer = nnet.layerList[2];

	float* dev_inputMatrix = nnet.matrixList[0];
	float* dev_hiddenMatrix = nnet.matrixList[1];

	float* dev_hiddenBiases = nnet.biasWeightList[0];
	float* dev_outputBiases = nnet.biasWeightList[1];

	float biasVector1[] = { 0, 0, };
	CUBLAS_CHECK_RETURN(cublasSetVector(hiddenDim, sizeof(*biasVector1), biasVector1, 1, dev_hiddenBiases, 1));

	float biasVector2[] = { 0, 0, 1, 0, };
	CUBLAS_CHECK_RETURN(cublasSetVector(outputDim, sizeof(*biasVector2), biasVector2, 1, dev_outputBiases, 1));

	for (unsigned inMatrixId = 0; inMatrixId < 9; inMatrixId++) {
		float* input_matrix = input_weight_matrices[inMatrixId];

		CUBLAS_CHECK_RETURN(cublasSetVector(inputDim * hiddenDim, sizeof(*input_matrix), input_matrix, 1, dev_inputMatrix, 1));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		for (unsigned hidMatrixId = 0; hidMatrixId < 9; hidMatrixId++) {
			cout << "!!!!!!!!!!!!!!!!! ";
			showMatrix(2, 4, input_matrix);
			cout << "::::";

			float* hidden_matrix = hidden_weight_matrices[hidMatrixId];
			showMatrix(4, 2, hidden_matrix);
			cout << endl;

			CUBLAS_CHECK_RETURN(cublasSetVector(hiddenDim * outputDim, sizeof(*hidden_matrix), hidden_matrix, 1, dev_hiddenMatrix, 1));
//			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			for (unsigned i = 0; i < 16; i++) {
				float* input_vector = four_dim_test_inputs[i];

				cout << "  i:";
				showVector(inputDim, four_dim_test_inputs[i]);
				cout << endl;

				/////////// compute the output
				nnet.operator ()(input_vector);

				float* hidden_vector = new float[hiddenDim];
				CUBLAS_CHECK_RETURN(cublasGetVector(hiddenDim, sizeof(*dev_output_layer), dev_hidden_layer, 1, hidden_vector, 1));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());

				float* output_vector = new float[outputDim];
				CUBLAS_CHECK_RETURN(cublasGetVector(outputDim, sizeof(*dev_output_layer), dev_output_layer, 1, output_vector, 1));
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());

				cout << "  i:";
				showVector(inputDim, four_dim_test_inputs[i]);
				cout << "  h:";
				showVector(hiddenDim, hidden_vector);
				cout << "  o:";
				delete [] hidden_vector;
				showVector(outputDim, output_vector);
				cout << "     !!!!! ";
				showMatrix(2, 4, input_matrix);
				cout << "::::";
				float* hidden_matrix = hidden_weight_matrices[hidMatrixId];
				showMatrix(4, 2, hidden_matrix);
				cout << endl << "////////////////////////////////////////////////////////////" << endl;
				cout << endl;

				delete [] output_vector;
			}
			cout << endl;
		}
		cout << endl;
	}
}
#endif

void test_weights(BPNeuralNetwork& nnet) {
	vector<float*> inputList;

	for (unsigned i = 0; i < NUM_FOUR_PATTERNS; i++) {
		inputList.push_back(four_dim_input_patterns[i]);
	}

	vector<float*> targetList;
	for (unsigned i = 0; i < NUM_FOUR_PATTERNS; i++) {
		targetList.push_back(four_dim_target_patterns[i]);
	}

	float mrmse;

//	twiddleWeights(nnet);
}

void test_matrix_vector_multiplication() {
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
CUBLAS_CHECK_RETURN(cublasCreate(&handle));

// query the GPU properties
//cudaDeviceProp gpu_properties;
//query_cuda_device(deviceNum, gpu_properties);

int const inputDim = 4;
int const outputDim = 2;

float output[outputDim];

float input[] = { 1, 10, 100, 1000, };
float* dev_input;
float* dev_output;

CUDA_CHECK_RETURN(cudaMalloc(&dev_input, inputDim * sizeof(float)));
CUBLAS_CHECK_RETURN(cublasSetVector(inputDim, sizeof(float), input, 1, dev_input, 1));

CUDA_CHECK_RETURN(cudaMalloc(&dev_output, 4 * sizeof(float)));

float matrix[] = { 1, 0, 1, 1, 2, 0, 0, 1, };
int const matrixDim = inputDim * outputDim;


float* dev_matrix;
CUDA_CHECK_RETURN(cudaMalloc(&dev_matrix, matrixDim * sizeof(float)));
CUBLAS_CHECK_RETURN(cublasSetVector(matrixDim, sizeof(float), matrix, 1, dev_matrix, 1));

float alpha = 1.0;
float beta  = 0.0;

CUBLAS_CHECK_RETURN(cublasSgemv(handle,
		CUBLAS_OP_N,
		outputDim,
		inputDim,
		&alpha,
		dev_matrix,
		outputDim,// leading dim of matrix
		dev_input,
		1,
		&beta,
		dev_output,
		1));
//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
CUBLAS_CHECK_RETURN(cublasGetVector(outputDim, sizeof(float), dev_output, 1, output, 1));

cout << "m:";
showMatrix(outputDim, inputDim, matrix);
cout << "  i:";
showVector(inputDim, input);
cout << "  o:";
showVector(outputDim, output);


CUBLAS_CHECK_RETURN(cublasDestroy(handle));
}

} /* cublas_bp_nnet */
