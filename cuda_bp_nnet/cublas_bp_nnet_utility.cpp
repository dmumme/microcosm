/*
 * utility.cpp
 *
 *  Created on: Dec 28, 2014
 *      Author: dean
 */
#include "cublas_bp_nnet_utility.h"

#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

#include <cmath>


#include "ImageHandle.h"
#include <cuda.h>
#include <cuda_runtime.h>


namespace cublas_bp_nnet {

void showVector(int dim, float* v) {
	for (unsigned j = 0; j < dim; j++) {
		float value = fabs(v[j]) < 1e-2 ? 0 : v[j];
		cout << "  :" << value;
	}
}

void showMatrix(int rows, int cols, float* matrix) {
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			cout.precision(5);
			float value = fabs(matrix[IDX2C(i,j,rows)]) < 1e-2 ? 0 : matrix[IDX2C(i,j,rows)];
			cout << "  :" << value;
		}
		cout << "::";
	}
}


void sequence_fill(int start, int step, vector<int>& v) {
	for (unsigned i = 0; i < v.size(); i++) {
		v[i] = start + i * step;
	}
}

void in_place_shuffle(vector<int>& v) {
	for (unsigned i = 0; i < v.size(); i++) {
		int k = v.size() * (rand() / ((double)RAND_MAX));
		float temp = v[k];
		v[k] = v[i];
		v[i] = temp;
	}
}

// read the specified file in the specified directory
// to get a list of filenames.  Test that each can be opened
// before adding it to the list
void readFilenameList(string const& directory,
		string const& filename,
		vector<string>& filenameList) {

	ifstream input((directory + filename).c_str());
	if (! input.good()) {
		cout << "Could not open: " << filename << " for reading" << endl;
		exit(1);
	}

	string line;
	while (input.good()) {
		getline(input, line);
		line = line.substr(0, line.find_first_of('#'));
		if (line.empty()) continue;

		ifstream test_open((directory + line).c_str());
		if (! test_open) continue;

		test_open.close();
		filenameList.push_back(line);
	}
}


// don't want to instantiate neural networks here. They need the
// context handle which we don't have
bool readNeuronFile(char const* filename, vector<vector<int> >& specList)
{
	ifstream neurnonSpecStream(filename);
	if (! neurnonSpecStream.good()) return false;

	string line;
	while(neurnonSpecStream.good()) {

		getline(neurnonSpecStream, line);
		line = line.substr(0, line.find_first_of("#"));
		if (line.empty()) continue;

		stringstream stream(line);
		int numLayers;
		stream >> numLayers;
		if (stream.fail()) {
			return false;
		}

		vector<int> layerDimensions;
		for (int i = 0; i < numLayers; i++) {
			int dim;
			stream >> dim;
			if (stream.fail()) {
				return false;
			}
			layerDimensions.push_back(dim);
		}
		specList.push_back(layerDimensions);
		//	cublasStatus_t cublasStatus = cublasCreate(*pHandle);
		//		neuralNetworks.push_back(new BPNeuralNetwork(layerDimensions));
	}
	return specList.size() > 0;
}



bool readImageFile(char const* filename, vector<ImageHandle>& imageHandleList) {
	bool status;

	return status;
}

} /* cublas_bp_nnet */
