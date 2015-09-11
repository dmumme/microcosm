/*
 * utility.h
 *
 *  Created on: Dec 28, 2014
 *      Author: dean
 */

#ifndef NEURO_VISION_UTILITY_H_
#define NEURO_VISION_UTILITY_H_

#include <cstdlib>

#include <string>
#include <vector>
using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


namespace cublas_bp_nnet {

class ImageHandle;

void sequence_fill(int start, int step, vector<int>& v);
void in_place_shuffle(vector<int>& v);

void showVector(int dim, float* v);
void showMatrix(int rows, int cols, float* matrix);


void readFilenameList(string const& directory,
		string const& filename,
		vector<string>& filenameList);
bool readNeuronFile(char const* filename,  vector<vector<int> >& specList);
bool readImageFile(char const* filename, vector<ImageHandle>& imageHandleList);

} /* cublas_bp_nnet */


#endif /* NEURO_VISION_UTILITY_H_ */
