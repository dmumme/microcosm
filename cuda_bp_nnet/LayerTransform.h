/*
 * LayerTransform.h
 *
 *  Created on: Dec 27, 2014
 *      Author: dean
 */

#ifndef LAYERTRANSFORM_H_
#define LAYERTRANSFORM_H_

namespace cublas_bp_nnet {

struct LayerTransform {

	LayerTransform(int inDim, int outDim);
	virtual ~LayerTransform();

	void operator()(float const* input, float* output);

	float** matrix;
};

} /* namespace cublas_bp_nnet */
#endif /* LAYERTRANSFORM_H_ */
