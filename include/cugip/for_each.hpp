#pragma once

#include <cugip/detail/include.hpp>

namespace cugip {

namespace detail {

template <typename TView, typename TFunctor>
__global__ void 
kernel_for_each(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.size();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aOperator(aView[coord]);
	} 
	if (threadIdx.x % 4) {
		printf("Y");
	}

}



}//namespace detail


template <typename TView, typename TFunctor>
void 
for_each(TView aView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	//dim3 gridSize((aView.size().template get<0>() / blockSize.x + 1), aView.size().template get<1>(), 1);
	dim3 gridSize(4, 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );       
	detail::kernel_for_each<TView, TFunctor>
		<<<gridSize, blockSize>>>(aView, aOperator);
	cudaThreadSynchronize();
}

}//namespace cugip

