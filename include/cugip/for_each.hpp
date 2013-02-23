#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>

namespace cugip {

namespace detail {

template <typename TView, typename TFunctor>
__global__ void 
kernel_for_each(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aOperator(aView[coord]);
	} 
}



}//namespace detail


template <typename TView, typename TFunctor>
void 
for_each(TView aView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aView.dimensions().template get<0>() / blockSize.x + 1), aView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_for_each<TView, TFunctor>
		<<<gridSize, blockSize>>>(aView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

//*************************************************************************************************************

namespace detail {

template <typename TView, typename TFunctor>
__global__ void 
kernel_for_each_position(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aOperator(aView[coord], coord);
	} 
}

}

template <typename TView, typename TFunctor>
void 
for_each_position(TView aView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aView.dimensions().template get<0>() / blockSize.x + 1), aView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_for_each_position<TView, TFunctor>
		<<<gridSize, blockSize>>>(aView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

}//namespace cugip

