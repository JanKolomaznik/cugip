#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>

namespace cugip {

namespace detail {

template <typename TInView, typename TOutView, typename TFunctor>
__global__ void 
kernel_transform(TInView aInView, TOutView aOutView, TFunctor aOperator )
{
	typename TOutView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TOutView::extents_t extents = aInView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aOutView[coord] = aOperator(aInView[coord]);
	} 
}



}//namespace detail


template <typename TInView, typename TOutView, typename TFunctor>
void 
transform(TInView aInView, TOutView aOutView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aInView.dimensions().template get<0>() / blockSize.x + 1), aInView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_transform<TInView, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(aInView, aOutView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

//*************************************************************************************************************

namespace detail {

template <typename TInView, typename TOutView, typename TFunctor>
CUGIP_GLOBAL void 
kernel_transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator )
{
	typename TOutView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TOutView::extents_t extents = aInView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aOutView[coord] = aOperator(aInView[coord], coord);
	} 
}



}//namespace detail


template <typename TInView, typename TOutView, typename TFunctor>
void 
transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aInView.dimensions().template get<0>() / blockSize.x + 1), aInView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_transform_position<TInView, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(aInView, aOutView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

}//namespace cugip


