#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>

namespace cugip {

namespace detail {

template <typename TView, typename TType>
CUGIP_GLOBAL void 
kernel_fill(TView aView, TType aValue)
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aValue;
	} 
}

}//namespace detail


template <typename TView, typename TType>
void 
fill(TView aView, TType aValue)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aView.dimensions().template get<0>() / blockSize.x + 1), aView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_fill<TView, TType>
		<<<gridSize, blockSize>>>(aView, aValue);
	CUGIP_CHECK_ERROR_STATE("kernel_fill");
}

}//namespace cugip

