#pragma once


#include <cugip/detail/include.hpp>
#include <cugip/exception.hpp>
#include <cugip/image_locator.hpp>

namespace cugip {

namespace detail {

template <typename TInView, typename TOutView, typename TFunctor>
CUGIP_GLOBAL void 
kernel_filter(TInView aInView, TOutView aOutView, TFunctor aOperator )
{
	typename TOutView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TOutView::extents_t extents = aOutView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aOutView[coord] = aOperator(aInView.template locator<cugip::border_handling_repeat_t>(coord));
	} 
}



}//namespace detail


/** \ingroup meta_algorithm
 * @{
 **/

template <typename TInView, typename TOutView, typename TFunctor>
void 
filter(TInView aInView, TOutView aOutView, TFunctor aOperator)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aInView.dimensions().template get<0>() / blockSize.x + 1), aInView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_filter<TInView, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(aInView, aOutView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

/** 
 * @}
 **/

//*************************************************************************************************************

}//namespace cugip

