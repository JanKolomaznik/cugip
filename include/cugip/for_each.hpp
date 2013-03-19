#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>

namespace cugip {

namespace detail {

template <typename TView, typename TFunctor>
CUGIP_GLOBAL void 
kernel_for_each(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aOperator(aView[coord]);
	} 
}



}//namespace detail

/** \ingroup meta_algorithm
 * @{
 **/

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

/** 
 * @}
 **/


//*************************************************************************************************************
namespace detail {

template <typename TView1, typename TView2, typename TFunctor>
CUGIP_GLOBAL void 
kernel_for_each(TView1 aView1, TView2 aView2, TFunctor aOperator )
{
	typename TView1::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView1::extents_t extents = aView1.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aOperator(aView1[coord], aView2[coord]);
	} 
}



}//namespace detail

/** \ingroup meta_algorithm
 * @{
 **/

template <typename TView1, typename TView2, typename TFunctor>
void 
for_each(TView1 aView1, TView2 aView2, TFunctor aOperator)
{
	CUGIP_ASSERT(aView1.dimensions() == aView2.dimensions());

	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aView1.dimensions().template get<0>() / blockSize.x + 1), aView1.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_for_each<TView1, TView2, TFunctor>
		<<<gridSize, blockSize>>>(aView1, aView2, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

/** 
 * @}
 **/


//*************************************************************************************************************

namespace detail {

template <typename TView, typename TFunctor>
CUGIP_GLOBAL void 
kernel_for_each_position(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TView::extents_t extents = aView.dimensions();

	if (coord.template get<0>() < extents.template get<0>() && coord.template get<1>() < extents.template get<1>()) {
		aView[coord] = aOperator(aView[coord], coord);
	} 
}

}

/** \ingroup meta_algorithm
 * @{
 **/
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

/** 
 * @}
 **/


}//namespace cugip

