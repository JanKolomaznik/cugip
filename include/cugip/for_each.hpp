#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/cuda_utils.hpp>

namespace cugip {

namespace detail {

template<int tDimension>
dim3 defaultBlockDimForDimension();

template<>
inline dim3 defaultBlockDimForDimension<2>() { return dim3(32, 16, 1); }

template<>
inline dim3 defaultBlockDimForDimension<3>() { return dim3(32, 4, 4); }

template<int tDimension>
dim3 defaultGridSizeForBlockDim(cugip::simple_vector<int, tDimension> aViewDimensions, dim3 aBlockSize);

template<>
inline dim3
defaultGridSizeForBlockDim<2>(cugip::simple_vector<int, 2> aViewDimensions, dim3 aBlockSize)
{
	return dim3(
		(aViewDimensions[0] - 1) / aBlockSize.x + 1,
		(aViewDimensions[1] - 1) / aBlockSize.y + 1,
		1);
}

template<>
inline dim3
defaultGridSizeForBlockDim<3>(cugip::simple_vector<int, 3> aViewDimensions, dim3 aBlockSize)
{
	return dim3(
		(aViewDimensions[0] - 1) / aBlockSize.x + 1,
		(aViewDimensions[1] - 1) / aBlockSize.y + 1,
		(aViewDimensions[2] - 1) / aBlockSize.z + 1);
}


template <typename TView, typename TFunctor>
CUGIP_GLOBAL void
kernel_for_each(TView aView, TFunctor aOperator )
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
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
	dim3 blockSize = detail::defaultBlockDimForDimension<dimension<TView>::value>();
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize);

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
	typename TView1::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView1>::value>();
	typename TView1::extents_t extents = aView1.dimensions();

	if (coord < extents) {
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

	dim3 blockSize = detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aView1.dimensions(), blockSize);

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
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aView[coord] = aOperator(aView[coord], coord);
	}
}

} // namespace detail

/** \ingroup meta_algorithm
 * @{
 **/
template <typename TView, typename TFunctor>
void
for_each_position(TView aView, TFunctor aOperator)
{
	dim3 blockSize = detail::defaultBlockDimForDimension<dimension<TView>::value>();
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize);

	D_PRINT("Executing kernel: for_each_position(); blockSize = "
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

//*************************************************************************************************************

namespace detail {

template <typename TView, typename TFunctor>
CUGIP_GLOBAL void
kernel_for_each_locator(TView aView, TFunctor aOperator)
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aOperator(aView.template locator<cugip::border_handling_repeat_t>(coord));
	}
}

} // namespace detail

/** \ingroup meta_algorithm
 * @{
 **/
template <typename TView, typename TFunctor>
void
for_each_locator(TView aView, TFunctor aOperator)
{
	dim3 blockSize = detail::defaultBlockDimForDimension<dimension<TView>::value>();
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize);

	D_PRINT("Executing kernel: for_each_locator(); blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_for_each_locator<TView, TFunctor>
		<<<gridSize, blockSize>>>(aView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each_locator");
}

/**
 * @}
 **/


//*************************************************************************************************************

namespace detail {

template <typename TView1, typename TView2, typename TFunctor>
CUGIP_GLOBAL void
kernel_for_each_locator(TView1 aView1, TView2 aView2, TFunctor aOperator)
{
	typename TView1::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView1>::value>();
	typename TView1::extents_t extents = aView1.dimensions();

	if (coord < extents) {
		aOperator(
			aView1.template locator<cugip::border_handling_repeat_t>(coord),
			aView2.template locator<cugip::border_handling_repeat_t>(coord)
			);
	}
}

} // namespace detail

/** \ingroup meta_algorithm
 * @{
 **/
template <typename TView1, typename TView2, typename TFunctor>
void
for_each_locator(TView1 aView1, TView2 aView2, TFunctor aOperator)
{
	CUGIP_ASSERT(aView1.dimensions() == aView2.dimensions());

	dim3 blockSize = detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aView1.dimensions(), blockSize);

	D_PRINT("Executing kernel: for_each_locator(); blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_for_each_locator<TView1, TView2, TFunctor>
		<<<gridSize, blockSize>>>(aView1, aView2, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each_locator");
}

/**
 * @}
 **/


}//namespace cugip
