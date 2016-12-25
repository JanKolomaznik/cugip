#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>

#if defined(__CUDACC__)
#include <cugip/cuda_utils.hpp>
#endif //defined(__CUDACC__)


namespace cugip {
namespace detail {

template<bool tRunOnDevice>
struct ForEachImplementation;

template<bool tRunOnDevice>
struct ForEachPositionImplementation;

}//namespace detail
}//namespace cugip

#include <cugip/detail/for_each_host_implementation.hpp>

#if defined(__CUDACC__)
	#include <cugip/detail/for_each_device_implementation.hpp>
#endif //defined(__CUDACC__)

namespace cugip {

/** \addtogroup meta_algorithm
 * @{
 **/

template <int tDimension, typename TFunctor>
TFunctor
for_each(region<tDimension> aRegion, TFunctor aOperator)
{
	return cugip::detail::for_each_implementation(aRegion, aOperator);
}


template<typename TView>
struct DefaultForEachPolicy {
#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID dim3 blockSize() const
	{
		return detail::defaultBlockDimForDimension<dimension<TView>::value>();
	}

	CUGIP_DECL_HYBRID dim3 gridSize(const TView &aView) const
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
#endif //defined(__CUDACC__)
};

template <typename TView, typename TFunctor>
void
for_each(TView aView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}

	detail::ForEachImplementation<
		is_device_view<TView>::value>::run(aView, aOperator, DefaultForEachPolicy<TView>(), aCudaStream);
}

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}

	detail::ForEachImplementation<
		is_device_view<TView>::value>::run(aView, aOperator, aPolicy, aCudaStream);
}

template <typename TView, typename TFunctor>
void
for_each_position(TView aView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}

	detail::ForEachPositionImplementation<
		is_device_view<TView>::value>::run(aView, aOperator, DefaultForEachPolicy<TView>(), aCudaStream);
}

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each_position(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}

	detail::ForEachPositionImplementation<
		is_device_view<TView>::value>::run(aView, aOperator, aPolicy, aCudaStream);
}

/**
 * @}
 **/

#if 0

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
#endif

}//namespace cugip
