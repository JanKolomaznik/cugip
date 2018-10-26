#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/image_view.hpp>

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

#include <cugip/detail/meta_algorithm_utils.hpp>
#include <cugip/detail/for_each_host_implementation.hpp>

#if defined(__CUDACC__)
	#include <cugip/detail/for_each_device_implementation.hpp>
#endif //defined(__CUDACC__)

namespace cugip {

/** \addtogroup meta_algorithm
 * @{
 **/

template <int tDimension, typename TFunctor, bool tRunOnDevice>
void
for_each_in_region(region<tDimension> aRegion, TFunctor aOperator, BoolValue<tRunOnDevice>)
{
	//return cugip::detail::for_each_implementation(aRegion, aOperator);

	detail::ForEachPositionImplementation<
		tRunOnDevice>::run(aRegion, aOperator/*, DefaultForEachPolicy<TView>(), aCudaStream*/);

}


template<int tDimension>
struct DefaultForEachPolicy {
	static constexpr bool cPreload = false;
#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID dim3 blockSize() const
	{
		return detail::defaultBlockDimForDimension<tDimension>();
	}

	CUGIP_DECL_HYBRID dim3 gridSize(const region<tDimension> aRegion) const
	{
		return detail::defaultGridSizeForBlockDim(aRegion.size, blockSize());
	}
#endif //defined(__CUDACC__)
};


template<typename TOperator>
struct ForEachFunctor {
	template<typename TCoords, typename TView>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TView aView) {
		mOperator(aView[aToCoords]);
	}

	TOperator mOperator;
};

template<typename TOperator>
struct ForEachPositionFunctor {
	template<typename TCoords, typename TView>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TView aView) {
		mOperator(aView[aToCoords], aToCoords);
	}

	TOperator mOperator;
};

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}


	detail::UniversalRegionCoverImplementation<is_device_view<TView>::value>
		::run(ForEachFunctor<TFunctor>{aOperator}, aPolicy, aCudaStream, aView);
}

template <typename TView, typename TFunctor>
void
for_each(TView aView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	for_each(aView, aOperator, DefaultForEachPolicy<dimension<TView>::value>{}, aCudaStream);
}

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each_position(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value, "Input view must be an image view");
	if(isEmpty(aView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TView>::value>
		::run(ForEachPositionFunctor<TFunctor>{aOperator}, aPolicy, aCudaStream, aView);

}


template <typename TView, typename TFunctor>
void
for_each_position(TView aView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	for_each_position(aView, aOperator, DefaultForEachPolicy<dimension<TView>::value>{}, aCudaStream);
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
