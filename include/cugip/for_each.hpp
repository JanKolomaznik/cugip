#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/image_view.hpp>
#include <cugip/view_utils.hpp>

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
	using BorderHandling = BorderHandlingTraits<border_handling_enum::REPEAT>;

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
	CUGIP_HD_WARNING_DISABLE
	template<typename TCoords, typename TView>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TView aView) {
		mOperator(aView[aToCoords]);
	}

	TOperator mOperator;
};

template<typename TOperator>
struct ForEachPositionFunctor {
	CUGIP_HD_WARNING_DISABLE
	template<typename TCoords, typename TView>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TView aView) {
		mOperator(aView[aToCoords], aToCoords);
	}

	TOperator mOperator;
};

template<typename TOperator, typename TBorderHandling>
struct ForEachLocatorFunctor {
	CUGIP_HD_WARNING_DISABLE
	template<typename TCoords, typename TView>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TView aView) {
		mOperator(create_locator<TView, TBorderHandling>(aView, aToCoords));
	}

	TOperator mOperator;
};

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value || is_array_view<TView>::value, "Input view must be an image or array view");
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
	static_assert(is_image_view<TView>::value || is_array_view<TView>::value, "Input view must be an image or array view");
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

template <typename TView, typename TFunctor, typename TPolicy>
void
for_each_locator(TView aView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TView>::value || is_array_view<TView>::value, "Input view must be an image or array view");
	if(isEmpty(aView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TView>::value>
		::run(ForEachLocatorFunctor<TFunctor, typename TPolicy::BorderHandling>{aOperator}, aPolicy, aCudaStream, aView);

}


template <typename TView, typename TFunctor>
void
for_each_locator(TView aView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	for_each_locator(aView, aOperator, DefaultForEachPolicy<dimension<TView>::value>{}, aCudaStream);
}

/**
 * @}
 **/

}//namespace cugip
