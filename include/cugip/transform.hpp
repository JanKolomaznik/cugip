#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/access_utils.hpp>

#if defined(__CUDACC__)
#include <cugip/cuda_utils.hpp>
#include <cugip/detail/shared_memory.hpp>
#endif //defined(__CUDACC__)

namespace cugip {
namespace detail {

template<bool tRunOnDevice>
struct TransformImplementation;

template<bool tRunOnDevice>
struct TransformPositionImplementation;

template<bool tRunOnDevice>
struct TransformLocatorImplementation;

} // namespace detail
} // namespace cugip

#include <cugip/detail/meta_algorithm_utils.hpp>
#include <cugip/detail/transform_host_implementation.hpp>

#if defined(__CUDACC__)
	#include <cugip/detail/transform_device_implementation.hpp>
#endif //defined(__CUDACC__)


namespace cugip {

struct transform_update_assign {
	template<typename TOutput, typename TValue>
	CUGIP_DECL_HYBRID
	void operator()(TOutput &aOutput, const TValue &aValue) const {
		aOutput = aValue;
	}
};

struct transform_update_add {
	template<typename TOutput, typename TValue>
	CUGIP_DECL_HYBRID
	void operator()(TOutput &aOutput, const TValue &aValue) const {
		aOutput += aValue;
	}
};

template<int tDimension>
struct DefaultTransformPolicy {
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


template<typename TOperator, typename TAssignOperator>
struct TransformFunctor {
	template<typename TCoords, typename TOutView, typename... TInViews>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TOutView aOutView, TInViews... aInViews) {
		static_assert(is_image_view<TOutView>::value, "Output view must be image view");
		mAssignOperator(aOutView[aToCoords], mOperator(aInViews[aToCoords]...));
	}

	TOperator mOperator;
	TAssignOperator mAssignOperator;
};

template<typename TOperator, typename TAssignOperator>
struct TransformPositionFunctor {
	template<typename TCoords, typename TOutView, typename... TInViews>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TOutView aOutView, TInViews... aInViews) {
		mAssignOperator(aOutView[aToCoords], mOperator(aInViews[aToCoords]..., aToCoords));
	}

	TOperator mOperator;
	TAssignOperator mAssignOperator;
};

template<typename TOperator, typename TAssignOperator, typename TBorderHandling>
struct TransformLocatorFunctor {
	template<typename TCoords, typename TOutView, typename... TInViews>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aToCoords, TOutView aOutView, TInViews... aInViews) {
		//TODO - check the locator - for shared memory we should use the basic one
		mAssignOperator(aOutView[aToCoords], mOperator(create_locator<TInViews, TBorderHandling>(aInViews, aToCoords)...));
	}

	template<typename TCoords, typename TPreloadedView, typename TOutView, typename... TInViews>
	CUGIP_DECL_HYBRID
	void operator()(TCoords aPreloadedCoords, TPreloadedView aPreloadedView, TCoords aGlobalCoords, TOutView aOutView, TInViews... aInViews) {
		//TODO - check the locator - for shared memory we should use the basic one
		mAssignOperator(aOutView[aGlobalCoords], mOperator(create_locator<TPreloadedView, TBorderHandling>(aPreloadedView, aPreloadedCoords)));
	}

	TOperator mOperator;
	TAssignOperator mAssignOperator;
};

/** \addtogroup meta_algorithm
 * @{
 **/

template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transform(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TInView>::value && is_device_view<TOutView>::value>
		::run(TransformFunctor<TFunctor, transform_update_assign>{aOperator, transform_update_assign{}}, aPolicy, aCudaStream, aOutView, aInView);

}

template <typename TInView, typename TOutView, typename TFunctor>
void
transform(TInView aInView, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform(aInView, aOutView, aOperator, DefaultTransformPolicy<dimension<TInView>::value>{}, aCudaStream);
}

template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TPolicy>
void
transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView1>::value, "Input view 1 must be image view");
	static_assert(is_image_view<TInView2>::value, "Input view 2 must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView1.dimensions() == aOutView.dimensions());
	CUGIP_ASSERT(aInView2.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView1)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TInView1>::value && is_device_view<TInView2>::value && is_device_view<TOutView>::value>
		::run(TransformFunctor<TFunctor, transform_update_assign>{aOperator, transform_update_assign{}}, aPolicy, aCudaStream, aOutView, aInView1, aInView2);
}

template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor>
void
transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform2(aInView1, aInView2, aOutView, aOperator, DefaultTransformPolicy<dimension<TInView1>::value>{}, aCudaStream);
}

template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TInView>::value && is_device_view<TOutView>::value>
		::run(TransformPositionFunctor<TFunctor, transform_update_assign>{aOperator, transform_update_assign{}}, aPolicy, aCudaStream, aOutView, aInView);
}

template <typename TInView, typename TOutView, typename TFunctor>
void
transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform_position(aInView, aOutView, aOperator, DefaultTransformPolicy<dimension<TInView>::value>{}, aCudaStream);
}

//TODO - assign versions of the transform alg.

//*************************************************************************************************************

template<int tDimension>
struct DefaultTransformLocatorPolicy : DefaultTransformPolicy<tDimension> {
	typedef BorderHandlingTraits<border_handling_enum::REPEAT>/*border_handling_repeat_t*/ BorderHandling;

	static constexpr bool cPreload = false;
};

template<int tDimension, int tRadius>
struct PreloadingTransformLocatorPolicy : DefaultTransformPolicy<tDimension> {
	typedef BorderHandlingTraits<border_handling_enum::REPEAT>/*border_handling_repeat_t*/ BorderHandling;
	//typedef border_handling_repeat_t BorderHandling;
	static constexpr int cDimension = tDimension;
	//typedef detail::defaultBlockSize<cDimension>::type BlockSize;
	//typedef detail::defaultBlockSize<cDimension>::type PreloadedBlockSize;

	typedef StaticSize<32+2*tRadius, 4+2*tRadius, 4+2*tRadius> RegionSize;
	//static constexpr vect3i_t cRegionSize(32+2*tRadius, 4+2*tRadius, 4+2*tRadius);

	static constexpr bool cPreload = true;

#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID region<cDimension> regionForBlock() const
	{
		return region<cDimension>{
			simple_vector<int, cDimension>(-tRadius, FillFlag()),
			dim3_to_vector<cDimension>(this->blockSize()) + simple_vector<int, cDimension>(2*tRadius, FillFlag())
			};
	}

	CUGIP_DECL_HYBRID
	vect3i_t corner1()
	{
       		return vect3i_t(-tRadius, FillFlag{});
	}
#endif //defined(__CUDACC__)

	template<typename TOutView, typename TFirstView, typename... TViews>
	CUGIP_DECL_HYBRID TFirstView GetViewForLoading(TOutView aOutView, TFirstView aFirstView, TViews... aViews) {
		return aFirstView;
	}
};

template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
void
transform_locator(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TInView>::value && is_device_view<TOutView>::value>
		::run(TransformLocatorFunctor<TOperator, transform_update_assign, typename TPolicy::BorderHandling>{aOperator, transform_update_assign{}}, aPolicy, aCudaStream, aOutView, aInView);

}


template <typename TInView, typename TOutView, typename TOperator>
void
transform_locator(TInView aInView, TOutView aOutView, TOperator aOperator, cudaStream_t aCudaStream = 0)
{
	transform_locator(aInView, aOutView, aOperator, DefaultTransformLocatorPolicy<dimension<TInView>::value>{}, aCudaStream);
}

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
void
transform_locator_assign(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream = 0)
{

	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::UniversalRegionCoverImplementation<is_device_view<TInView>::value && is_device_view<TOutView>::value>
		::run(TransformLocatorFunctor<TOperator, transform_update_assign, typename TPolicy::BorderHandling>{aOperator, aAssignOperation}, aPolicy, aCudaStream, aOutView, aInView);
}

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation>
void
transform_locator_assign(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, cudaStream_t aCudaStream = 0)
{
	transform_locator_assign(aInView, aOutView, aOperator, aAssignOperation, DefaultTransformLocatorPolicy<dimension<TInView>::value>{}, aCudaStream);
}


/**
 * @}
 **/


}//namespace cugip
