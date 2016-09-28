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

template<typename TView1, typename TView2>
struct DefaultTransformPolicy {
#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID dim3 blockSize() const
	{
		return detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	}

	CUGIP_DECL_HYBRID dim3 gridSize(const TView1 &aView) const
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
#endif //defined(__CUDACC__)
};



/** \ingroup meta_algorithm
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

	detail::TransformImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, transform_update_assign(), aPolicy, aCudaStream);
}

template <typename TInView, typename TOutView, typename TFunctor>
void
transform(TInView aInView, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform(aInView, aOutView, aOperator, DefaultTransformPolicy<TInView, TOutView>(), aCudaStream);
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

	detail::TransformImplementation<
		is_device_view<TInView1>::value && is_device_view<TInView2>::value && is_device_view<TOutView>::value>::run(aInView1, aInView2, aOutView, aOperator, transform_update_assign(), aPolicy, aCudaStream);
}

template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor>
void
transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform2(aInView1, aInView2, aOutView, aOperator, DefaultTransformPolicy<TInView1, TOutView>(), aCudaStream);
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

	detail::TransformPositionImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, transform_update_assign(), aPolicy, aCudaStream);
}

template <typename TInView, typename TOutView, typename TFunctor>
void
transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform_position(aInView, aOutView, aOperator, DefaultTransformPolicy<TInView, TOutView>(), aCudaStream);
}

//TODO - assign versions of the transform alg.

//*************************************************************************************************************
template<typename TView1, typename TView2>
struct DefaultTransformLocatorPolicy {
	typedef BorderHandlingTraits<border_handling_enum::REPEAT>/*border_handling_repeat_t*/ BorderHandling;

	static constexpr bool cPreload = false;

#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID dim3 blockSize() const
	{
		return detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	}

	CUGIP_DECL_HYBRID dim3 gridSize(const TView1 &aView) const
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
#endif //defined(__CUDACC__)
};

template<typename TView, int tRadius>
struct PreloadingTransformLocatorPolicy {
	typedef BorderHandlingTraits<border_handling_enum::REPEAT>/*border_handling_repeat_t*/ BorderHandling;
	//typedef border_handling_repeat_t BorderHandling;
	static constexpr int cDimension = dimension<TView>::value;
	//typedef detail::defaultBlockSize<cDimension>::type BlockSize;
	//typedef detail::defaultBlockSize<cDimension>::type PreloadedBlockSize;

	typedef StaticSize<32+2*tRadius, 4+2*tRadius, 4+2*tRadius> RegionSize;

	static constexpr bool cPreload = true;

#if defined(__CUDACC__)
	CUGIP_DECL_HYBRID region<cDimension> regionForBlock()
	{
		return region<cDimension>{
			simple_vector<int, cDimension>(-tRadius, FillFlag()),
			dim3_to_vector<cDimension>(blockSize()) + simple_vector<int, cDimension>(2*tRadius, FillFlag())
			};
	}

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

template <typename TInView, typename TOutView, typename TOperator>
void
transform_locator(TInView aInView, TOutView aOutView, TOperator aOperator, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::TransformLocatorImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, transform_update_assign(), DefaultTransformLocatorPolicy<TInView, TOutView>(), aCudaStream);
}


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

	detail::TransformLocatorImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, transform_update_assign(), aPolicy, aCudaStream);

}

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation>
void
transform_locator_assign(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, cudaStream_t aCudaStream = 0)
{
	static_assert(is_image_view<TInView>::value, "Input view must be image view");
	static_assert(is_image_view<TOutView>::value, "Output view must be image view");
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	if(isEmpty(aInView)) {
		return;
	}

	detail::TransformLocatorImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, aAssignOperation, DefaultTransformLocatorPolicy<TInView, TOutView>(), aCudaStream);
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

	detail::TransformLocatorImplementation<
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, aAssignOperation, aPolicy, aCudaStream);
}
/**
 * @}
 **/


}//namespace cugip
