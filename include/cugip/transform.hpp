#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/cuda_utils.hpp>
#include <cugip/access_utils.hpp>

namespace cugip {

template<typename TView1, typename TView2>
struct DefaultTransformPolicy {
	static CUGIP_DECL_HYBRID dim3 blockSize()
	{
		return detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	}

	static CUGIP_DECL_HYBRID dim3 gridSize(const TView1 &aView)
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
};

namespace detail {

template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy)
{
	auto coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	auto extents = aInView.dimensions();

	if (coord < extents) {
		aOutView[coord] = aOperator(aInView[coord]);
	}
}

template <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy)
{
	auto coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView1>::value>();
	auto extents = aInView1.dimensions();

	if (coord < extents) {
		aOutView[coord] = aOperator(aInView1[coord], aInView2[coord]);
	}
}

template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transformHost(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		linear_access(aOutView, i) = aOperator(linear_access(aInView, i));
	}
}

template <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TPolicy>
void
transformHost(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView1); ++i) {
		linear_access(aOutView, i) = aOperator(linear_access(aInView1, i), linear_access(aInView2, i));
	}
}

template<bool tRunOnDevice>
struct TransformImplementation {

	template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_transform<TInView, TOutView, TFunctor>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aPolicy);
	}

	template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TPolicy>
	static void run(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView1);

		detail::kernel_transform<TInView1, TInView2, TOutView, TFunctor>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView1, aInView2, aOutView, aOperator, aPolicy);
	}
};

template<>
struct TransformImplementation<false> {
	template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformHost(aInView, aOutView, aOperator, aPolicy);
	}

	template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TPolicy>
	static void run(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformHost(aInView1, aInView1, aOutView, aOperator, aPolicy);
	}
};


}//namespace detail

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
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, aPolicy, aCudaStream);
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
		is_device_view<TInView1>::value && is_device_view<TInView2>::value && is_device_view<TOutView>::value>::run(aInView1, aInView2, aOutView, aOperator, aPolicy, aCudaStream);
}

template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor>
void
transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, cudaStream_t aCudaStream = 0)
{
	transform2(aInView1, aInView2, aOutView, aOperator, DefaultTransformPolicy<TInView1, TOutView>(), aCudaStream);
}

/**
 * @}
 **/
/*
template <typename TInView1, typename TInView2, typename TOutView, typename TFunctor>
void
transform(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator)
{
	CUGIP_ASSERT(aInView1.dimensions() == aOutView.dimensions());
	CUGIP_ASSERT(aInView2.dimensions() == aOutView.dimensions());

	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aInView1.dimensions().template get<0>() / blockSize.x + 1), aInView1.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_transform2<TInView1, TInView2, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(aInView1, aInView2, aOutView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_transform2");
}*/

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

/** \ingroup meta_algorithm
 * @{
 **/

template <typename TInView, typename TOutView, typename TFunctor>
void
transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator)
{
	CUGIP_ASSERT(aInView.dimensions() == aOutView.dimensions());

	dim3 blockSize(256, 1, 1);
	dim3 gridSize((aInView.dimensions().template get<0>() / blockSize.x + 1), aInView.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_transform_position<TInView, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(aInView, aOutView, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_transform_position");
}

/**
 * @}
 **/

//*************************************************************************************************************
template<typename TView1, typename TView2>
struct DefaultTransformLocatorPolicy {
	typedef border_handling_repeat_t BorderHandling;

	static constexpr bool cPreload = false;

	static CUGIP_DECL_HYBRID dim3 blockSize()
	{
		return detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	}

	static CUGIP_DECL_HYBRID dim3 gridSize(const TView1 &aView)
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
};

/*template<typename TView, typename TRadius>
struct PreloadingTransformLocatorPolicy {
	typedef border_handling_repeat_t BorderHandling;
	static constexpr int cDimension = dimension<TView>::value;

	typedef detail::defaultBlockSize<cDimension>::type BlockSize;
	typedef detail::defaultBlockSize<cDimension>::type PreloadedBlockSize;

	static constexpr bool cPreload = true;

	static CUGIP_DECL_HYBRID region<cDimension> regionForBlock()
	{

	}

	static CUGIP_DECL_HYBRID dim3 blockSize()
	{
		return detail::defaultBlockDimForDimension<dimension<TView1>::value>();
	}

	static CUGIP_DECL_HYBRID dim3 gridSize(const TView1 &aView)
	{
		return detail::defaultGridSizeForBlockDim(aView.dimensions(), blockSize());
	}
};*/


namespace detail {

template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform_locator(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy)
{
	typename TInView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	typename TInView::extents_t extents = aInView.dimensions();

	if (coord < extents) {
		aOutView[coord] = aOperator(aInView.template locator<typename TPolicy::BorderHandling>(coord));
	}
}


template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
void
transformLocatorHost(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy)
{
	for (int i = 0; i < elementCount(aInView); ++i) {
		linear_access(aOutView, i) = aOperator(aInView.template locator<typename TPolicy::BorderHandling>(index_from_linear_access_index(aInView, i)));
	}
}

template<bool tRunOnDevice>
struct TransformLocatorImplementation {
	template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
	static typename std::enable_if<!TPolicy::cPreload, int>::type
	run(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		// TODO - do this only in code processed by nvcc
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_transform_locator<TInView, TOutView, TOperator, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aPolicy);
		return 0;
	}

	/*template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
	static typename std::enable_if<TPolicy::cPreload, int>::type
	run(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		// TODO - do this only in code processed by nvcc
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_transform_locator<TInView, TOutView, TOperator, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aPolicy);
		return 0;
	}*/
};

template<>
struct TransformLocatorImplementation<false> {

	template <typename TInView, typename TOutView, typename TOperator, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TOperator aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		detail::transformLocatorHost(aInView, aOutView, aOperator, aPolicy);
	}
};

}//namespace detail

/** \ingroup meta_algorithm
 * @{
 **/

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
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, DefaultTransformLocatorPolicy<TInView, TOutView>(), aCudaStream);
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
		is_device_view<TInView>::value && is_device_view<TOutView>::value>::run(aInView, aOutView, aOperator, aPolicy, aCudaStream);
}
/**
 * @}
 **/


}//namespace cugip
