#pragma once

namespace cugip {
namespace detail {

/*
template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	const auto blockCount = gridSize();
	const auto extents = aView.dimensions();
	const auto requiredBlockCount = multiply(div_up(extents, blockDimensions<dimension<TView>::value>()));
	auto blockIndex = blockOrderFromIndex();

	auto threadIndex = currentThreadIndexDim<dimension<TView>::value>();
	while (blockIndex < requiredBlockCount) {
		auto corner = cornerFromBlockIndex(extents, blockIndex);
		auto coord = corner + threadIndex;
		if (coord < extents) {
			aAssignOperation(aOutView[coord], aOperator(aInView[coord]));
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}

template <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform2(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	const auto blockCount = gridSize();
	const auto extents = aView.dimensions();
	const auto requiredBlockCount = multiply(div_up(extents, blockDimensions<dimension<TView>::value>()));
	auto blockIndex = blockOrderFromIndex();

	auto threadIndex = currentThreadIndexDim<dimension<TView>::value>();
	while (blockIndex < requiredBlockCount) {
		auto corner = cornerFromBlockIndex(extents, blockIndex);
		auto coord = corner + threadIndex;
		if (coord < extents) {
			aAssignOperation(aOutView[coord], aOperator(aInView1[coord], aInView2[coord]));
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}

template<>
struct TransformImplementation<true> {

	template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aInView));

		detail::kernel_transform<TInView, TOutView, TFunctor, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
	}

	template  <typename TInView1, typename TInView2, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView1 aInView1, TInView2 aInView2, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView1);

		detail::kernel_transform2<TInView1, TInView2, TOutView, TFunctor, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView1, aInView2, aOutView, aOperator, aAssignOperation, aPolicy);
	}
};

template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform_position(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	const auto blockCount = gridSize();
	const auto extents = aView.dimensions();
	const auto requiredBlockCount = multiply(div_up(extents, blockDimensions<dimension<TView>::value>()));
	auto blockIndex = blockOrderFromIndex();

	auto threadIndex = currentThreadIndexDim<dimension<TView>::value>();
	while (blockIndex < requiredBlockCount) {
		auto corner = cornerFromBlockIndex(extents, blockIndex);
		auto coord = corner + threadIndex;
		if (coord < extents) {
			aAssignOperation(aOutView[coord], aOperator(aInView[coord], coord));
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}

template<>
struct TransformPositionImplementation<true> {
	template <typename TInView, typename TOutView, typename TFunctor, typename TAssignOperation, typename TPolicy>
	static void run(TInView aInView, TOutView aOutView, TFunctor aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aInView));

		detail::kernel_transform_position<TInView, TOutView, TFunctor, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
	}
};

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform_locator(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{
	typename TInView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	typename TInView::extents_t extents = aInView.dimensions();

	if (coord < extents) {
		aAssignOperation(aOutView[coord], aOperator(create_locator<TInView, typename TPolicy::BorderHandling>(aInView, coord)));
	}
}

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_transform_locator_preload(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy)
{

	typename TInView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	typename TInView::extents_t extents = aInView.dimensions();
	typedef typename TPolicy::RegionSize Size;
	__shared__ cugip::detail::SharedMemory<typename TInView::value_type, Size> buffer;

	auto loadedRegion = aPolicy.regionForBlock();
	auto corner = mapBlockIdxToViewCoordinates<dimension<TInView>::value>() + loadedRegion.corner;
	auto preloadCoords = coord - corner;// current element coords in the preload buffer

	auto dataView = buffer.view();
	typedef decltype(dataView) DataView;
	buffer.load(aInView, corner);
	__syncthreads();
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	Locator itemLocator(dataView, preloadCoords);
	if (coord < extents) {
		aAssignOperation(aOutView[coord], aOperator(itemLocator));
	}
}

template<>
struct TransformLocatorImplementation<true> {
	template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
	static typename std::enable_if<!TPolicy::cPreload, int>::type
	run(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		// TODO - do this only in code processed by nvcc
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aInView));

		detail::kernel_transform_locator<TInView, TOutView, TOperator, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
		return 0;
	}

	template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
	static typename std::enable_if<TPolicy::cPreload, int>::type
	run(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		// TODO - do this only in code processed by nvcc
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aInView));

		detail::kernel_transform_locator_preload<TInView, TOutView, TOperator, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
		return 0;
	}
};
*/

} // namespace detail
} // namespace cugip
