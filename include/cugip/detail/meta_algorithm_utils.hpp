#pragma once

#include <cugip/math.hpp>
#include <cugip/detail/shared_memory.hpp>


namespace cugip {
namespace detail {

template<int tDimension>
CUGIP_DECL_DEVICE simple_vector<int, tDimension>
cornerFromBlockIndex(const simple_vector<int, tDimension> &aExtents, int aIdx)
{
	auto block = blockDimensions<tDimension>();

	return product(block, index_from_linear_access_index(div_up(aExtents, block), aIdx));
}

template<bool tRunOnDevice>
struct UniversalRegionCoverImplementation;

#if defined(__CUDACC__)
template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
CUGIP_GLOBAL void
universal_region_cover_kernel(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
{
	const auto blockCount = gridSize();
	const auto extents = aFirstView.dimensions();
	const auto requiredBlockCount = multiply(div_up(extents, blockDimensions<dimension<TFirstView>::value>()));
	auto blockIndex = blockOrderFromIndex();

	auto threadIndex = currentThreadIndexDim<dimension<TFirstView>::value>();
	while (blockIndex < requiredBlockCount) {
		auto corner = cornerFromBlockIndex(extents, blockIndex);
		auto coord = corner + threadIndex;
		if (coord < extents) {
			aOperator(coord, aFirstView, aViews...);
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}

template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
CUGIP_GLOBAL void
universal_region_cover_preload_kernel(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
{
	const auto blockCount = gridSize();
	const auto extents = aFirstView.dimensions();
	typedef typename TPolicy::RegionSize Size;
	__shared__ cugip::detail::SharedMemory<typename TFirstView::value_type, Size> buffer;
	//__shared__ cugip::detail::SharedMemory<typename TPolicy::shmem_value_type, Size> buffer;

	const auto requiredBlockCount = multiply(div_up(extents, blockDimensions<dimension<TFirstView>::value>()));
	auto blockIndex = blockOrderFromIndex();

	auto loadedRegion = aPolicy.regionForBlock();

	auto threadIndex = currentThreadIndexDim<dimension<TFirstView>::value>();
	while (blockIndex < requiredBlockCount) {
		auto corner = cornerFromBlockIndex(extents, blockIndex);
		auto coord = corner + threadIndex;

		auto preloadCoords = coord - corner;// current element coords in the preload buffer


		auto dataView = buffer.view();
		typedef decltype(dataView) DataView;
		buffer.load(aPolicy.GetViewForLoading(aFirstView, aViews...), corner);

		__syncthreads();

		if (coord < extents) {
			aOperator(preloadCoords, dataView, coord, aFirstView, aViews...);
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}


#endif // defined(__CUDACC__)

template<>
struct UniversalRegionCoverImplementation<true> {

	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
	static typename std::enable_if<!TPolicy::cPreload, int>::type
	run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
#		if defined(__CUDACC__)
		// TODO assert on region sizes
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aFirstView));

		detail::universal_region_cover_kernel<TFunctor, TPolicy, TFirstView, TViews...>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aOperator, aPolicy, aFirstView, aViews...);
#		endif // defined(__CUDACC__)
		return 0;
	}

	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
	static typename std::enable_if<TPolicy::cPreload, int>::type
	run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
#		if defined(__CUDACC__)
		// TODO assert on region sizes
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aFirstView));

		detail::universal_region_cover_preload_kernel<TFunctor, TPolicy, TFirstView, TViews...>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aOperator, aPolicy, aFirstView, aViews...);
#		endif // defined(__CUDACC__)
		return 0;
	}
};


template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
void
universal_region_cover_host(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
{
	for (int i = 0; i < elementCount(aFirstView); ++i) {
		auto coord = index_from_linear_access_index(aFirstView);
		aOperator(coord, coord, aFirstView, aViews...);
	}
}

template<>
struct UniversalRegionCoverImplementation<false> {
	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
	static void run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
		// TODO assert on region sizes
		detail::universal_region_cover_host<TFunctor, TPolicy, TFirstView, TViews...>
			(aOperator, aPolicy, aFirstView, aViews...);
	}
};


} // namespace detail
} // namespace cugip
