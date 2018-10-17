#pragma once

namespace cugip {
namespace detail {

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
#endif // defined(__CUDACC__)

template<>
struct UniversalRegionCoverImplementation<true> {

	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
	static void run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
#		if defined(__CUDACC__)
		// TODO assert on region sizes
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aFirstView));

		detail::universal_region_cover_kernel<TFunctor, TPolicy, TFirstView, TViews...>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aOperator, aPolicy, aFirstView, aViews...);
#		endif // defined(__CUDACC__)
	}
};


template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
void
universal_region_cover_host(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
{
	for (int i = 0; i < elementCount(aFirstView); ++i) {
		auto coord = index_from_linear_access_index(aFirstView);
		aOperator(coord, aFirstView, aViews...);
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
