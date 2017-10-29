#pragma once

#include <cugip/region.hpp>

namespace cugip {
namespace detail {

template<int tDimension>
CUGIP_DECL_DEVICE simple_vector<int, tDimension>
cornerFromBlockIndex(const simple_vector<int, tDimension> &aExtents, int aIdx)
{
	auto block = blockDimensions<tDimension>();

	return product(block, index_from_linear_access_index(div_up(aExtents, block), aIdx));
}

template <typename TView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_for_each(TView aView, TFunctor aOperator, TPolicy aPolicy)
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
			aOperator(aView[coord]);
		}
		blockIndex += blockCount;
		__syncthreads();
	}
}


/*template <typename TView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_for_each(TView aView, TFunctor aOperator, TPolicy aPolicy)
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aOperator(aView[coord]);
	}
}*/

template<>
struct ForEachImplementation<true> {

	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		CUGIP_DFORMAT("Execution of 'kernel_for_each' block: %1%, grid %2%", blockSize, gridSize);
		detail::kernel_for_each<TInView, TFunctor, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOperator, aPolicy);
	}
};


template <typename TView, typename TFunctor, typename TPolicy>
CUGIP_GLOBAL void
kernel_for_each_position(TView aView, TFunctor aOperator, TPolicy aPolicy)
{
	typename TView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TView>::value>();
	typename TView::extents_t extents = aView.dimensions();

	if (coord < extents) {
		aOperator(aView[coord], coord);
	}
}

template<>
struct ForEachPositionImplementation<true> {

	template <typename TInView, typename TFunctor, typename TPolicy>
	static void run(TInView aInView, TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream) {
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(aInView);

		detail::kernel_for_each_position<TInView, TFunctor, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOperator, aPolicy);
	}
};

}//namespace detail
}//namespace cugip
