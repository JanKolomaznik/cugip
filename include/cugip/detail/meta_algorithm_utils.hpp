#pragma once

#include <cugip/math.hpp>
#include <cugip/access_utils.hpp>

#if defined(__CUDACC__)
	#include <cugip/detail/shared_memory.hpp>
#endif //defined(__CUDACC__)


namespace cugip {
namespace detail {

#if defined(__CUDACC__)
template<int tDimension>
CUGIP_DECL_DEVICE simple_vector<int, tDimension>
cornerFromTileIndex(const simple_vector<int, tDimension> &aExtents, const simple_vector<int, tDimension> &aTileSize, int aIdx)
{
	return product(aTileSize, index_from_linear_access_index(div_up(aExtents, aTileSize), aIdx));
}

template<int tDimension>
CUGIP_DECL_DEVICE simple_vector<int, tDimension>
cornerFromBlockIndex(const simple_vector<int, tDimension> &aExtents, int aIdx)
{
	auto block = blockDimensions<tDimension>();
	return cornerFromTileIndex(aExtents, block, aIdx);
}
#endif //defined(__CUDACC__)

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

		auto preloadCoords = coord - corner - aPolicy.corner1();// current element coords in the preload buffer


		auto dataView = buffer.view();
		typedef decltype(dataView) DataView;
		buffer.load(aPolicy.GetViewForLoading(aFirstView, aViews...), corner + aPolicy.corner1());

		__syncthreads();

		if (coord < extents) {
			// if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y > 0) {
			// 	auto v = vect3i_t(0, 1, 0);
			// 	printf("[%d, %d, %d], [%d, %d, %d], %d -> %d\n", preloadCoords[0], preloadCoords[1], preloadCoords[2], coord[0], coord[1], coord[2], dataView[preloadCoords], aPolicy.GetViewForLoading(aFirstView, aViews...)[coord - v]);
			// }
			aOperator(preloadCoords, dataView, coord, aFirstView, aViews...);
		}
		blockIndex += blockCount;

		// if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y == 0) {
		// 	for (int k = 0; k < dataView.dimensions()[2]; ++k) {
		// 		for (int j = 0; j < dataView.dimensions()[1]; ++j) {
		// 			for (int i = 0; i < dataView.dimensions()[0]; ++i) {
		// 				printf("%d ", dataView[vect3i_t(i, j, k)]);
		// 			}
		// 			printf("\n");
		// 		}
		// 		printf("\n----------------------------------\n");
		// 	}
		// }
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
		auto coord = index_from_linear_access_index(aFirstView, i);
		aOperator(coord, /*coord,*/ aFirstView, aViews...);
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

//************************************************************************************************************
//************************************************************************************************************
//************************************************************************************************************

#if defined(__CUDACC__)

template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
CUGIP_GLOBAL void
tiled_region_cover_preload_kernel(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
{
	const auto blockCount = gridSize();
	const auto extents = aFirstView.dimensions();
	using HaloTileSize = typename TPolicy::HaloTileSize;
	using TileSize = typename TPolicy::TileSize;

	__shared__ cugip::detail::SharedMemory<typename TFirstView::value_type, HaloTileSize> buffer;
	//__shared__ cugip::detail::SharedMemory<typename TPolicy::shmem_value_type, Size> buffer;

	const auto requiredTileCount = multiply(div_up(extents, to_vector(TileSize())));
	auto tileIndex = blockOrderFromIndex();

	//auto loadedRegion = aPolicy.regionForBlock();

	auto threadIndex = currentThreadIndexDim<dimension<TFirstView>::value>();
	while (tileIndex < requiredTileCount) {
		auto tileCorner = cornerFromTileIndex(extents, to_vector(TileSize{}), tileIndex);
		auto haloTileCorner = tileCorner + aPolicy.corner1(); //TODO

		//auto corner = cornerFromBlockIndex(extents, blockIndex);
		//auto coord = corner + threadIndex;

		//auto preloadCoords = coord - corner + aPolicy.corner1();// current element coords in the preload buffer


		auto dataView = buffer.view();
		typedef decltype(dataView) DataView;
		buffer.load(aFirstView, haloTileCorner);

		__syncthreads();

		if (tileCorner < extents) {
			aOperator(tileCorner, haloTileCorner, dataView, aFirstView, aViews...);
		}
		tileIndex += blockCount;

		__syncthreads();
	}
}

#endif // defined(__CUDACC__)

struct TileCoverImplementation {
	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
	static void
	run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
#		if defined(__CUDACC__)
		// TODO assert on region sizes
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aFirstView));

		detail::tiled_region_cover_preload_kernel<TFunctor, TPolicy, TFirstView, TViews...>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aOperator, aPolicy, aFirstView, aViews...);
#		endif // defined(__CUDACC__)
	}
};
//
//
// template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
// void
// universal_region_cover_host(TFunctor aOperator, TPolicy aPolicy, TFirstView aFirstView, TViews... aViews)
// {
// 	for (int i = 0; i < elementCount(aFirstView); ++i) {
// 		auto coord = index_from_linear_access_index(aFirstView);
// 		aOperator(coord, coord, aFirstView, aViews...);
// 	}
// }
//
// template<>
// struct UniversalRegionCoverImplementation<false> {
// 	template <typename TFunctor, typename TPolicy, typename TFirstView, typename... TViews>
// 	static void run(TFunctor aOperator, TPolicy aPolicy, cudaStream_t aCudaStream, TFirstView aFirstView, TViews... aViews) {
// 		// TODO assert on region sizes
// 		detail::universal_region_cover_host<TFunctor, TPolicy, TFirstView, TViews...>
// 			(aOperator, aPolicy, aFirstView, aViews...);
// 	}
// };





} // namespace detail
} // namespace cugip
