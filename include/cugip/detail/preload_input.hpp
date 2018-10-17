#pragma once

#include <cugip/detail/include.hpp>
#include <cugip/meta_algorithm.hpp>
#include <cugip/exception.hpp>
#include <cugip/access_utils.hpp>

template <typename TInView, typename TOutView, typename TOperator, typename TAssignOperation, typename TPolicy>
CUGIP_GLOBAL void
kernel_preload_input(TInView aInView, TOperator aOperator, TPolicy aPolicy)
{
	typename TInView::coord_t coord = mapBlockIdxAndThreadIdxToViewCoordinates<dimension<TInView>::value>();
	typename TInView::extents_t extents = aInView.dimensions();
	typedef typename TPolicy::RegionSize Size;
	__shared__ cugip::detail::SharedMemory<typename TInView::value_type, Size> buffer;

	auto loadedRegion = aPolicy.regionForBlock();
	auto loadCorner = mapBlockIdxToViewCoordinates<dimension<TInView>::value>() + loadedRegion.corner;
	auto preloadCoords = coord - loadCorner;// current element coords in the preload buffer

	//auto dataView = makeDeviceImageView(&(buffer.get(Int3())), to_vector(Size()));
	auto dataView = buffer.view();
	typedef decltype(dataView) DataView;
	buffer.load(aInView, load_corner);
	__syncthreads();


	/*typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	Locator itemLocator(dataView, preloadCoords);*/
	if (coord < extents) {
		aOperator(dataView, coord, preloadCoords);
	}
}

struct ProcessWithPreloadIntoSM {
	template <typename TInView, typename TOperator, typename TPolicy>
	static void
	run(TInView aInView, TOutView aOutView, TOperator aOperator, TAssignOperation aAssignOperation, TPolicy aPolicy, cudaStream_t aCudaStream) {
		// TODO - map threads for bigger block
		dim3 blockSize = aPolicy.blockSize();
		dim3 gridSize = aPolicy.gridSize(active_region(aInView));

		detail::kernel_transform_locator_preload<TInView, TOutView, TOperator, TAssignOperation, TPolicy>
			<<<gridSize, blockSize, 0, aCudaStream>>>(aInView, aOutView, aOperator, aAssignOperation, aPolicy);
		return 0;
	}
};

struct F
{
	template<typename TView, TCoordinates>
	void operator()(TView aView, TCoordinates aCoordinates, TCoordinates aPreloadedCoordinates)
	{

	}

};
