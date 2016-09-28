#pragma once

#include <cugip/subview.hpp>
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/transform.hpp>
#include <cugip/timers.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/kernel_execution_utils.hpp>
#include <cugip/access_utils.hpp>

#include <cugip/neighborhood.hpp>
#include <cugip/neighborhood_accessor.hpp>

#include <cugip/detail/shared_memory.hpp>



namespace cugip {

namespace detail {

template<int tPatchRadius>
struct compute_weight
{
	template <typename TLocator>
	CUGIP_DECL_DEVICE static float
	run(TLocator aOrigin, TLocator aPatchCenter, float aVariance)
	{
		float weight = 0;
		//for_each_neighbor(simple_vector<int, dimension<TLocator>::value>(tPatchRadius, FillFlag()),
		for_each_in_radius<tPatchRadius>(
			[&](simple_vector<int, dimension<TLocator>::value> aOffset){
				weight += sqr(aOrigin[aOffset] - aPatchCenter[aOffset]);
			});
		return __expf(-weight / aVariance);
	}

	template <typename TLocator>
	CUGIP_DECL_DEVICE static void
	run(TLocator aOrigin1, TLocator aPatchCenter1, float &weight1, TLocator aOrigin2, TLocator aPatchCenter2, float &weight2, float aVariance)
	{
		weight1 = 0;
		weight2 = 0;
		for_each_in_radius<tPatchRadius>(
			[&](simple_vector<int, dimension<TLocator>::value> aOffset){
				weight1 += sqr(aOrigin1[aOffset] - aPatchCenter1[aOffset]);
				weight2 += sqr(aOrigin2[aOffset] - aPatchCenter2[aOffset]);
			});
		weight1 = __expf(-weight1 / aVariance);
		weight2 = __expf(-weight2 / aVariance);
	}
};

/*template<typename DataView, typename TParameters>
struct ProcessPatch
{
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>>
	void operator()()
	{
		auto loc = create_locator<
				DataView,
				BorderHandlingTraits<border_handling_enum::NONE>>(dataView, preloadCoords + aCoord);
		auto weight = compute_weight<TParameters::patch_radius>::run(originLocator, loc, aParameters.variance);
		acc += weight;
		value += weight * loc.get();
	}
	(dataView, preloadCoords + aCoord);

};*/

template <typename TInImageView, typename TOutImageView, typename TParameters>
CUGIP_GLOBAL void
kernel_nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	constexpr int cDimension = dimension<TInImageView>::value;
	constexpr int cBorder = TParameters::patch_radius + TParameters::search_radius;
	typedef StaticSize<2*cBorder + 16, 2*cBorder + 8, 2*cBorder + 8> Size;
	__shared__ cugip::detail::SharedMemory<int, Size> buffer;


	auto dataView = buffer.view();
	typedef decltype(dataView) DataView;
	auto coords = mapBlockIdxAndThreadIdxToViewCoordinates<cDimension>();
	auto extents = aIn.dimensions();
	auto searchRadius = simple_vector<int, cDimension>(TParameters::search_radius, FillFlag());
	auto border = simple_vector<int, cDimension>(cBorder, FillFlag());
	auto corner = mapBlockIdxToViewCoordinates<cDimension>() - border;
	auto preloadCoords = coords - corner;// current element coords in the preload buffer
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	Locator originLocator(dataView, preloadCoords);

	buffer.load(aIn, corner);
	__syncthreads();

	float acc = 0;
	float value = 0;
	if (coords < extents) {
		for_each_in_radius<TParameters::search_radius>(
		//for_each_neighbor(simple_vector<int, cDimension>(TParameters::search_radius, FillFlag()),
			[&](const simple_vector<int, cDimension> &aCoord){
				Locator loc(dataView, preloadCoords + aCoord);
				auto weight = compute_weight<TParameters::patch_radius>::run(originLocator, loc, aParameters.variance);
				acc += weight;
				value += weight * loc.get();
			});
		aOut[coords] = value / acc;
	}
}

template <typename TInImageView, typename TOutImageView, typename TParameters>
CUGIP_GLOBAL void
kernel_nonlocal_means3(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	constexpr int cDimension = dimension<TInImageView>::value;
	constexpr int cBorder = TParameters::patch_radius + TParameters::search_radius;
	typedef StaticSize<2*cBorder + 16, 2*cBorder + 8, 2*cBorder + 8> Size;
	__shared__ cugip::detail::SharedMemory<int, Size> buffer;
	const auto searchRadius = simple_vector<int, cDimension>(TParameters::search_radius, FillFlag());
	const auto border = simple_vector<int, cDimension>(cBorder, FillFlag());


	auto dataView = buffer.view();
	typedef decltype(dataView) DataView;
	auto corner = mapBlockIdxToViewCoordinates<cDimension>();
	corner[2] *= 2;
	auto coords1 = corner + currentThreadIndex();
	auto coords2 = coords1;
	coords2[2] += 4;
	auto extents = aIn.dimensions();
	corner -= border;
	auto preloadCoords1 = coords1 - corner;// current element coords in the preload buffer
	auto preloadCoords2 = coords2 - corner;//preloadCoords1;
	//preloadCoords2[2] += 4;
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	Locator originLocator1(dataView, preloadCoords1);
	Locator originLocator2(dataView, preloadCoords2);

	buffer.load(aIn, corner);
	__syncthreads();

	float acc1 = 0;
	float value1 = 0;
	float acc2 = 0;
	float value2 = 0;
	if (coords1 < extents) {
		for_each_in_radius2<TParameters::search_radius>(
			[&](const simple_vector<int, cDimension> &aCoord){
				Locator loc1(dataView, preloadCoords1 + aCoord);
				Locator loc2(dataView, preloadCoords2 + aCoord);
				float weight1 = 0.0f;
				float weight2 = 0.0f;
				compute_weight<TParameters::patch_radius>::run(originLocator1, loc1, weight1, originLocator2, loc2, weight2, aParameters.variance);
				acc1 += weight1;
				value1 += weight1 * loc1.get();
				acc2 += weight2;
				value2 += weight2 * loc2.get();
			});
		aOut[coords1] = value1 / acc1;
		if (coords2 < extents) {
			aOut[coords2] = value2 / acc2;
		}
	}
}

template <typename TInImageView, typename TOutImageView, typename TParameters/*, typename TBlockSize*/, int tStepCount>
CUGIP_GLOBAL void
kernel_nonlocal_means2(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	constexpr int cDimension = dimension<TInImageView>::value;
	constexpr int cBorder = TParameters::patch_radius + TParameters::search_radius;
	constexpr int cLayerSize = 8;
	typedef StaticSize<2*cBorder + 16, 2*cBorder + 8, 2*cBorder + cLayerSize> Size;
	//typedef StaticSize<2*cBorder + 8, 2*cBorder + 8, 2*cBorder + 4> Size;
	__shared__ cugip::detail::SharedMemory<int, Size> buffer;


	auto dataView = buffer.view();
	typedef decltype(dataView) DataView;
	typedef image_locator<DataView, BorderHandlingTraits<border_handling_enum::NONE>> Locator;
	auto coords = mapBlockIdxAndThreadIdxToViewCoordinates<cDimension>();
	auto extents = aIn.dimensions();
	auto searchRadius = simple_vector<int, cDimension>(TParameters::search_radius, FillFlag());
	auto border = simple_vector<int, cDimension>(cBorder, FillFlag());
	auto corner = mapBlockIdxToViewCoordinates<cDimension>() - border;

	auto preloadCoords = coords - corner;// current element coords in the preload buffer
	Locator originLocator(dataView, preloadCoords);

	buffer.load(aIn, corner);
	__syncthreads();
	for(int i = 0; i < tStepCount; ++i) {
		if (coords < extents) {
			float acc = 0;
			float value = 0;
			for_each_neighbor(
			searchRadius,
			[&](const simple_vector<int, cDimension> &aCoord){
				Locator loc(dataView, preloadCoords + aCoord);
				auto weight = compute_weight<TParameters::patch_radius>::run(originLocator, loc, aParameters.variance);
				acc += weight;
				value += weight * loc.get();
			});
			aOut[coords] = value / acc;
		}
		if (tStepCount + 1 == tStepCount) {
			break;
		}
		__syncthreads();
		coords[2] += cLayerSize;
		corner[2] += cLayerSize;
		buffer.shift_and_load(aIn, corner, cLayerSize);
		__syncthreads();
	}
}

template<typename TParameters>
bool
subvolume_too_big(dim3 aBlockSize, const simple_vector<int, 3> &aGridSize, TParameters aParameters)
{
	int blockCount = multiply(aGridSize);
	return blockCount > (16 * 16 * 8);
}

} // namespace detail

template <int tPatchRadius, int tSearchRadius>
struct nl_means_parameters
{
	nl_means_parameters(float aVariance = 10.0) :
		variance(aVariance)
	{}

	static const int patch_radius = tPatchRadius;
	static const int search_radius = tSearchRadius;
	float variance;
};


template <typename TInImageView, typename TOutImageView, typename TParameters>
void
nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters)
{
	typedef simple_vector<int, 3> coord_t;

	dim3 blockSize(16, 8, 8);
	dim3 gridSize = detail::defaultGridSizeForBlockDim(aIn.dimensions(), blockSize);
	//gridSize.z = (gridSize.z + 1) / 2;
	//coord_t gridSize = compute_grid_size(blockSize, aIn.dimensions());
	//coord_t subvolume = gridSize;
	//coord_t subvolumeCount(1, 1, 1);

	cugip::AggregatingTimerSet<1> timers;
	{
		auto interval = timers.start<0>(0);
		/*cugip::detail::kernel_nonlocal_means2<TInImageView, TOutImageView, TParameters, 4>
			<<<gridSize, blockSize>>>(aIn, aOut, aParameters);*/
		D_PRINT("Variance " << aParameters.variance);
		cugip::detail::kernel_nonlocal_means<TInImageView, TOutImageView, TParameters>
			<<<gridSize, blockSize>>>(aIn, aOut, aParameters);
		/*cugip::detail::kernel_nonlocal_means3<TInImageView, TOutImageView, TParameters>
			<<<gridSize, blockSize>>>(aIn, aOut, aParameters);*/
		CUGIP_CHECK_RESULT(cudaThreadSynchronize());
	}
	cudaDeviceSynchronize();
	std::cout << timers.createCompactReport({"NLM"});

	/*while (detail::subvolume_too_big(blockSize, subvolume, aParameters)) {
		int maxIndex = 0;
		for (int i = 1; i < 3; ++i) {
			if (subvolume[i] > subvolume[maxIndex]) {
				maxIndex = i;
			}
		}

		subvolume[maxIndex] = (subvolume[maxIndex] + 1) / 2;
		subvolumeCount[maxIndex] *= 2;
	}*/
	//int count = multiply(subvolumeCount);
	//int counter = 0;
	/*for (int k = 0; k < subvolumeCount[2]; ++k) {
		for (int j = 0; j < subvolumeCount[1]; ++j) {
			for (int i = 0; i < subvolumeCount[0]; ++i) {
				//TODO - last subvolume can be smaller
				coord_t offset(i * blockSize.x * subvolume[0], j * blockSize.y * subvolume[1], k * blockSize.z * subvolume[2]);
				dim3 currentGridSize(subvolume[0], subvolume[1], subvolume[2]);
				D_PRINT("Executing kernel_nonlocal_means: blockSize = "
						<< blockSize
						<< "; gridSize = "
						<< currentGridSize
						<< "; offset = "
						<< offset
				       );
				cugip::detail::kernel_nonlocal_means<TInImageView, TOutImageView, TParameters>
					<<<currentGridSize, blockSize>>>(aIn, aOut, aParameters, offset);
				cudaThreadSynchronize();
				//++counter;
				//D_PRINT("=== " << (float(counter) / count * 100) << "% finished.");
			}
		}
	}*/

}

}//namespace cugip
