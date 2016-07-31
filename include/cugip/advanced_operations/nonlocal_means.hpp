#pragma once

#include <cugip/subview.hpp>
#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/kernel_execution_utils.hpp>
#include <cugip/access_utils.hpp>

#include <cugip/neighborhood.hpp>

#include <cugip/detail/shared_memory.hpp>



namespace cugip {

namespace detail {

template<int tPatchRadius>
struct compute_weight
{
	/*template <typename TMemoryBlock, typename TCoordinates>
	CUGIP_DECL_DEVICE static float
	run(const TMemoryBlock &aData, const TCoordinates &aCoord1, const TCoordinates &aCoord2, float aVariance)
	{
		typedef simple_vector<int, 3> coord_t;
		float weight = 0;
		for(int k = -tPatchRadius; k <= tPatchRadius; ++k) {
			for(int j = -tPatchRadius; j <= tPatchRadius; ++j) {
				for(int i = -tPatchRadius; i <= tPatchRadius; ++i) {
					coord_t offset(i, j, k);
					float diff = aData[aCoord1 + offset] - aData[aCoord2 + offset];
					weight += diff*diff;
				}
			}
		}
		return exp(-weight / aVariance);
	}*/
	template <typename TLocator>
	CUGIP_DECL_DEVICE static float
	run(TLocator aOrigin, TLocator aPatchCenter, float aVariance)
	{
		float weight = 0;
		for_each_neighbor(
			simple_vector<int, dimension<TLocator>::value>::fill(tPatchRadius),
			[&](simple_vector<int, dimension<TLocator>::value> aOffset){
				weight += sqr(aOrigin[aOffset] - aPatchCenter[aOffset]);
			});
		/*typedef typename TLocator::diff_t diff_t;

		for(int k = -tPatchRadius; k <= tPatchRadius; ++k) {
			for(int j = -tPatchRadius; j <= tPatchRadius; ++j) {
				for(int i = -tPatchRadius; i <= tPatchRadius; ++i) {
					coord_t offset(i, j, k);
					float diff = aData[aCoord1 + offset] - aData[aCoord2 + offset];
					weight += diff*diff;
				}
			}
		}*/
		return exp(-weight / aVariance);
	}
};


template <typename TInImageView, typename TOutImageView, typename TParameters>
CUGIP_GLOBAL void
kernel_nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters, simple_vector<int, 3> aOffset)
{
	constexpr int cDimension = dimension<TInImageView>::value;
	constexpr int cBorder = TParameters::patch_radius + TParameters::search_radius;
	typedef StaticSize<2*cBorder + 8, 2*cBorder + 8, 2*cBorder + 8> Size;
	__shared__ cugip::detail::SharedMemory<int, Size> buffer;


	auto dataView = buffer.view();
	auto coords = mapBlockIdxAndThreadIdxToViewCoordinates<cDimension>();
	auto extents = aIn.dimensions();
	auto border = simple_vector<int, cDimension>::fill(cBorder);
	auto corner = mapBlockIdxToViewCoordinates<cDimension>() - border;
	auto preloadCoords = coords - corner;
	auto originLocator = create_locator<
				decltype(dataView),
				BorderHandlingTraits<border_handling_enum::NONE>>(dataView, preloadCoords);

	buffer.load(aIn, corner);
	__syncthreads();

	float acc = 0;
	float value = 0;
	if (coords < extents) {
		for_each_neighbor(
			border,
			[&](const simple_vector<int, cDimension> &aCoord){
				auto loc = create_locator<
						decltype(dataView),
						BorderHandlingTraits<border_handling_enum::NONE>>(dataView, preloadCoords + aCoord);
				auto weight = compute_weight<TParameters::patch_radius>::run(originLocator, loc, aParameters.variance);
				acc += weight;
				value += weight * loc.get();
			});
		aOut[coords] = value / acc;
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

	dim3 blockSize(8, 8, 4);
	coord_t gridSize = compute_grid_size(blockSize, aIn.dimensions());
	coord_t subvolume = gridSize;
	coord_t subvolumeCount(1, 1, 1);


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
