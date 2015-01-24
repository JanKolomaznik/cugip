#pragma once

#include <cugip/math.hpp>
#include <cugip/traits.hpp>
#include <cugip/utils.hpp>
#include <cugip/transform.hpp>
#include <cugip/filter.hpp>
#include <cugip/device_flag.hpp>
#include <cugip/kernel_execution_utils.hpp>
#include <cugip/access_utils.hpp>

#include <cugip/neighborhood.hpp>

#include <cugip/static_memory_block.hpp>


#define BLOCK_NLOPT	12
#define BLOCK_WEIGHTS	8

namespace cugip {

namespace detail {

template<typename TMemoryBlock, typename TCoordinates, typename TLocator, typename TParameters>
CUGIP_DECL_DEVICE void
loadData(TMemoryBlock &aData, TLocator aLocator, const TCoordinates &aPaddedCoords, const TCoordinates &aLocalCoords, TParameters aParameters)
{
	typename TMemoryBlock::extents_t extents = TMemoryBlock::dimensions();
	TCoordinates blockExtents(blockDim.x, blockDim.y, blockDim.z);

	TCoordinates counts;
	for (int i = 0; i < 3; ++i) {
		counts[i] = (extents[i] + blockExtents[i] - 1) / blockExtents[i];
	}

	for (int k = 0; k < counts[2]; ++k) {
		TCoordinates offset(0, 0, k * blockExtents[2]);
		for (int j = 0; j < counts[1]; ++j) {
			offset[1] = j * blockExtents[1];
			for (int i = 0; i < counts[0]; ++i) {
				offset[0] = i * blockExtents[0];
				TCoordinates coords = aLocalCoords + offset;
				if (coords < extents) {
					aData[coords] = aLocator[coords];
				}
			}
		}
	}
}

template<int tPatchRadius>
struct compute_weight
{
	template <typename TMemoryBlock, typename TCoordinates>
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
	}
};



//tPatchRadius <= 2 int tSearchRadius <= 4
template <typename TInImageView, typename TOutImageView, typename TParameters>
CUGIP_GLOBAL void
kernel_nonlocal_means(TInImageView aIn, TOutImageView aOut, TParameters aParameters, simple_vector<int, 3> aOffset)
{
	const int cBorder = TParameters::patch_radius + TParameters::search_radius;
	const int cPatchRadius = TParameters::patch_radius;
	const int cSearchRadius = TParameters::search_radius;
	typedef typename TInImageView::value_type value_type;
	typedef size_traits_3d<8 + 2*(cBorder), 8 + 2*(cBorder), 4 + 2*(cBorder)> shared_data_size;
	typedef static_memory_block<value_type, shared_data_size> MemoryBlock;

	typedef simple_vector<int, 3> coord_t;

	typename TInImageView::extents_t extents = aIn.dimensions();
	coord_t coords = coords_from_block_dim<coord_t>() + aOffset;
	coord_t blockCornerCoords = corner_coords_from_block_dim<coord_t>() + aOffset;
	coord_t localCoords(threadIdx.x, threadIdx.y, threadIdx.z);
	coord_t paddingOffset = coord_t::fill(cBorder);
	coord_t paddedCoords = localCoords + paddingOffset;
	coord_t searchedBlockCornerCoords = blockCornerCoords - paddingOffset;

	CUGIP_SHARED MemoryBlock data;
	loadData(data, aIn.template locator<border_handling_mirror_t>(searchedBlockCornerCoords), paddedCoords, localCoords, aParameters);
	__syncthreads();

	bool use = !(threadIdx.x || threadIdx.y || threadIdx.z);
	if (coords < extents) {
		float acc = 0;
		float value = 0;
		for(int k = -cSearchRadius; k <= cSearchRadius; ++k) {
			/*if (use) {
				printf("AACCC %d\n", k);
			}*/

			for(int j = -cSearchRadius; j <= cSearchRadius; ++j) {
				for(int i = -cSearchRadius; i <= cSearchRadius; ++i) {

					coord_t offset(i, j, k);
					coord_t sampleCoords = paddedCoords + offset;
					float weight = compute_weight<TParameters::patch_radius>::run(data, paddedCoords, sampleCoords, aParameters.variance);
					value += weight * data[sampleCoords];
					acc += weight;
				}
			}
		}
		/*if (use) {
			printf("AA %d %f %f\n", cSearchRadius, acc, value);
		}*/
		aOut[coords] = value / acc;
		/*if (!(threadIdx.x || threadIdx.y || threadIdx.z)) {
			printf("AAA %d %d %d %f %f\n", coords[0], coords[1], coords[2], aIn[coords], aOut[coords]);
		}*/
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
	while (detail::subvolume_too_big(blockSize, subvolume, aParameters)) {
		int maxIndex = 0;
		for (int i = 1; i < 3; ++i) {
			if (subvolume[i] > subvolume[maxIndex]) {
				maxIndex = i;
			}
		}

		subvolume[maxIndex] = (subvolume[maxIndex] + 1) / 2;
		subvolumeCount[maxIndex] *= 2;
	}
	//int count = multiply(subvolumeCount);
	//int counter = 0;
	for (int k = 0; k < subvolumeCount[2]; ++k) {
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
	}

}

}//namespace cugip
